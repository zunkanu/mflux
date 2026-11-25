import math
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn as tnn
import torch.nn.functional as F

from mflux.models.common.pixel_head import (
    ConvNeXtBlock2d,
    LatentSpatialHighway,
    PixelHeadBase,
    PixelHeadOps,
    PixelHeadWeightUtil,
    PixelRefineBlock2d,
    SigmaAdaLN,
    WaveletConvBlock2d,
)
from mflux.models.common.vae.vae_decode_options import VAEDecodeOptions


def _to_mx_state(module: tnn.Module) -> dict[str, mx.array]:
    return {k: mx.array(v.detach().float().numpy()) for k, v in module.state_dict().items()}


@pytest.mark.fast
def test_haar_dwt_and_idwt_are_exact_inverses():
    rng = np.random.default_rng(0)
    img = mx.array(rng.standard_normal((1, 8, 12, 4)).astype(np.float32))
    ll, lh, hl, hh = PixelHeadOps.dwt_haar_2d_nhwc(img)
    reconstructed = PixelHeadOps.idwt_haar_2d_nhwc(ll, lh, hl, hh)
    np.testing.assert_allclose(np.asarray(reconstructed), np.asarray(img), rtol=1e-5, atol=1e-5)


@pytest.mark.fast
def test_pixel_shuffle_and_unpatchify_match_pytorch():
    torch.manual_seed(0)
    x_pt = torch.randn(2, 12, 3, 4)
    expected = F.pixel_shuffle(x_pt, 2).movedim(1, -1).numpy()
    actual = np.asarray(PixelHeadOps.pixel_shuffle_nhwc(mx.array(x_pt.movedim(1, -1).numpy()), 2))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    tokens = torch.randn(1, 2, 3, 4 * 4 * 4)
    expected_up = tokens.view(1, 2, 3, 4, 4, 4).permute(0, 1, 3, 2, 4, 5).contiguous().view(1, 8, 12, 4).numpy()
    actual_up = np.asarray(PixelHeadOps.unpatchify_nhwc(mx.array(tokens.numpy()), 4, 4))
    np.testing.assert_allclose(actual_up, expected_up, rtol=1e-5, atol=1e-5)


@pytest.mark.fast
def test_pixel_head_blocks_match_pytorch():
    torch.manual_seed(5)
    x_pt = torch.randn(1, 16, 4, 4)
    x_mx = mx.array(x_pt.movedim(1, -1).numpy())

    # ConvNeXtBlock2d
    class _PtConvNeXt(tnn.Module):
        def __init__(self):
            super().__init__()
            self.dw = tnn.Conv2d(16, 16, 3, padding=1, groups=16)
            self.norm = tnn.LayerNorm(16, eps=1e-6)
            self.fc1 = tnn.Linear(16, 64)
            self.fc2 = tnn.Linear(64, 16)
            self.gamma = tnn.Parameter(torch.full((16,), 0.7))

        def forward(self, x):
            h = self.fc2(F.gelu(self.fc1(self.norm(self.dw(x).movedim(1, -1)))))
            return x + (h * self.gamma).movedim(-1, 1)

    pt_cx = _PtConvNeXt()
    mx_cx = ConvNeXtBlock2d(16, ls_init=0.7)
    PixelHeadWeightUtil.apply_weights(mx_cx, _to_mx_state(pt_cx), dtype=mx.float32)
    np.testing.assert_allclose(
        np.asarray(mx_cx(x_mx)), pt_cx(x_pt).movedim(1, -1).detach().numpy(), rtol=1e-4, atol=1e-4
    )

    # PixelRefineBlock2d
    class _PtRefine(tnn.Module):
        def __init__(self):
            super().__init__()
            self.dw = tnn.Conv2d(16, 16, 3, padding=1, groups=16)
            self.norm = tnn.GroupNorm(8, 16, eps=1e-6)
            self.pw1 = tnn.Conv2d(16, 32, 1)
            self.pw2 = tnn.Conv2d(32, 16, 1)

        def forward(self, x):
            return x + self.pw2(F.silu(self.pw1(self.norm(self.dw(x)))))

    pt_ref = _PtRefine()
    mx_ref = PixelRefineBlock2d(16)
    PixelHeadWeightUtil.apply_weights(mx_ref, _to_mx_state(pt_ref), dtype=mx.float32)
    np.testing.assert_allclose(
        np.asarray(mx_ref(x_mx)), pt_ref(x_pt).movedim(1, -1).detach().numpy(), rtol=1e-4, atol=1e-4
    )

    # WaveletConvBlock2d
    class _PtWavelet(tnn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = tnn.Conv2d(16, 16, 3, padding=1)
            self.conv2 = tnn.Conv2d(16, 16, 3, padding=1)
            self.gamma = tnn.Parameter(torch.full((1, 16, 1, 1), 0.5))

        def forward(self, x):
            return x + self.conv2(F.silu(self.conv1(x))) * self.gamma

    pt_wav = _PtWavelet()
    mx_wav = WaveletConvBlock2d(16, ls_init=0.5)
    PixelHeadWeightUtil.apply_weights(mx_wav, _to_mx_state(pt_wav), dtype=mx.float32)
    np.testing.assert_allclose(
        np.asarray(mx_wav(x_mx)), pt_wav(x_pt).movedim(1, -1).detach().numpy(), rtol=1e-4, atol=1e-4
    )

    # SigmaAdaLN
    class _PtSigma(tnn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = tnn.Sequential(tnn.Linear(32, 32), tnn.SiLU(), tnn.Linear(32, 32), tnn.SiLU())
            self.heads = tnn.ModuleList([tnn.Linear(32, 32)])

        def forward(self, x, sigma):
            log_s = torch.log(torch.tensor([sigma], dtype=torch.float32).clamp(1e-4, 2.0))
            freqs = torch.exp(-math.log(1000.0) * torch.arange(16, dtype=torch.float32) / 15)
            ang = log_s[:, None] * freqs[None, :]
            cond = self.mlp(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))
            scale, shift = self.heads[0](cond).chunk(2, dim=-1)
            return x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

    pt_sig = _PtSigma()
    mx_sig = SigmaAdaLN(embed_dim=32, target_dims=(16,))
    PixelHeadWeightUtil.apply_weights(mx_sig, _to_mx_state(pt_sig), sequential_prefixes=("mlp.",), dtype=mx.float32)
    cond = mx_sig.embed_sigma(0.15, 1, mx.float32)
    np.testing.assert_allclose(
        np.asarray(mx_sig.modulate_nhwc(x_mx, cond, 0)),
        pt_sig(x_pt, 0.15).movedim(1, -1).detach().numpy(),
        rtol=1e-4,
        atol=1e-4,
    )

    # LatentSpatialHighway shapes & finite output
    hw = LatentSpatialHighway(d_lat=8, c_lat16=16, c_lat8=12, c_lat4=8, w8=24, c_4px=16, c_2px=8)
    j8, j4, j2 = hw(mx.zeros((1, 2, 3, 8)))
    assert j8.shape == (1, 4, 6, 24)
    assert j4.shape == (1, 8, 12, 16)
    assert j2.shape == (1, 16, 24, 8)


@pytest.mark.fast
def test_pixel_head_base_tiled_decode_and_scale_validation():
    class _ConstantHead(PixelHeadBase):
        token_pixels = 4
        out_channels = 4
        supports_x2 = False

        def __init__(self):
            super().__init__()
            self._d_feat = 4
            self.feat_proj = SimpleNamespace(weight=mx.zeros((4, 4), dtype=mx.float32))

        def _forward_nhwc(self, feat: mx.array, lat: mx.array, sigma: float = 0.10, scale: int = 1) -> mx.array:
            b, gh, gw, _ = lat.shape
            return mx.ones((b, gh * self.token_pixels * scale, gw * self.token_pixels * scale, 4), dtype=mx.float32)

    head = _ConstantHead()
    feat = mx.zeros((1, 16, 4))
    x0 = mx.zeros((1, 4, 4, 4))
    full = head.decode(feat, x0, tiled=False)
    tiled = head.decode(feat, x0, tiled=True, tile_tokens=3, overlap_tokens=1)
    np.testing.assert_allclose(np.asarray(tiled), np.asarray(full), rtol=1e-5, atol=1e-5)

    with pytest.raises(ValueError, match="1x checkpoint"):
        head.resolve_scale(2)


@pytest.mark.fast
def test_vae_decode_options_apply_is_safe_when_vae_is_none():
    model = SimpleNamespace(vae=None)
    VAEDecodeOptions.apply(SimpleNamespace(vae_full_precision=True), model)
    assert VAEDecodeOptions.generation_parameters(model) == {}
