# Shared NHWC MLX building blocks, Haar wavelet iDWT, tiled decoder, and weight utilities for
# FastFrames-style Direct Pixel Readout Heads (MIT License).
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten


class PixelHeadOps:
    @staticmethod
    def pixel_shuffle_nhwc(x: mx.array, factor: int = 2) -> mx.array:
        # Matches PyTorch F.pixel_shuffle channel ordering [C, r_h, r_w] in NHWC layout.
        b, h, w, c_in = x.shape
        c_out = c_in // (factor * factor)
        x = x.reshape(b, h, w, c_out, factor, factor)
        x = x.transpose(0, 1, 4, 2, 5, 3)
        return x.reshape(b, h * factor, w * factor, c_out)

    @staticmethod
    def unpatchify_nhwc(tokens: mx.array, patch: int, out_channels: int) -> mx.array:
        b, gh, gw, _ = tokens.shape
        x = tokens.reshape(b, gh, gw, patch, patch, out_channels)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(b, gh * patch, gw * patch, out_channels)

    @staticmethod
    def dwt_haar_2d_nhwc(img: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        x00 = img[:, 0::2, 0::2, :]
        x01 = img[:, 0::2, 1::2, :]
        x10 = img[:, 1::2, 0::2, :]
        x11 = img[:, 1::2, 1::2, :]
        ll = 0.25 * (x00 + x01 + x10 + x11)
        lh = 0.25 * ((x00 + x01) - (x10 + x11))
        hl = 0.25 * ((x00 + x10) - (x01 + x11))
        hh = 0.25 * ((x00 + x11) - (x01 + x10))
        return ll, lh, hl, hh

    @staticmethod
    def idwt_haar_2d_nhwc(ll: mx.array, lh: mx.array, hl: mx.array, hh: mx.array) -> mx.array:
        x00 = ll + lh + hl + hh
        x01 = ll + lh - hl - hh
        x10 = ll - lh + hl - hh
        x11 = ll - lh - hl + hh
        b, h, w, c = ll.shape
        top = mx.stack([x00, x01], axis=3)
        bot = mx.stack([x10, x11], axis=3)
        quad = mx.stack([top, bot], axis=2)
        return quad.reshape(b, h * 2, w * 2, c)

    @staticmethod
    def prepare_inputs(feat: mx.array, x0_hat_norm: mx.array, d_feat: int) -> tuple[mx.array, mx.array]:
        if x0_hat_norm.ndim == 5:
            if x0_hat_norm.shape[2] != 1:
                raise ValueError("PixelHead supports single-frame images only.")
            x0_hat_norm = x0_hat_norm[:, :, 0]
        if x0_hat_norm.ndim != 4:
            raise ValueError(f"Expected 4D or 5D x0_hat_norm, got shape {x0_hat_norm.shape}.")
        b, _, gh, gw = x0_hat_norm.shape
        lat_nhwc = x0_hat_norm.transpose(0, 2, 3, 1)
        if feat.ndim == 3:
            feat_nhwc = feat.reshape(b, gh, gw, d_feat)
        elif feat.ndim == 4:
            feat_nhwc = feat
        else:
            raise ValueError(f"Expected 3D or 4D feat, got shape {feat.shape}.")
        return feat_nhwc, lat_nhwc

    @staticmethod
    def tile_starts(total: int, tile_tokens: int, overlap_tokens: int) -> list[int]:
        if total <= tile_tokens:
            return [0]
        stride = max(1, tile_tokens - overlap_tokens)
        starts = list(range(0, total - tile_tokens, stride))
        starts.append(total - tile_tokens)
        return starts

    @staticmethod
    def ramp_1d(length: int, overlap: int, at_start: bool, at_end: bool) -> np.ndarray:
        w = np.ones(length, dtype=np.float32)
        ov = min(overlap, length)
        if not at_start and ov > 0:
            t = np.linspace(0.0, math.pi, ov, dtype=np.float32)
            w[:ov] = 0.5 * (1.0 - np.cos(t))
        if not at_end and ov > 0:
            t = np.linspace(0.0, math.pi, ov, dtype=np.float32)
            w[-ov:] = 0.5 * (1.0 + np.cos(t))
        return w


class ConvNeXtBlock2d(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4, ls_init: float = 1.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.gamma = mx.full((dim,), float(ls_init))

    def __call__(self, x: mx.array) -> mx.array:
        h = self.dw(x)
        h = self.fc2(nn.gelu(self.fc1(self.norm(h))))
        return x + h * self.gamma


class PixelRefineBlock2d(nn.Module):
    def __init__(self, dim: int, expand: int = 2):
        super().__init__()
        hidden = dim * expand
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.GroupNorm(num_groups=min(8, dim), dims=dim, eps=1e-6, pytorch_compatible=True)
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.pw2(nn.silu(self.pw1(self.norm(self.dw(x)))))


class WaveletConvBlock2d(nn.Module):
    def __init__(self, dim: int, ls_init: float = 1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.gamma = mx.full((dim,), float(ls_init))

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv2(nn.silu(self.conv1(x)))
        return x + h * self.gamma


class LatentSpatialHighway(nn.Module):
    def __init__(
        self,
        d_lat: int = 64,
        c_lat16: int = 128,
        c_lat8: int = 96,
        c_lat4: int = 64,
        w8: int = 512,
        c_4px: int = 256,
        c_2px: int = 64,
    ):
        super().__init__()
        self.c_lat16 = int(c_lat16)
        self.c_lat8 = int(c_lat8)
        self.c_lat4 = int(c_lat4)
        self.stem16 = nn.Sequential(
            nn.Conv2d(d_lat, c_lat16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c_lat16, c_lat16, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.up8 = nn.Conv2d(c_lat16, 4 * c_lat8, kernel_size=1)
        self.conv8 = nn.Conv2d(c_lat8, c_lat8, kernel_size=3, padding=1)
        self.inj8 = nn.Conv2d(c_lat8, w8, kernel_size=1)

        self.up4 = nn.Conv2d(c_lat8, 4 * c_lat4, kernel_size=1)
        self.conv4 = nn.Conv2d(c_lat4, c_lat4, kernel_size=3, padding=1)
        self.inj4 = nn.Conv2d(c_lat4, c_4px, kernel_size=1)

        self.up2 = nn.Conv2d(c_lat4, 4 * c_2px, kernel_size=1)
        self.inj2 = nn.Conv2d(c_2px, c_2px, kernel_size=1)

    def __call__(self, x0_hat_nhwc: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        z16 = self.stem16(x0_hat_nhwc)
        z8 = nn.silu(self.conv8(PixelHeadOps.pixel_shuffle_nhwc(self.up8(z16), 2)))
        z4 = nn.silu(self.conv4(PixelHeadOps.pixel_shuffle_nhwc(self.up4(z8), 2)))
        z2 = PixelHeadOps.pixel_shuffle_nhwc(self.up2(z4), 2)
        return self.inj8(z8), self.inj4(z4), self.inj2(z2)


class SigmaAdaLN(nn.Module):
    def __init__(self, embed_dim: int = 256, target_dims: tuple[int, ...] = (1024, 512)):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )
        self.heads = [nn.Linear(embed_dim, 2 * d) for d in target_dims]

    def embed_sigma(self, sigma: mx.array | float, batch: int, dtype: mx.Dtype) -> mx.array:
        if isinstance(sigma, mx.array):
            s = sigma.astype(mx.float32).reshape(-1)
            if s.size == 1 and batch > 1:
                s = mx.broadcast_to(s, (batch,))
        else:
            s = mx.full((batch,), float(sigma), dtype=mx.float32)
        log_s = mx.log(mx.clip(s, 1e-4, 2.0))
        half = self.embed_dim // 2
        freqs = mx.exp(-math.log(1000.0) * mx.arange(half, dtype=mx.float32) / max(1, half - 1))
        ang = log_s[:, None] * freqs[None, :]
        emb = mx.concatenate([mx.sin(ang), mx.cos(ang)], axis=-1).astype(dtype)
        return self.mlp(emb)

    def modulate_nhwc(self, x: mx.array, cond: mx.array, head_idx: int) -> mx.array:
        scale, shift = mx.split(self.heads[head_idx](cond), 2, axis=-1)
        return x * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :]


class PixelHeadBase(nn.Module):
    token_pixels: int = 16
    out_channels: int = 4
    supports_x2: bool = False
    arch: str = "pixel_head"

    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}
        self.source_label: str = ""

    @property
    def d_feat(self) -> int:
        return self.base.d_feat if hasattr(self, "base") else self._d_feat

    @property
    def trunk_dtype(self) -> mx.Dtype:
        return self.base.feat_proj.weight.dtype if hasattr(self, "base") else self.feat_proj.weight.dtype

    def resolve_scale(self, scale: int | None) -> int:
        if scale is None:
            return 2 if self.supports_x2 else 1
        if scale not in (1, 2):
            raise ValueError(f"pixel_head_scale must be 1, 2, or None, got {scale!r}.")
        if scale == 2 and not self.supports_x2:
            raise ValueError(
                "Loaded PixelHead is a 1x checkpoint; load an x2 checkpoint (e.g. --pixel-head-path x2) for 2x output."
            )
        return scale

    def decode(
        self,
        feat: mx.array,
        x0_hat_norm: mx.array,
        sigma: float = 0.10,
        scale: int | None = None,
        tiled: bool | None = None,
        tile_tokens: int = 64,
        overlap_tokens: int = 8,
    ) -> mx.array:
        resolved_scale = self.resolve_scale(scale)
        feat_nhwc, lat_nhwc = PixelHeadOps.prepare_inputs(feat, x0_hat_norm, self.d_feat)
        gh, gw = lat_nhwc.shape[1], lat_nhwc.shape[2]
        use_tiled = tiled if tiled is not None else (gh * gw > 96 * 96)
        dtype = self.trunk_dtype
        if use_tiled:
            return self._decode_tiled(
                feat_nhwc.astype(dtype),
                lat_nhwc.astype(dtype),
                sigma=sigma,
                scale=resolved_scale,
                tile_tokens=tile_tokens,
                overlap_tokens=overlap_tokens,
            )
        out_nhwc = self._forward_nhwc(
            feat_nhwc.astype(dtype), lat_nhwc.astype(dtype), sigma=sigma, scale=resolved_scale
        )
        return out_nhwc.astype(mx.float32).transpose(0, 3, 1, 2)

    def _forward_nhwc(self, feat: mx.array, lat: mx.array, sigma: float = 0.10, scale: int = 1) -> mx.array:
        raise NotImplementedError

    def _decode_tiled(
        self,
        feat_nhwc: mx.array,
        lat_nhwc: mx.array,
        sigma: float,
        scale: int,
        tile_tokens: int,
        overlap_tokens: int,
    ) -> mx.array:
        b, gh, gw, _ = lat_nhwc.shape
        px_per_tok = self.token_pixels * scale
        out_h, out_w = gh * px_per_tok, gw * px_per_tok
        acc = np.zeros((b, self.out_channels, out_h, out_w), dtype=np.float32)
        w_sum = np.zeros((1, 1, out_h, out_w), dtype=np.float32)

        ys = PixelHeadOps.tile_starts(gh, tile_tokens, overlap_tokens)
        xs = PixelHeadOps.tile_starts(gw, tile_tokens, overlap_tokens)
        ov_px = overlap_tokens * px_per_tok

        for y0 in ys:
            y1 = min(gh, y0 + tile_tokens)
            wy = PixelHeadOps.ramp_1d((y1 - y0) * px_per_tok, ov_px, y0 == 0, y1 == gh)[:, None]
            for x0 in xs:
                x1 = min(gw, x0 + tile_tokens)
                wx = PixelHeadOps.ramp_1d((x1 - x0) * px_per_tok, ov_px, x0 == 0, x1 == gw)[None, :]
                win = (wy * wx)[None, None]

                f_tile = feat_nhwc[:, y0:y1, x0:x1, :]
                x_tile = lat_nhwc[:, y0:y1, x0:x1, :]
                pred_tile = self._forward_nhwc(f_tile, x_tile, sigma=sigma, scale=scale)
                pred_nchw = pred_tile.astype(mx.float32).transpose(0, 3, 1, 2)
                mx.eval(pred_nchw)

                py0, py1 = y0 * px_per_tok, y1 * px_per_tok
                px0, px1 = x0 * px_per_tok, x1 * px_per_tok
                acc[:, :, py0:py1, px0:px1] += np.asarray(pred_nchw) * win
                w_sum[:, :, py0:py1, px0:px1] += win

        return mx.array(acc / np.maximum(w_sum, 1e-6))


class PixelHeadWeightUtil:
    @staticmethod
    def read_metadata(path: Path) -> dict[str, str]:
        with path.open("rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        raw_meta = header.get("__metadata__") or {}
        return {str(k): str(v) for k, v in raw_meta.items()}

    @staticmethod
    def apply_weights(
        model: nn.Module,
        raw: dict[str, mx.array],
        *,
        fp32_prefixes: tuple[str, ...] = (),
        sequential_prefixes: tuple[str, ...] = ("sigma_adaln.mlp.", "lat_highway.stem16."),
        dtype: mx.Dtype | None = None,
        name: str = "PixelHead",
    ) -> None:
        trunk_dtype = dtype or PixelHeadWeightUtil._infer_trunk_dtype(raw)
        converted: dict[str, mx.array] = {}
        for key, value in raw.items():
            target = PixelHeadWeightUtil._map_sequential_key(key, sequential_prefixes)
            tensor = value
            if tensor.ndim == 4 and target.endswith(".weight"):
                tensor = tensor.transpose(0, 2, 3, 1)
            elif tensor.ndim == 4 and target.endswith(".gamma"):
                tensor = tensor.reshape(-1)
            target_dtype = mx.float32 if key.startswith(fp32_prefixes) else trunk_dtype
            converted[target] = tensor.astype(target_dtype)

        expected = dict(tree_flatten(model.parameters()))
        missing = sorted(set(expected) - set(converted))
        unexpected = sorted(set(converted) - set(expected))
        mismatched = sorted(k for k in set(expected) & set(converted) if expected[k].shape != converted[k].shape)
        if missing or unexpected or mismatched:
            raise ValueError(
                f"{name} checkpoint mismatch: missing={missing[:5]} ({len(missing)}), "
                f"unexpected={unexpected[:5]} ({len(unexpected)}), shapes={mismatched[:5]} ({len(mismatched)})"
            )
        model.update(tree_unflatten(list(converted.items())))
        mx.eval(model.parameters())

    @staticmethod
    def _map_sequential_key(key: str, prefixes: tuple[str, ...]) -> str:
        # MLX nn.Sequential stores submodules under `.layers.<idx>` instead of `.<idx>`.
        for prefix in prefixes:
            if key.startswith(prefix):
                return f"{prefix}layers.{key.removeprefix(prefix)}"
        return key

    @staticmethod
    def _infer_trunk_dtype(raw: dict[str, mx.array]) -> mx.Dtype:
        for probe in ("base.feat_proj.weight", "feat_proj.weight"):
            if probe in raw:
                return raw[probe].dtype
        return mx.float32
