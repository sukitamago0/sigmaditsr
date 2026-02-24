#!/usr/bin/env python3
"""Audit PixArt-Sigma SR weight loading and quantify skipped keys.

This script reproduces the current training-script/model loading path:
1) load base checkpoint
2) drop pos_embed
3) adapt x_embedder 4->8 using w4*0.5 for channels 4:8
4) apply the same shape-aware matching logic used by PixArtSigmaSR.load_pretrained_weights_with_zero_init

It prints key-level reasons for loaded/skipped tensors so we can quantify impact.
"""

import argparse
import os
from collections import Counter

import torch

from diffusion.model.nets.PixArtSigma_SR import PixArtSigmaSR_XL_2


def build_model(device: str):
    model = PixArtSigmaSR_XL_2(
        input_size=64,
        in_channels=8,
        sparse_inject_ratio=1.0,
        injection_cutoff_layer=28,
        injection_strategy="full",
    ).to(device)
    return model


def preprocess_base_state_dict(base):
    if "state_dict" in base:
        base = base["state_dict"]

    base = dict(base)
    if "pos_embed" in base:
        del base["pos_embed"]

    if "x_embedder.proj.weight" in base and base["x_embedder.proj.weight"].shape[1] == 4:
        w4 = base["x_embedder.proj.weight"]
        w8 = torch.zeros((w4.shape[0], 8, w4.shape[2], w4.shape[3]), dtype=w4.dtype)
        w8[:, :4] = w4
        w8[:, 4:] = w4 * 0.5
        base["x_embedder.proj.weight"] = w8

    return base


def audit_loading(model, ckpt_state):
    own_state = model.state_dict()

    loaded = []
    skipped = []

    for name, param in ckpt_state.items():
        if name not in own_state:
            skipped.append((name, "not_in_model", tuple(param.shape), None))
            continue

        src = param.data if isinstance(param, torch.nn.Parameter) else param
        dst = own_state[name]

        if name == "x_embedder.proj.weight" and src.ndim == 4 and dst.ndim == 4 and src.shape[1] == 4 and dst.shape[1] == 8:
            loaded.append((name, "adapted_4to8", tuple(src.shape), tuple(dst.shape)))
            continue

        if (
            name == "y_embedder.y_embedding"
            and src.ndim == 2
            and dst.ndim == 2
            and src.shape[1] == dst.shape[1]
            and src.shape[0] != dst.shape[0]
        ):
            loaded.append((name, "partial_prefix_copy", tuple(src.shape), tuple(dst.shape)))
            continue

        if name.startswith("final_layer.linear") and tuple(src.shape) != tuple(dst.shape):
            skipped.append((name, "final_layer_shape_mismatch", tuple(src.shape), tuple(dst.shape)))
            continue

        if tuple(src.shape) == tuple(dst.shape):
            loaded.append((name, "shape_match", tuple(src.shape), tuple(dst.shape)))
        else:
            skipped.append((name, "shape_mismatch", tuple(src.shape), tuple(dst.shape)))

    return loaded, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-ckpt",
        default=os.path.join("output", "pretrained_models", "PixArt-Sigma-XL-2-512-MS.pth"),
        help="Path to base PixArt checkpoint",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--show", type=int, default=300, help="Max skipped keys to print")
    args = parser.parse_args()

    if not os.path.exists(args.base_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.base_ckpt}")

    print(f"[INFO] Loading model on {args.device}")
    model = build_model(args.device)

    print(f"[INFO] Loading checkpoint: {args.base_ckpt}")
    base = torch.load(args.base_ckpt, map_location="cpu")
    base = preprocess_base_state_dict(base)

    loaded, skipped = audit_loading(model, base)

    load_reason = Counter([r for _, r, _, _ in loaded])
    skip_reason = Counter([r for _, r, _, _ in skipped])

    print("\n=== Summary ===")
    print(f"ckpt_keys_after_preprocess: {len(base)}")
    print(f"loaded_keys: {len(loaded)}")
    print(f"skipped_keys: {len(skipped)}")

    print("\nLoaded reasons:")
    for k, v in load_reason.items():
        print(f"  - {k}: {v}")

    print("\nSkipped reasons:")
    for k, v in skip_reason.items():
        print(f"  - {k}: {v}")

    print("\n=== Skipped key details ===")
    for i, (name, reason, src_shape, dst_shape) in enumerate(skipped):
        if i >= args.show:
            rest = len(skipped) - args.show
            if rest > 0:
                print(f"... ({rest} more skipped keys)")
            break
        print(f"[{i:04d}] {reason:28s} | {name} | src={src_shape} dst={dst_shape}")

    print("\n=== Final-layer entries in checkpoint/model ===")
    final_in_ckpt = sorted([k for k in base.keys() if k.startswith("final_layer.linear")])
    final_in_model = sorted([k for k in model.state_dict().keys() if k.startswith("final_layer.linear")])
    print(f"final_layer.linear keys in ckpt: {len(final_in_ckpt)}")
    print(f"final_layer.linear keys in model: {len(final_in_model)}")
    for k in final_in_ckpt:
        s = tuple(base[k].shape)
        d = tuple(model.state_dict()[k].shape) if k in model.state_dict() else None
        print(f"  - {k}: ckpt={s}, model={d}")


if __name__ == "__main__":
    main()
