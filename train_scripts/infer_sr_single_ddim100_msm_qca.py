import os
import sys
import math
import argparse
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler

from diffusion.model.nets.PixArtSigma_SR import PixArtSigmaSR_XL_2
from diffusion.model.nets.adapter import build_adapter_msm_qca
from train_scripts.msm_qca_utils import (
    DEFAULT_MEMORY_TOKEN_COUNTS,
    DEFAULT_RESAMPLER_DIM,
    DEFAULT_RESAMPLER_DEPTH,
    DEFAULT_RESAMPLER_HEADS,
    apply_lora_attn_only,
    load_pixart_subset_compatible,
    build_msm_qca_config,
    assert_msm_qca_config_compatible,
)

ADAPTER_CA_BLOCK_IDS = [14, 18, 22, 26]


def randn_like_with_generator(tensor, generator):
    return torch.randn(tensor.shape, device=tensor.device, dtype=tensor.dtype, generator=generator)


def get_lq_init_latents(z_lr, scheduler, steps, generator, strength, dtype):
    strength = float(max(0.0, min(1.0, strength)))
    scheduler.set_timesteps(steps, device=z_lr.device)
    timesteps = scheduler.timesteps
    start_index = int(round(strength * (len(timesteps) - 1)))
    start_index = min(max(start_index, 0), len(timesteps) - 1)
    t_start = timesteps[start_index]
    noise = randn_like_with_generator(z_lr, generator)
    latents = scheduler.add_noise(z_lr, noise, t_start) if hasattr(scheduler, "add_noise") else (z_lr + noise)
    return latents.to(dtype=dtype), timesteps[start_index:]


@torch.no_grad()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    ckpt = torch.load(args.ckpt_path, map_location="cpu")

    pixart = PixArtSigmaSR_XL_2(
        input_size=64,
        in_channels=4,
        out_channels=4,
        adapter_ca_block_ids=ADAPTER_CA_BLOCK_IDS,
    ).to(device)

    base = torch.load(args.pixart_path, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    if hasattr(pixart, "load_pretrained_weights_with_zero_init"):
        pixart.load_pretrained_weights_with_zero_init(base)
    else:
        pixart.load_state_dict(base, strict=False)

    adapter = build_adapter_msm_qca(
        hidden_size=1152,
        memory_token_counts=DEFAULT_MEMORY_TOKEN_COUNTS,
        resampler_dim=DEFAULT_RESAMPLER_DIM,
        resampler_depth=DEFAULT_RESAMPLER_DEPTH,
        resampler_heads=DEFAULT_RESAMPLER_HEADS,
    ).to(device).float()

    saved_trainable = ckpt.get("pixart_keep", ckpt.get("pixart_trainable", {}))
    has_lora = any(("lora_A" in k) or ("lora_B" in k) for k in saved_trainable.keys())
    lora_rank = int(ckpt.get("lora_rank", args.lora_rank))
    lora_alpha = int(ckpt.get("lora_alpha", args.lora_alpha))
    if has_lora:
        apply_lora_attn_only(pixart, rank=lora_rank, alpha=lora_alpha)

    current_cfg = build_msm_qca_config(
        adapter_ca_block_ids=ADAPTER_CA_BLOCK_IDS,
        memory_token_counts=DEFAULT_MEMORY_TOKEN_COUNTS,
        resampler_dim=DEFAULT_RESAMPLER_DIM,
        resampler_depth=DEFAULT_RESAMPLER_DEPTH,
        resampler_heads=DEFAULT_RESAMPLER_HEADS,
        batch_size=1,
        grad_accum_steps=1,
        max_train_steps=1,
        dataset_name="infer_runtime",
        crop_size=args.crop_size,
        scale=4,
        optimizer_lrs={},
    )
    assert_msm_qca_config_compatible(current_cfg, ckpt.get("msm_qca_config", {}), context="infer")
    load_pixart_subset_compatible(pixart, saved_trainable, context="infer")
    miss, unexp = adapter.load_state_dict(ckpt["adapter"], strict=True)
    print(f"[infer-adapter] strict load ok: missing={len(miss)}, unexpected={len(unexp)}")

    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device).float().eval()
    vae.enable_slicing()

    null_pack = torch.load(args.null_t5_embed_path, map_location="cpu")
    y = null_pack["y"].to(device)
    data_info = {
        "img_hw": torch.tensor([[float(args.crop_size), float(args.crop_size)]], device=device),
        "aspect_ratio": torch.tensor([1.0], device=device),
    }

    pixart.eval()
    adapter.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="v_prediction",
        set_alpha_to_one=False,
    )

    lr_pil = Image.open(args.lr_image).convert("RGB").resize((args.crop_size, args.crop_size), Image.BICUBIC)
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    lr_in = norm(to_tensor(lr_pil)).unsqueeze(0).to(device)
    lr_small = lr_in if args.input_is_lr_small else transforms.Resize((args.crop_size // 4, args.crop_size // 4), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(lr_in)

    z_lr = vae.encode(lr_in).latent_dist.mean * vae.config.scaling_factor
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    if args.use_lq_init:
        latents, run_timesteps = get_lq_init_latents(z_lr.to(compute_dtype), scheduler, args.steps, gen, args.lq_init_strength, compute_dtype)
    else:
        scheduler.set_timesteps(args.steps, device=device)
        latents = randn_like_with_generator(z_lr.to(compute_dtype), gen)
        run_timesteps = scheduler.timesteps

    aug_level = torch.zeros((1,), device=device, dtype=compute_dtype)
    for t in run_timesteps:
        t_b = torch.tensor([t], device=device)
        t_embed = pixart.t_embedder(t_b.to(dtype=compute_dtype))
        cond = adapter(lr_small.to(dtype=compute_dtype), t_embed=t_embed)
        out = pixart(x=latents.to(compute_dtype), timestep=t_b, y=y, aug_level=aug_level, mask=None, data_info=data_info,
                     adapter_cond=cond, force_drop_ids=torch.ones(1, device=device))
        latents = scheduler.step(out.float(), t, latents.float()).prev_sample

    pred = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)
    out_img = ((pred[0].detach().cpu().permute(1, 2, 0).float().numpy() + 1.0) * 127.5).clip(0, 255).astype("uint8")
    Image.fromarray(out_img).save(args.output)
    print(f"✅ Saved: {args.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixart_path", type=str, required=True)
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--vae_path", type=str, required=True)
    ap.add_argument("--null_t5_embed_path", type=str, required=True)
    ap.add_argument("--lr_image", type=str, required=True)
    ap.add_argument("--output", type=str, default="infer_out.png")
    ap.add_argument("--crop_size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--lora_rank", type=int, default=4)
    ap.add_argument("--lora_alpha", type=int, default=4)
    ap.add_argument("--use_lq_init", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lq_init_strength", type=float, default=0.3)
    ap.add_argument("--input_is_lr_small", action="store_true")
    run(ap.parse_args())
