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
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler

from diffusion.model.nets.PixArtSigma_SR import PixArtSigmaSR_XL_2
from diffusion.model.nets.adapter import build_adapter_v7


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
    if hasattr(scheduler, "add_noise"):
        latents = scheduler.add_noise(z_lr, noise, t_start)
    else:
        latents = z_lr + noise
    return latents.to(dtype=dtype), timesteps[start_index:]




def build_adapter_struct_input(lr_small_m11: torch.Tensor) -> torch.Tensor:
    return lr_small_m11.float().clamp(-1.0, 1.0)

def mask_adapter_cond(cond, keep_mask: torch.Tensor):
    if cond is None:
        return None
    if not torch.is_tensor(keep_mask):
        keep_mask = torch.tensor(keep_mask)

    def _find_device_dtype(x):
        if torch.is_tensor(x):
            return x.device, x.dtype
        if isinstance(x, (list, tuple)):
            for item in x:
                found = _find_device_dtype(item)
                if found is not None:
                    return found
        return None

    found = _find_device_dtype(cond)
    if found is None:
        return cond
    dev, _ = found
    keep_mask = keep_mask.to(device=dev, dtype=torch.float32)

    def _mask(x: torch.Tensor):
        m = keep_mask
        while m.ndim < x.ndim:
            m = m.unsqueeze(-1)
        return x * m.to(dtype=x.dtype)

    if torch.is_tensor(cond):
        return _mask(cond)
    if isinstance(cond, dict):
        return {k: (_mask(v) if torch.is_tensor(v) else v) for k, v in cond.items()}
    if isinstance(cond, (list, tuple)):
        if len(cond) == 2 and isinstance(cond[0], list) and torch.is_tensor(cond[1]):
            spatial = [_mask(c) for c in cond[0]]
            style = _mask(cond[1])
            return (spatial, style)
        masked = []
        for c in cond:
            if torch.is_tensor(c):
                masked.append(_mask(c))
            elif isinstance(c, list):
                masked.append([_mask(ci) if torch.is_tensor(ci) else ci for ci in c])
            else:
                masked.append(c)
        return masked if isinstance(cond, list) else tuple(masked)
    return cond


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.scaling = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False, dtype=torch.float32)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, dtype=torch.float32)
        self.lora_A.to(base.weight.device)
        self.lora_B.to(base.weight.device)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x):
        out = self.base(x)
        delta = self.lora_B(self.lora_A(x.float())) * self.scaling
        return out + delta.to(out.dtype)


def apply_lora(model, rank=16, alpha=16):
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            key in name for key in ("qkv", "proj", "to_q", "to_k", "to_v", "q_linear", "kv_linear")
        ):
            parent = model.get_submodule(name.rsplit('.', 1)[0])
            child = name.rsplit('.', 1)[1]
            setattr(parent, child, LoRALinear(module, rank, alpha))
            cnt += 1
    print(f"✅ LoRA applied to {cnt} layers.")


def _load_pixart_subset_compatible(pixart: nn.Module, saved_trainable: dict, context: str):
    saved = set(saved_trainable.keys())
    model_keys = set(pixart.state_dict().keys())
    curr = pixart.state_dict()
    loaded, skipped_shape, missing_in_model = 0, 0, 0
    for k in sorted(saved):
        if k not in model_keys:
            missing_in_model += 1
            continue
        ckpt_t = saved_trainable[k]
        if tuple(ckpt_t.shape) == tuple(curr[k].shape):
            curr[k] = ckpt_t.to(dtype=curr[k].dtype)
            loaded += 1
        else:
            skipped_shape += 1
    pixart.load_state_dict(curr, strict=False)
    print(f"[{context}] pixart subset load: loaded={loaded}, model_miss={missing_in_model}, shape_skip={skipped_shape}, saved_total={len(saved)}")


@torch.no_grad()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if (device == "cuda") else torch.float32

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    inj_cfg = ckpt.get("injection_config", {}) if isinstance(ckpt, dict) else {}
    injection_strategy = str(inj_cfg.get("injection_strategy", "three_stage_sr"))
    injection_cutoff_layer = int(inj_cfg.get("injection_cutoff_layer", 28))
    hard_layers = list(inj_cfg.get("hard_layers", [2, 4, 6, 8, 10, 12]))
    transition_layers = list(inj_cfg.get("transition_layers", []))
    detail_layers = list(inj_cfg.get("detail_layers", [14, 16, 18, 20, 22, 24]))

    pixart = PixArtSigmaSR_XL_2(
        input_size=64,
        in_channels=4,
        out_channels=4,
        sparse_inject_ratio=1.0,
        
        
        
        
        
        
        
        
        
        dualstream_enabled=False,
        cross_attn_start_layer=16,
        dual_num_heads=16,
    ).to(device)

    base = torch.load(args.pixart_path, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    if hasattr(pixart, "load_pretrained_weights_with_zero_init"):
        pixart.load_pretrained_weights_with_zero_init(base)
        if hasattr(pixart, "init_lr_embedder_from_x_embedder"):
            pixart.init_lr_embedder_from_x_embedder()
    else:
        pixart.load_state_dict(base, strict=False)

    adapter = build_adapter_v7(
        in_channels=4,
        hidden_size=1152,
        injection_layers_map=getattr(pixart, "injection_layer_to_level", getattr(pixart, "injection_layers", None)),
    ).to(device).float()

    saved_trainable = ckpt.get("pixart_keep", ckpt.get("pixart_trainable", {}))

    has_lora = any(("lora_A" in k) or ("lora_B" in k) for k in saved_trainable.keys())
    if "lora_rank" in ckpt:
        ckpt_lora_rank = int(ckpt["lora_rank"])
    else:
        ckpt_lora_rank = int(args.lora_rank)
    if "lora_alpha" in ckpt:
        ckpt_lora_alpha = int(ckpt["lora_alpha"])
    else:
        ckpt_lora_alpha = int(args.lora_alpha)
    if has_lora:
        apply_lora(pixart, rank=ckpt_lora_rank, alpha=ckpt_lora_alpha)

    _load_pixart_subset_compatible(pixart, saved_trainable, context="infer")
    adapter.load_state_dict(ckpt["adapter"], strict=True)

    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device).float().eval()
    vae.enable_slicing()

    null_pack = torch.load(args.null_t5_embed_path, map_location="cpu")
    y = null_pack["y"].to(device)
    data_info = {
        "img_hw": torch.tensor([[512.0, 512.0]], device=device),
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

    lr_pil = Image.open(args.lr_image).convert("RGB")
    lr_pil = lr_pil.resize((args.crop_size, args.crop_size), Image.BICUBIC)
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    lr_in = norm(to_tensor(lr_pil)).unsqueeze(0).to(device)
    if args.input_is_lr_small:
        lr_small = lr_in
        lr = F.interpolate(lr_small, size=(args.crop_size, args.crop_size), mode="bicubic", align_corners=False, antialias=True)
    else:
        lr = lr_in
        lr_small = F.interpolate(lr, size=(args.crop_size // 4, args.crop_size // 4), mode="bicubic", align_corners=False, antialias=True)

    z_lr = vae.encode(lr).latent_dist.mean * vae.config.scaling_factor
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    if args.use_lq_init:
        latents, run_timesteps = get_lq_init_latents(
            z_lr.to(compute_dtype), scheduler, args.steps, gen, args.lq_init_strength, compute_dtype
        )
    else:
        scheduler.set_timesteps(args.steps, device=device)
        latents = randn_like_with_generator(z_lr.to(compute_dtype), gen)
        run_timesteps = scheduler.timesteps

    adapter_in = build_adapter_struct_input(lr_small).to(device=device, dtype=torch.float32)
    aug_level = torch.zeros((latents.shape[0],), device=device, dtype=compute_dtype)

    for t in run_timesteps:
        t_b = torch.tensor([t], device=device).expand(latents.shape[0])
        t_embed = pixart.t_embedder(t_b.to(dtype=compute_dtype))
        cond = adapter(adapter_in, t_embed=t_embed.float())
        with torch.autocast(device_type="cuda", dtype=compute_dtype) if device == "cuda" else torch.no_grad():
            drop_uncond = torch.ones(latents.shape[0], device=device)
            drop_cond = torch.ones(latents.shape[0], device=device)
            model_in = latents.to(compute_dtype)
            if args.cfg_scale == 1.0:
                out = pixart(
                    x=model_in,
                    timestep=t_b,
                    y=y,
                    aug_level=aug_level,
                    mask=None,
                    data_info=data_info,
                    adapter_cond=cond,
                                        force_drop_ids=drop_cond,
                )
            else:
                cond_zero = mask_adapter_cond(cond, torch.zeros((latents.shape[0],), device=device))
                out_uncond = pixart(
                    x=model_in,
                    timestep=t_b,
                    y=y,
                    aug_level=aug_level,
                    mask=None,
                    data_info=data_info,
                    adapter_cond=cond_zero,
                                        force_drop_ids=drop_uncond,
                )
                out_cond = pixart(
                    x=model_in,
                    timestep=t_b,
                    y=y,
                    aug_level=aug_level,
                    mask=None,
                    data_info=data_info,
                    adapter_cond=cond,
                                        force_drop_ids=drop_cond,
                )
                out = out_uncond + args.cfg_scale * (out_cond - out_uncond)

            if out.shape[1] != 4:
                raise RuntimeError(f"Expected 4-channel output, got {out.shape[1]}")
        latents = scheduler.step(out.float(), t, latents.float()).prev_sample

    pred = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)
    pred01 = ((pred[0].detach().cpu().float() + 1.0) * 0.5).clamp(0.0, 1.0)
    pred_pil = transforms.ToPILImage()(pred01)

    out_path = Path(args.out_image)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_pil.save(out_path)
    print(f"✅ Saved SR image to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Single-image SR inference (DDIM, dualstream, validation-aligned settings)")
    parser.add_argument("--lr-image", type=str, required=True)
    parser.add_argument("--out-image", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--pixart-path", type=str, required=True)
    parser.add_argument("--vae-path", type=str, required=True)
    parser.add_argument("--null-t5-embed-path", type=str, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--use-lq-init", dest="use_lq_init", action="store_true")
    parser.add_argument("--no-lq-init", dest="use_lq_init", action="store_false")
    parser.set_defaults(use_lq_init=True)
    parser.add_argument("--lq-init-strength", type=float, default=0.3)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--input_is_lr_small", type=lambda x: str(x).lower() in ("1","true","yes","y"), default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
