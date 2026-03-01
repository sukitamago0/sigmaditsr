import os
import glob
import math
import random
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm

import lpips
from diffusers import AutoencoderKL, DDIMScheduler
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

from diffusion.model.nets.PixArtSigma_SR import PixArtSigmaSR_XL_2
from diffusion.model.nets.adapter import build_adapter_v7


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def file_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_sigma_sr_base_weights(pixart: torch.nn.Module, pixart_path: str):
    base = torch.load(pixart_path, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]

    # Keep same 4->8 adaptation behavior as training script.
    if "x_embedder.proj.weight" in base and base["x_embedder.proj.weight"].shape[1] == 4:
        w4 = base["x_embedder.proj.weight"]
        w8 = torch.zeros((w4.shape[0], 8, w4.shape[2], w4.shape[3]), dtype=w4.dtype)
        w8[:, :4] = w4
        w8[:, 4:] = w4 * 0.5
        base["x_embedder.proj.weight"] = w8

    if hasattr(pixart, "load_pretrained_weights_with_zero_init"):
        loaded, skipped = pixart.load_pretrained_weights_with_zero_init(base)
        print(f"✅ Loaded base PixArt-Sigma weights via shape-aware loader: loaded={loaded}, skipped={skipped}")
    else:
        missing, unexpected = pixart.load_state_dict(base, strict=False)
        print(f"✅ Loaded base PixArt-Sigma weights: missing={len(missing)}, unexpected={len(unexpected)}")


def rgb01_to_y01(rgb01):
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    return (16.0 + 65.481 * r + 128.553 * g + 24.966 * b) / 255.0


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.scaling = alpha / r
        self.lora_A = torch.nn.Linear(base.in_features, r, bias=False, dtype=torch.float32)
        self.lora_B = torch.nn.Linear(r, base.out_features, bias=False, dtype=torch.float32)
        self.lora_A.to(base.weight.device)
        self.lora_B.to(base.weight.device)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)
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
        if isinstance(module, torch.nn.Linear) and any(
            key in name for key in ("qkv", "proj", "to_q", "to_k", "to_v", "q_linear", "kv_linear")
        ):
            parent = model.get_submodule(name.rsplit('.', 1)[0])
            child = name.rsplit('.', 1)[1]
            setattr(parent, child, LoRALinear(module, rank, alpha))
            cnt += 1
    print(f"✅ LoRA applied to {cnt} layers.")


def _strict_load_pixart_trainable_subset(pixart: torch.nn.Module, saved_trainable: dict):
    expected = {k for k, p in pixart.named_parameters() if p.requires_grad}
    saved = set(saved_trainable.keys())
    missing = sorted(expected - saved)
    unexpected = sorted(saved - expected)
    if missing or unexpected:
        msg = ["pixart_trainable key mismatch."]
        if missing:
            msg.append(f"missing({len(missing)}): {missing[:20]}")
        if unexpected:
            msg.append(f"unexpected({len(unexpected)}): {unexpected[:20]}")
        raise RuntimeError(" ".join(msg))

    curr = pixart.state_dict()
    bad_shapes = []
    for k in sorted(saved & expected):
        ckpt_t = saved_trainable[k]
        if tuple(ckpt_t.shape) != tuple(curr[k].shape):
            bad_shapes.append((k, tuple(ckpt_t.shape), tuple(curr[k].shape)))
    if bad_shapes:
        preview = "; ".join([f"{k}: ckpt{a} vs model{b}" for k, a, b in bad_shapes[:20]])
        raise RuntimeError(f"pixart_trainable shape mismatch count={len(bad_shapes)}. {preview}")

    for k in sorted(expected):
        curr[k] = saved_trainable[k].to(dtype=curr[k].dtype)
    pixart.load_state_dict(curr, strict=False)


def center_crop_aligned_pair(lr_pil: Image.Image, hr_pil: Image.Image, scale: int = 4):
    wl, hl = lr_pil.size
    wh, hh = hr_pil.size
    h2 = min(hh, hl * scale)
    w2 = min(wh, wl * scale)
    h2 = (h2 // scale) * scale
    w2 = (w2 // scale) * scale
    if h2 <= 0 or w2 <= 0:
        raise ValueError(f"Invalid aligned size with LR={lr_pil.size}, HR={hr_pil.size}")
    hr_top = (hh - h2) // 2
    hr_left = (wh - w2) // 2
    lr_h2 = h2 // scale
    lr_w2 = w2 // scale
    lr_top = (hl - lr_h2) // 2
    lr_left = (wl - lr_w2) // 2
    hr_aligned = TF.crop(hr_pil, hr_top, hr_left, h2, w2)
    lr_aligned = TF.crop(lr_pil, lr_top, lr_left, lr_h2, lr_w2)
    return lr_aligned, hr_aligned


class DF2KValFixedDataset(Dataset):
    def __init__(self, hr_root, lr_root=None, crop_size=512):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_root, "*.png")))
        if len(self.hr_paths) == 0:
            raise RuntimeError(f"No HR PNGs found in {hr_root}")
        self.lr_root = lr_root
        self.crop_size = crop_size
        self.norm = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]
        hr_pil = Image.open(hr_path).convert("RGB")
        lr_crop = None
        if self.lr_root:
            base = os.path.basename(hr_path)
            lr_name = base.replace(".png", "x4.png")
            lr_p = os.path.join(self.lr_root, lr_name)
            if os.path.exists(lr_p):
                lr_pil = Image.open(lr_p).convert("RGB")
                lr_aligned, hr_aligned = center_crop_aligned_pair(lr_pil, hr_pil, scale=4)
                lr_crop = TF.center_crop(lr_aligned, (self.crop_size // 4, self.crop_size // 4))
                hr_crop = TF.center_crop(hr_aligned, (self.crop_size, self.crop_size))
                lr_crop = TF.resize(
                    lr_crop,
                    (self.crop_size, self.crop_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                )
        if lr_crop is None:
            hr_crop = TF.center_crop(hr_pil, (self.crop_size, self.crop_size))
            w, h = hr_crop.size
            lr_small = hr_crop.resize((w // 4, h // 4), Image.BICUBIC)
            lr_crop = lr_small.resize((w, h), Image.BICUBIC)

        hr_tensor = self.norm(self.to_tensor(hr_crop))
        lr_tensor = self.norm(self.to_tensor(lr_crop))
        return {"hr": hr_tensor, "lr": lr_tensor, "path": hr_path}


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


@torch.no_grad()
def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    hr_root = args.val_hr_dir
    lr_root = args.val_lr_dir
    ds = DF2KValFixedDataset(hr_root=hr_root, lr_root=lr_root, crop_size=args.crop_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    pixart = PixArtSigmaSR_XL_2(
        input_size=64,
        in_channels=8,
        sparse_inject_ratio=1.0,
        injection_cutoff_layer=28,
        injection_strategy="full",
    ).to(device)
    load_sigma_sr_base_weights(pixart, args.pixart_path)
    apply_lora(pixart, rank=16, alpha=16)

    adapter = build_adapter_v7(
        in_channels=4,
        hidden_size=1152,
        injection_layers_map=getattr(pixart, "injection_layer_to_level", getattr(pixart, "injection_layers", None)),
    ).to(device).float()

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    if "pixart_trainable" not in ckpt or "adapter" not in ckpt:
        raise KeyError("Checkpoint must contain keys: pixart_trainable and adapter")

    ckpt_base_hash = ckpt.get("base_pixart_sha256", None)
    if ckpt_base_hash is not None:
        current_base_hash = file_sha256(args.pixart_path)
        if str(current_base_hash) != str(ckpt_base_hash):
            raise RuntimeError(
                "Base PixArt checkpoint hash mismatch. "
                f"ckpt expects {ckpt_base_hash}, current is {current_base_hash}. "
                "Please use the same PIXART_PATH as training."
            )
        print("✅ Base PixArt hash matched training checkpoint metadata.")

    _strict_load_pixart_trainable_subset(pixart, ckpt["pixart_trainable"])
    adapter.load_state_dict(ckpt["adapter"], strict=True)

    pixart.eval()
    adapter.eval()

    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device).float().eval()
    vae.enable_slicing()

    null_pack = torch.load(args.null_t5_embed_path, map_location="cpu")
    if "y" not in null_pack:
        raise KeyError("Null T5 embed file missing key 'y'")
    y_embed = null_pack["y"].to(device)

    d_info = {
        "img_hw": torch.tensor([[float(args.crop_size), float(args.crop_size)]], device=device),
        "aspect_ratio": torch.tensor([1.0], device=device),
    }

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="v_prediction",
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(args.steps, device=device)

    lpips_fn = lpips.LPIPS(net='vgg').to("cpu").eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    val_gen = torch.Generator(device=device)
    val_gen.manual_seed(seed)

    base_psnr, base_ssim, base_lpips = [], [], []
    model_psnr, model_ssim, model_lpips = [], [], []

    pbar = tqdm(loader, desc=f"Eval@{args.steps}")
    for idx, batch in enumerate(pbar):
        if args.max_samples > 0 and idx >= args.max_samples:
            break

        hr = batch["hr"].to(device)
        lr = batch["lr"].to(device)

        # Baseline: bicubic-upsampled LR from dataset output vs HR
        lr01 = (lr + 1.0) / 2.0
        hr01 = (hr + 1.0) / 2.0
        ly = rgb01_to_y01(lr01)[..., 4:-4, 4:-4]
        hy = rgb01_to_y01(hr01)[..., 4:-4, 4:-4]
        base_psnr.append(psnr(ly, hy, data_range=1.0).item())
        base_ssim.append(ssim(ly, hy, data_range=1.0).item())
        base_lpips.append(lpips_fn(lr.detach().cpu().float(), hr.detach().cpu().float()).mean().item())

        # Model
        z_hr = vae.encode(hr).latent_dist.mean * vae.config.scaling_factor
        z_lr = vae.encode(lr).latent_dist.mean * vae.config.scaling_factor

        if args.use_lq_init:
            latents, run_timesteps = get_lq_init_latents(
                z_lr.to(compute_dtype), scheduler, args.steps, val_gen, args.lq_init_strength, compute_dtype
            )
        else:
            latents = randn_like_with_generator(z_hr, val_gen)
            run_timesteps = scheduler.timesteps

        cond = adapter(z_lr.float())
        aug_level = torch.zeros((latents.shape[0],), device=device, dtype=compute_dtype)

        for t in run_timesteps:
            t_b = torch.tensor([t], device=device).expand(latents.shape[0])
            with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=(device == "cuda")):
                drop_cond = torch.ones(latents.shape[0], device=device)
                model_in = torch.cat([latents.to(compute_dtype), z_lr.to(compute_dtype)], dim=1)
                out = pixart(
                    x=model_in,
                    timestep=t_b,
                    y=y_embed,
                    aug_level=aug_level,
                    mask=None,
                    data_info=d_info,
                    adapter_cond=cond,
                    injection_mode="hybrid",
                    force_drop_ids=drop_cond,
                )
                if out.shape[1] == 8:
                    out, _ = out.chunk(2, dim=1)
            latents = scheduler.step(out.float(), t, latents.float()).prev_sample

        pred = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)
        p01 = (pred + 1.0) / 2.0
        py = rgb01_to_y01(p01)[..., 4:-4, 4:-4]
        model_psnr.append(psnr(py, hy, data_range=1.0).item())
        model_ssim.append(ssim(py, hy, data_range=1.0).item())
        model_lpips.append(lpips_fn(pred.detach().cpu().float(), hr.detach().cpu().float()).mean().item())

    def fmt(arr):
        return float(np.mean(arr)) if len(arr) > 0 else float("nan")

    b_psnr, b_ssim, b_lpips = fmt(base_psnr), fmt(base_ssim), fmt(base_lpips)
    m_psnr, m_ssim, m_lpips = fmt(model_psnr), fmt(model_ssim), fmt(model_lpips)

    print("\n===== Baseline vs Model on VAL_MODE=lr_dir set =====")
    print(f"Samples: {len(model_psnr)}")
    print(f"[Baseline: LR bicubic -> HR] PSNR={b_psnr:.4f} | SSIM={b_ssim:.6f} | LPIPS={b_lpips:.6f}")
    print(f"[Model]                    PSNR={m_psnr:.4f} | SSIM={m_ssim:.6f} | LPIPS={m_lpips:.6f}")
    print(f"[Delta Model-Baseline]     dPSNR={m_psnr-b_psnr:+.4f} | dSSIM={m_ssim-b_ssim:+.6f} | dLPIPS={m_lpips-b_lpips:+.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SR model vs bicubic baseline on lr_dir val set")
    parser.add_argument("--pixart_path", type=str, default=os.path.join(PROJECT_ROOT, "output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth"))
    parser.add_argument("--ckpt_path", type=str, default=os.path.join(PROJECT_ROOT, "experiments_results/train_sigma_sr_vpred/checkpoints/last.pth"))
    parser.add_argument("--vae_path", type=str, default=os.path.join(PROJECT_ROOT, "output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers/vae"))
    parser.add_argument("--null_t5_embed_path", type=str, default=os.path.join(PROJECT_ROOT, "output/pretrained_models/null_t5_embed_sigma_300.pth"))
    parser.add_argument("--val_hr_dir", type=str, default="/data/DF2K/DF2K_valid_HR")
    parser.add_argument("--val_lr_dir", type=str, default="/data/DF2K/DF2K_valid_LR_unknown/X4")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--use_lq_init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lq_init_strength", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples")
    args = parser.parse_args()

    if not os.path.exists(args.pixart_path):
        raise FileNotFoundError(f"pixart_path not found: {args.pixart_path}")
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    if not os.path.exists(args.vae_path):
        raise FileNotFoundError(f"VAE path not found: {args.vae_path}")
    if not os.path.exists(args.null_t5_embed_path):
        raise FileNotFoundError(f"null_t5_embed_path not found: {args.null_t5_embed_path}")
    if not os.path.exists(args.val_hr_dir):
        raise FileNotFoundError(f"val_hr_dir not found: {args.val_hr_dir}")

    evaluate(args)
