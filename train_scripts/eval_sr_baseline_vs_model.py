import os
import glob
import math
import random
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
from diffusion.model.nets.adapter import build_adapter_msm_qca
from train_scripts.msm_qca_utils import (
    DEFAULT_MEMORY_TOKEN_COUNTS,
    DEFAULT_RESAMPLER_DIM,
    DEFAULT_RESAMPLER_DEPTH,
    DEFAULT_RESAMPLER_HEADS,
    build_msm_qca_config,
    assert_msm_qca_config_compatible,
)


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
ADAPTER_CA_BLOCK_IDS = [14, 18, 22, 26]


def file_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_state_dict_shape_compatible(model: nn.Module, state_dict: dict, context: str = "load"):
    curr = model.state_dict()
    filt = {}
    skipped = []
    for k, v in state_dict.items():
        if k in curr and tuple(v.shape) == tuple(curr[k].shape):
            filt[k] = v
        else:
            skipped.append(k)
    missing, unexpected = model.load_state_dict(filt, strict=False)
    print(f"[{context}] compatible load: loaded={len(filt)}, skipped_shape_or_missing={len(skipped)}, missing={len(missing)}, unexpected={len(unexpected)}")
    if len(skipped) > 0:
        print(f"[{context}] skipped examples: {skipped[:5]}")
    return missing, unexpected, skipped


def load_sigma_sr_base_weights(pixart: nn.Module, pixart_path: str):
    base = torch.load(pixart_path, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]

    if hasattr(pixart, "load_pretrained_weights_with_zero_init"):
        loaded, skipped = pixart.load_pretrained_weights_with_zero_init(base)
        print(f"✅ Loaded base PixArt-Sigma weights via shape-aware loader: loaded={loaded}, skipped={skipped}")
    else:
        missing, unexpected, skipped = load_state_dict_shape_compatible(pixart, base, context="base-pretrain")
        print(f"✅ Loaded base PixArt-Sigma weights: missing={len(missing)}, unexpected={len(unexpected)}, skipped={len(skipped)}")


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


def _block_id_from_name(name: str):
    import re
    m = re.search(r"blocks\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def apply_lora(model, rank=4, alpha=4):
    cnt = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        block_id = _block_id_from_name(name)
        if block_id is None or not (0 <= block_id <= 27):
            continue
        if not (("attn.qkv" in name) or ("attn.proj" in name)):
            continue
        parent = model.get_submodule(name.rsplit('.', 1)[0])
        child = name.rsplit('.', 1)[1]
        setattr(parent, child, LoRALinear(module, rank, alpha))
        cnt += 1
    print(f"✅ Attention LoRA applied to {cnt} layers (rank={rank}, alpha={alpha}).")


def _load_pixart_trainable_subset_compatible(pixart: nn.Module, saved_trainable: dict, context: str):
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


class RealSRPairedDataset(Dataset):
    def __init__(self, roots, crop_size=512):
        self.crop_size = int(crop_size)
        self.norm = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        self.to_tensor = transforms.ToTensor()
        self.pairs = []
        for root in roots:
            root = str(root)
            if not os.path.isdir(root):
                continue
            for hr_path in sorted(glob.glob(os.path.join(root, "*_HR.png"))):
                lr_path = hr_path.replace("_HR.png", "_LR4.png")
                if os.path.exists(lr_path):
                    self.pairs.append((hr_path, lr_path))
        if len(self.pairs) == 0:
            raise FileNotFoundError(f"No RealSR pairs found in roots={roots}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hr_path, lr_path = self.pairs[idx]
        hr_pil = Image.open(hr_path).convert("RGB")
        lr_pil = Image.open(lr_path).convert("RGB")
        lr_aligned, hr_aligned = center_crop_aligned_pair(lr_pil, hr_pil, scale=4)
        lr_small_pil = TF.center_crop(lr_aligned, (self.crop_size // 4, self.crop_size // 4))
        hr_crop = TF.center_crop(hr_aligned, (self.crop_size, self.crop_size))
        lr_up_pil = TF.resize(lr_small_pil, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

        hr_tensor = self.norm(self.to_tensor(hr_crop))
        lr_tensor = self.norm(self.to_tensor(lr_up_pil))
        lr_small_tensor = self.norm(self.to_tensor(lr_small_pil))
        return {"hr": hr_tensor, "lr": lr_tensor, "lr_small": lr_small_tensor, "path": hr_path}


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


def _hf_mask_from_gt(gt_m11: torch.Tensor, q: float = 0.9):
    gt01 = (gt_m11 + 1.0) * 0.5
    y = rgb01_to_y01(gt01)
    kx = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], device=y.device, dtype=y.dtype)
    ky = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], device=y.device, dtype=y.dtype)
    gx = F.conv2d(y, kx, padding=1)
    gy = F.conv2d(y, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    th = torch.quantile(mag.flatten(1), q=q, dim=1, keepdim=True)
    th = th.view(-1, 1, 1, 1)
    return (mag >= th).float()


def masked_psnr(pred_y: torch.Tensor, gt_y: torch.Tensor, mask: torch.Tensor):
    mse = ((pred_y - gt_y) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
    return float((-10.0 * torch.log10(mse.clamp_min(1e-12))).item())


def masked_ssim(pred_y: torch.Tensor, gt_y: torch.Tensor, mask: torch.Tensor):
    # Practical masked proxy: apply mask before SSIM
    return float(ssim(pred_y * mask, gt_y * mask, data_range=1.0).item())


@torch.no_grad()
def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    roots = [r.strip() for r in args.realsr_roots.split(",") if r.strip()]
    ds = RealSRPairedDataset(roots=roots, crop_size=args.crop_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    baseline_modes = [x.strip() for x in args.baseline.split(",") if x.strip()]
    run_bicubic = "bicubic" in baseline_modes
    run_model = "current_model" in baseline_modes

    print(f"[Protocol] dataset_name=RealSRPaired num_samples={len(ds)} crop_size={args.crop_size} steps={args.steps} use_lq_init={args.use_lq_init} lq_init_strength={args.lq_init_strength}")
    print("[Protocol] PSNR/SSIM: RGB->Y, shave=4:-4, data_range=1.0 | LPIPS: full RGB no shave")

    lpips_fn = lpips.LPIPS(net='vgg').to("cpu").eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    dists_fn = None
    if args.with_dists:
        try:
            from DISTS_pytorch import DISTS
            dists_fn = DISTS().to(device).eval()
            for p in dists_fn.parameters():
                p.requires_grad_(False)
            print("✅ DISTS enabled")
        except Exception as e:
            print(f"⚠️ DISTS requested but unavailable: {e}")

    pixart, adapter, vae, y_embed, d_info, scheduler, val_gen = [None] * 7
    if run_model:
        pixart = PixArtSigmaSR_XL_2(input_size=64, in_channels=4, out_channels=4, adapter_ca_block_ids=ADAPTER_CA_BLOCK_IDS).to(device)
        load_sigma_sr_base_weights(pixart, args.pixart_path)

        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        current_cfg = build_msm_qca_config(
            adapter_ca_block_ids=ADAPTER_CA_BLOCK_IDS,
            memory_token_counts=DEFAULT_MEMORY_TOKEN_COUNTS,
            resampler_dim=DEFAULT_RESAMPLER_DIM,
            resampler_depth=DEFAULT_RESAMPLER_DEPTH,
            resampler_heads=DEFAULT_RESAMPLER_HEADS,
            batch_size=1,
            grad_accum_steps=1,
            max_train_steps=1,
            dataset_name="eval_runtime",
            crop_size=args.crop_size,
            scale=4,
            optimizer_lrs={},
        )
        assert_msm_qca_config_compatible(current_cfg, ckpt.get("msm_qca_config", {}), context="eval")
        saved_trainable = ckpt.get("pixart_keep", ckpt.get("pixart_trainable", {}))
        has_lora = any(("lora_A" in k) or ("lora_B" in k) for k in saved_trainable.keys())
        lora_rank = int(ckpt["lora_rank"]) if "lora_rank" in ckpt else 4
        lora_alpha = int(ckpt["lora_alpha"]) if "lora_alpha" in ckpt else 4
        if has_lora:
            apply_lora(pixart, rank=lora_rank, alpha=lora_alpha)

        _load_pixart_trainable_subset_compatible(pixart, saved_trainable, context="eval")

        adapter = build_adapter_msm_qca(hidden_size=1152).to(device).float()
        miss, unexp = adapter.load_state_dict(ckpt["adapter"], strict=True)
        print(f"[eval-adapter] strict load ok: missing={len(miss)}, unexpected={len(unexp)}")

        pixart.eval()
        adapter.eval()

        vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device).float().eval()
        vae.enable_slicing()

        null_pack = torch.load(args.null_t5_embed_path, map_location="cpu")
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
        val_gen = torch.Generator(device=device)
        val_gen.manual_seed(args.seed)

    metrics = {}
    for k in ["bicubic", "current_model", "extra"]:
        metrics[k] = {"psnr": [], "ssim": [], "lpips": [], "psnr_hf": [], "ssim_hf": [], "dists": []}

    pbar = tqdm(loader, desc=f"Eval@{args.steps}")
    for i, batch in enumerate(pbar):
        if args.max_samples > 0 and i >= args.max_samples:
            break

        hr = batch["hr"].to(device)
        lr_up = batch["lr"].to(device)
        lr_small = batch["lr_small"].to(device)

        hr01 = (hr + 1.0) / 2.0
        hy = rgb01_to_y01(hr01)[..., 4:-4, 4:-4]
        hf_mask = _hf_mask_from_gt(hr)[..., 4:-4, 4:-4]

        if run_bicubic:
            p01 = (lr_up + 1.0) / 2.0
            py = rgb01_to_y01(p01)[..., 4:-4, 4:-4]
            metrics["bicubic"]["psnr"].append(float(psnr(py, hy, data_range=1.0).item()))
            metrics["bicubic"]["ssim"].append(float(ssim(py, hy, data_range=1.0).item()))
            metrics["bicubic"]["lpips"].append(float(lpips_fn(lr_up.detach().cpu().float(), hr.detach().cpu().float()).mean().item()))
            metrics["bicubic"]["psnr_hf"].append(masked_psnr(py, hy, hf_mask))
            metrics["bicubic"]["ssim_hf"].append(masked_ssim(py, hy, hf_mask))
            if dists_fn is not None:
                metrics["bicubic"]["dists"].append(float(dists_fn((lr_up + 1.0) * 0.5, (hr + 1.0) * 0.5).mean().item()))

        if run_model:
            z_hr = vae.encode(hr).latent_dist.mean * vae.config.scaling_factor
            z_lr_up = vae.encode(lr_up).latent_dist.mean * vae.config.scaling_factor

            if args.use_lq_init:
                latents, run_timesteps = get_lq_init_latents(z_lr_up.to(compute_dtype), scheduler, args.steps, val_gen, args.lq_init_strength, compute_dtype)
            else:
                latents = randn_like_with_generator(z_hr, val_gen)
                run_timesteps = scheduler.timesteps

            aug_level = torch.zeros((latents.shape[0],), device=device, dtype=compute_dtype)
            for t in run_timesteps:
                t_b = torch.tensor([t], device=device).expand(latents.shape[0])
                with torch.no_grad():
                    t_embed = pixart.t_embedder(t_b.to(dtype=compute_dtype))
                with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=(device == "cuda")):
                    cond = adapter(lr_small.to(compute_dtype), t_embed=t_embed)
                    out = pixart(
                        x=latents.to(compute_dtype),
                        timestep=t_b,
                        y=y_embed,
                        aug_level=aug_level,
                        mask=None,
                        data_info=d_info,
                        adapter_cond=cond,
                        force_drop_ids=torch.ones(latents.shape[0], device=device),
                    )
                latents = scheduler.step(out.float(), t, latents.float()).prev_sample

            pred = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)
            p01 = (pred + 1.0) / 2.0
            py = rgb01_to_y01(p01)[..., 4:-4, 4:-4]
            metrics["current_model"]["psnr"].append(float(psnr(py, hy, data_range=1.0).item()))
            metrics["current_model"]["ssim"].append(float(ssim(py, hy, data_range=1.0).item()))
            metrics["current_model"]["lpips"].append(float(lpips_fn(pred.detach().cpu().float(), hr.detach().cpu().float()).mean().item()))
            metrics["current_model"]["psnr_hf"].append(masked_psnr(py, hy, hf_mask))
            metrics["current_model"]["ssim_hf"].append(masked_ssim(py, hy, hf_mask))
            if dists_fn is not None:
                metrics["current_model"]["dists"].append(float(dists_fn((pred + 1.0) * 0.5, (hr + 1.0) * 0.5).mean().item()))

        if args.extra_baseline_dir:
            hr_path = batch["path"][0]
            name = Path(hr_path).name
            pred_path = Path(args.extra_baseline_dir) / name
            if pred_path.exists():
                pred_img = Image.open(pred_path).convert("RGB")
                pred_img = TF.center_crop(pred_img, (args.crop_size, args.crop_size))
                pred = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)(transforms.ToTensor()(pred_img)).unsqueeze(0).to(device)
                p01 = (pred + 1.0) / 2.0
                py = rgb01_to_y01(p01)[..., 4:-4, 4:-4]
                metrics["extra"]["psnr"].append(float(psnr(py, hy, data_range=1.0).item()))
                metrics["extra"]["ssim"].append(float(ssim(py, hy, data_range=1.0).item()))
                metrics["extra"]["lpips"].append(float(lpips_fn(pred.detach().cpu().float(), hr.detach().cpu().float()).mean().item()))
                metrics["extra"]["psnr_hf"].append(masked_psnr(py, hy, hf_mask))
                metrics["extra"]["ssim_hf"].append(masked_ssim(py, hy, hf_mask))
                if dists_fn is not None:
                    metrics["extra"]["dists"].append(float(dists_fn((pred + 1.0) * 0.5, (hr + 1.0) * 0.5).mean().item()))

    def _mean(xs):
        return float(np.mean(xs)) if len(xs) > 0 else float("nan")

    print("\n===== RealSR Paired Evaluation =====")
    for mode in ["bicubic", "current_model", "extra"]:
        if len(metrics[mode]["psnr"]) == 0:
            continue
        print(
            f"[{mode}] N={len(metrics[mode]['psnr'])} "
            f"PSNR={_mean(metrics[mode]['psnr']):.4f} "
            f"SSIM={_mean(metrics[mode]['ssim']):.6f} "
            f"LPIPS={_mean(metrics[mode]['lpips']):.6f} "
            f"PSNR_HF={_mean(metrics[mode]['psnr_hf']):.4f} "
            f"SSIM_HF={_mean(metrics[mode]['ssim_hf']):.6f} "
            + (f"DISTS={_mean(metrics[mode]['dists']):.6f}" if len(metrics[mode]['dists']) > 0 else "DISTS=N/A")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RealSR paired baseline(s) and current model")
    parser.add_argument("--pixart_path", type=str, default=os.path.join(PROJECT_ROOT, "output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth"))
    parser.add_argument("--ckpt_path", type=str, default=os.path.join(PROJECT_ROOT, "experiments_results/train_sigma_sr_vpred_dualstream/checkpoints/best.pth"))
    parser.add_argument("--vae_path", type=str, default=os.path.join(PROJECT_ROOT, "output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers/vae"))
    parser.add_argument("--null_t5_embed_path", type=str, default=os.path.join(PROJECT_ROOT, "output/pretrained_models/null_t5_embed_sigma_300.pth"))
    parser.add_argument("--realsr_roots", type=str, default="/data/RealSR/Nikon/Test/4,/data/RealSR/Canon/Test/4")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--use_lq_init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lq_init_strength", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--baseline", type=str, default="bicubic,current_model", help="bicubic | current_model | bicubic,current_model")
    parser.add_argument("--extra_baseline_dir", type=str, default="", help="optional folder with predicted images named as HR files")
    parser.add_argument("--with_dists", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if "current_model" in args.baseline and (not os.path.exists(args.ckpt_path)):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    if "current_model" in args.baseline and (not os.path.exists(args.pixart_path)):
        raise FileNotFoundError(f"pixart_path not found: {args.pixart_path}")
    if not os.path.exists(args.vae_path):
        raise FileNotFoundError(f"VAE path not found: {args.vae_path}")
    if "current_model" in args.baseline and (not os.path.exists(args.null_t5_embed_path)):
        raise FileNotFoundError(f"null_t5_embed_path not found: {args.null_t5_embed_path}")

    evaluate(args)
