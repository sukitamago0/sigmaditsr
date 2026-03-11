import os
import sys
import glob
import math
import argparse
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import lpips
from diffusers import AutoencoderKL, DDIMScheduler
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import matplotlib.pyplot as plt

from diffusion.model.nets.PixArtSigma_SR import PixArtSigmaSR_XL_2
from diffusion.model.nets.adapter import build_adapter_msm_qca
from diffusion import IDDPM


# ============================== Interpretation Guide ==============================
# 1) Whole-image overfit:
#    - Train loss ↓ and full-image PSNR/SSIM ↑ with LPIPS ↓ => model can memorize this sample globally.
# 2) ROI overfit:
#    - In ROI mode, dataset is already cropped to ROI patch; local ROI = full patch.
#      So full metrics are ROI metrics. We still report both for readability.
# 3) Adapter on/off comparison (--disable_adapter):
#    - ON: adapter tokens from paired lr_small participate in training/inference.
#    - OFF: adapter path disabled and adapter params frozen.
#    - If ON significantly outperforms OFF, gains come from adapter conditioning.
# ===============================================================================


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rgb01_to_y01(rgb01):
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    return (16.0 + 65.481 * r + 128.553 * g + 24.966 * b) / 255.0


def parse_roi(roi_str):
    if roi_str is None or str(roi_str).strip() == "":
        return None
    vals = [int(x.strip()) for x in roi_str.split(",")]
    if len(vals) != 4:
        raise ValueError("--roi must be x1,y1,x2,y2")
    x1, y1, x2, y2 = vals
    if not (x2 > x1 and y2 > y1):
        raise ValueError("ROI must satisfy x2>x1 and y2>y1")
    return x1, y1, x2, y2


def clamp_roi_to_hw(roi, h, w):
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    x1 = max(0, min(w - 1, x1))
    x2 = max(1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI is invalid after clamping")
    return x1, y1, x2, y2


def load_state_dict_shape_compatible(model: nn.Module, state_dict: dict, context: str = "load"):
    curr = model.state_dict()
    filt, skipped = {}, []
    for k, v in state_dict.items():
        if k in curr and tuple(v.shape) == tuple(curr[k].shape):
            filt[k] = v
        else:
            skipped.append(k)
    missing, unexpected = model.load_state_dict(filt, strict=False)
    print(f"[{context}] compatible load: loaded={len(filt)}, skipped={len(skipped)}, missing={len(missing)}, unexpected={len(unexpected)}")
    return missing, unexpected, skipped


def _load_pixart_subset_compatible(pixart: nn.Module, saved_trainable: dict, context: str):
    curr = pixart.state_dict()
    loaded = skipped_shape = missing_in_model = 0
    for k, v in saved_trainable.items():
        if k not in curr:
            missing_in_model += 1
            continue
        if tuple(v.shape) == tuple(curr[k].shape):
            curr[k] = v.to(dtype=curr[k].dtype)
            loaded += 1
        else:
            skipped_shape += 1
    pixart.load_state_dict(curr, strict=False)
    print(f"[{context}] pixart subset load: loaded={loaded}, model_miss={missing_in_model}, shape_skip={skipped_shape}, saved_total={len(saved_trainable)}")


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


def _block_id_from_name(name: str):
    import re
    m = re.search(r"blocks\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def apply_lora_attn_only(model, rank=4, alpha=4):
    cnt = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        bid = _block_id_from_name(name)
        if bid is None or not (0 <= bid <= 27):
            continue
        if not (("attn.qkv" in name) or ("attn.proj" in name)):
            continue
        parent = model.get_submodule(name.rsplit('.', 1)[0])
        child = name.rsplit('.', 1)[1]
        setattr(parent, child, LoRALinear(module, rank, alpha))
        cnt += 1
    print(f"✅ Attention LoRA applied to {cnt} layers (rank={rank}, alpha={alpha}).")


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


class SingleRealSROverfitDataset(Dataset):
    """Return keys aligned to main training: hr/lr/lr_small/path."""

    def __init__(self, roots, pick_index=0, roi=None, crop_size=512):
        self.norm = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        self.to_tensor = transforms.ToTensor()
        self.pairs = []
        self.roi = roi
        self.crop_size = int(crop_size)

        for root in roots:
            if not os.path.isdir(root):
                continue
            for hr_path in sorted(glob.glob(os.path.join(root, "*_HR.png"))):
                lr_path = hr_path.replace("_HR.png", "_LR4.png")
                if os.path.exists(lr_path):
                    self.pairs.append((hr_path, lr_path))

        self.pairs = sorted(self.pairs)
        if len(self.pairs) == 0:
            raise FileNotFoundError(f"No RealSR pairs found in roots={roots}")
        if not (0 <= int(pick_index) < len(self.pairs)):
            raise IndexError(f"pick_index={pick_index} out of range [0,{len(self.pairs)-1}]")
        self.pick_index = int(pick_index)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        del idx
        hr_path, lr_path = self.pairs[self.pick_index]
        hr_pil = Image.open(hr_path).convert("RGB")
        lr_pil = Image.open(lr_path).convert("RGB")
        lr_aligned, hr_aligned = center_crop_aligned_pair(lr_pil, hr_pil, scale=4)

        hr_w, hr_h = hr_aligned.size
        if self.roi is None:
            # whole-image mode still follows main-train-like fixed crop size distribution
            c = int(self.crop_size)
            if hr_h < c or hr_w < c:
                # if sample is smaller than crop size, fallback to aligned full frame
                hr_crop = hr_aligned
                lr_small_pil = lr_aligned
            else:
                hr_crop = TF.center_crop(hr_aligned, (c, c))
                lr_small_pil = TF.center_crop(lr_aligned, (c // 4, c // 4))
        else:
            x1, y1, x2, y2 = clamp_roi_to_hw(self.roi, hr_h, hr_w)
            hr_crop = TF.crop(hr_aligned, y1, x1, y2 - y1, x2 - x1)
            lx1, ly1 = x1 // 4, y1 // 4
            lx2, ly2 = math.ceil(x2 / 4), math.ceil(y2 / 4)
            lr_w, lr_h = lr_aligned.size
            lx1 = max(0, min(lr_w - 1, lx1))
            ly1 = max(0, min(lr_h - 1, ly1))
            lx2 = max(lx1 + 1, min(lr_w, lx2))
            ly2 = max(ly1 + 1, min(lr_h, ly2))
            lr_small_pil = TF.crop(lr_aligned, ly1, lx1, ly2 - ly1, lx2 - lx1)

        hr_h2, hr_w2 = hr_crop.size[1], hr_crop.size[0]
        lr_up_pil = TF.resize(lr_small_pil, (hr_h2, hr_w2), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

        hr_tensor = self.norm(self.to_tensor(hr_crop))
        lr_tensor = self.norm(self.to_tensor(lr_up_pil))
        lr_small_tensor = self.norm(self.to_tensor(lr_small_pil))

        # In ROI mode, local-ROI for patch space is the whole patch.
        local_roi = (0, 0, hr_w2, hr_h2) if self.roi is not None else None

        return {
            "hr": hr_tensor,
            "lr": lr_tensor,
            "lr_small": lr_small_tensor,
            "path": hr_path,
            "local_roi": local_roi,
        }


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


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    if not torch.is_tensor(arr):
        arr = torch.tensor(arr, dtype=torch.float32, device=timesteps.device)
    out = arr.to(device=timesteps.device)[timesteps].float()
    while len(out.shape) < len(broadcast_shape):
        out = out[..., None]
    return out.expand(broadcast_shape)


# ===== loss pieces aligned with train_sigma_sr_vpred_dualstream.py =====
_SOBEL_X = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
_LAPLACE = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)


def _to_luma01(img_m11: torch.Tensor) -> torch.Tensor:
    img01 = (img_m11.float() + 1.0) * 0.5
    r = img01[:, 0:1]
    g = img01[:, 1:2]
    b = img01[:, 2:3]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).clamp(0.0, 1.0)


@torch.cuda.amp.autocast(enabled=False)
def edge_mask_from_gt(gt_m11: torch.Tensor, q: float = 0.90, pow_: float = 0.50, eps: float = 1e-6) -> torch.Tensor:
    x = _to_luma01(gt_m11)
    gx = F.conv2d(x, _SOBEL_X.to(x.device), padding=1)
    gy = F.conv2d(x, _SOBEL_Y.to(x.device), padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + eps)
    flat = mag.flatten(1)
    denom = torch.quantile(flat, q, dim=1, keepdim=True).clamp_min(eps)
    m = (flat / denom).view_as(mag).clamp(0.0, 1.0)
    return m.pow(pow_) if pow_ != 1.0 else m


@torch.cuda.amp.autocast(enabled=False)
def edge_guided_losses(pred_m11: torch.Tensor, gt_m11: torch.Tensor, q: float = 0.90, pow_: float = 0.50, eps: float = 1e-6):
    m = edge_mask_from_gt(gt_m11, q=q, pow_=pow_, eps=eps)
    p = _to_luma01(pred_m11)
    g = _to_luma01(gt_m11)
    pgx = F.conv2d(p, _SOBEL_X.to(p.device), padding=1)
    pgy = F.conv2d(p, _SOBEL_Y.to(p.device), padding=1)
    ggx = F.conv2d(g, _SOBEL_X.to(g.device), padding=1)
    ggy = F.conv2d(g, _SOBEL_Y.to(g.device), padding=1)
    loss_edge = (m * (pgx - ggx).abs() + m * (pgy - ggy).abs()).mean()
    plap = F.conv2d(p, _LAPLACE.to(p.device), padding=1)
    loss_flat_hf = ((1.0 - m) * plap.abs()).mean()
    return loss_edge, loss_flat_hf, m


@torch.cuda.amp.autocast(enabled=False)
def structure_consistency_loss(pred_m11: torch.Tensor, lr_m11: torch.Tensor) -> torch.Tensor:
    pred_lr = F.interpolate(pred_m11.float(), size=lr_m11.shape[-2:], mode='bilinear', align_corners=False)
    p = _to_luma01(pred_lr)
    l = _to_luma01(lr_m11.float())
    pgx = F.conv2d(p, _SOBEL_X.to(p.device), padding=1)
    pgy = F.conv2d(p, _SOBEL_Y.to(p.device), padding=1)
    lgx = F.conv2d(l, _SOBEL_X.to(l.device), padding=1)
    lgy = F.conv2d(l, _SOBEL_Y.to(l.device), padding=1)
    pl = F.conv2d(p, _LAPLACE.to(p.device), padding=1)
    ll = F.conv2d(l, _LAPLACE.to(l.device), padding=1)
    loss_sobel = (pgx - lgx).abs().mean() + (pgy - lgy).abs().mean()
    loss_lap = (pl - ll).abs().mean()
    p_low = F.avg_pool2d(p, kernel_size=5, stride=1, padding=2)
    l_low = F.avg_pool2d(l, kernel_size=5, stride=1, padding=2)
    loss_lowfreq = F.l1_loss(p_low, l_low)
    return 0.4 * loss_sobel + 0.4 * loss_lap + 0.2 * loss_lowfreq


def get_fixed_loss_weights():
    return {
        "latent_l1": 0.10,
        "lr_cons": 0.05,
        "edge_grad": 0.01,
        "flat_hf": 0.00,
    }


@torch.no_grad()
def run_formal_inference(pixart, adapter, vae, hr, lr, lr_small, y_embed, data_info, args, device, compute_dtype, lpips_fn):
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="v_prediction",
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(args.infer_steps, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    z_hr = vae.encode(hr).latent_dist.mean * vae.config.scaling_factor
    z_lr = vae.encode(lr).latent_dist.mean * vae.config.scaling_factor
    if args.use_lq_init:
        latents, run_timesteps = get_lq_init_latents(z_lr.to(compute_dtype), scheduler, args.infer_steps, gen, args.lq_init_strength, compute_dtype)
    else:
        latents = randn_like_with_generator(z_hr.to(compute_dtype), gen)
        run_timesteps = scheduler.timesteps

    aug_level = torch.zeros((latents.shape[0],), device=device, dtype=compute_dtype)
    for t in run_timesteps:
        t_b = torch.tensor([t], device=device).expand(latents.shape[0])
        t_embed = pixart.t_embedder(t_b.to(dtype=compute_dtype))
        cond = None
        if not args.disable_adapter:
            with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=(device == "cuda")):
                cond = adapter(lr_small.to(dtype=compute_dtype), t_embed=t_embed)

        with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=(device == "cuda")):
            out = pixart(
                x=latents.to(compute_dtype), timestep=t_b, y=y_embed,
                aug_level=aug_level, mask=None, data_info=data_info,
                adapter_cond=cond, force_drop_ids=torch.ones(latents.shape[0], device=device),
            )
        latents = scheduler.step(out.float(), t, latents.float()).prev_sample

    pred = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)
    pred01 = (pred + 1.0) * 0.5
    hr01 = (hr + 1.0) * 0.5

    py = rgb01_to_y01(pred01)[..., 4:-4, 4:-4]
    hy = rgb01_to_y01(hr01)[..., 4:-4, 4:-4]
    m_psnr = float(psnr(py, hy, data_range=1.0).item())
    m_ssim = float(ssim(py, hy, data_range=1.0).item())
    m_lpips = float(lpips_fn(pred.detach().cpu().float(), hr.detach().cpu().float()).mean().item())

    roi_metrics = None
    if args.local_roi is not None:
        x1, y1, x2, y2 = args.local_roi
        pr_roi = pred01[..., y1:y2, x1:x2]
        hr_roi = hr01[..., y1:y2, x1:x2]
        pry = rgb01_to_y01(pr_roi)
        hry = rgb01_to_y01(hr_roi)
        roi_metrics = {
            "psnr": float(psnr(pry, hry, data_range=1.0).item()),
            "ssim": float(ssim(pry, hry, data_range=1.0).item()),
            "lpips": float(lpips_fn((pr_roi * 2 - 1).detach().cpu().float(), (hr_roi * 2 - 1).detach().cpu().float()).mean().item()),
        }

    return pred, {"full": {"psnr": m_psnr, "ssim": m_ssim, "lpips": m_lpips}, "roi": roi_metrics}


def save_visuals(out_dir, step, hr, lr, pred, local_roi):
    os.makedirs(out_dir, exist_ok=True)
    hr_np = (hr[0].detach().cpu().float().permute(1, 2, 0).numpy() + 1.0) * 0.5
    lr_np = (lr[0].detach().cpu().float().permute(1, 2, 0).numpy() + 1.0) * 0.5
    pr_np = (pred[0].detach().cpu().float().permute(1, 2, 0).numpy() + 1.0) * 0.5

    fig = plt.figure(figsize=(12, 4))
    for i, (img, title) in enumerate([(lr_np, "LR(up)"), (hr_np, "HR GT"), (pr_np, "Pred")], start=1):
        ax = plt.subplot(1, 3, i)
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(os.path.join(out_dir, f"step_{step:06d}_full_triptych.png"), bbox_inches="tight")
    plt.close(fig)

    if local_roi is not None:
        x1, y1, x2, y2 = local_roi
        lr_roi = lr_np[y1:y2, x1:x2]
        pr_roi = pr_np[y1:y2, x1:x2]
        hr_roi = hr_np[y1:y2, x1:x2]
        err = np.mean(np.abs(pr_roi - hr_roi), axis=2)
        err = err / (err.max() + 1e-8)

        fig = plt.figure(figsize=(14, 4))
        ax1 = plt.subplot(1, 4, 1); ax1.imshow(np.clip(lr_roi, 0, 1)); ax1.set_title("ROI LR(up)"); ax1.axis("off")
        ax2 = plt.subplot(1, 4, 2); ax2.imshow(np.clip(pr_roi, 0, 1)); ax2.set_title("ROI Pred"); ax2.axis("off")
        ax3 = plt.subplot(1, 4, 3); ax3.imshow(np.clip(hr_roi, 0, 1)); ax3.set_title("ROI GT"); ax3.axis("off")
        ax4 = plt.subplot(1, 4, 4); im = ax4.imshow(err, cmap="jet", vmin=0.0, vmax=1.0); ax4.set_title("ROI Error Heatmap"); ax4.axis("off")
        fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        fig.savefig(os.path.join(out_dir, f"step_{step:06d}_roi_quad.png"), bbox_inches="tight")
        plt.close(fig)


def configure_trainable(pixart, adapter, disable_adapter=False, bridge_only_debug=False):
    for _, p in pixart.named_parameters():
        p.requires_grad_(False)
    for _, p in adapter.named_parameters():
        p.requires_grad_(False)

    allow = ["adapter_ca_norm_q", "adapter_ca_layers", "adapter_ca_out", "adapter_ca_gate", "final_layer", "lora_A", "lora_B"]
    for n, p in pixart.named_parameters():
        if any(k in n for k in allow):
            p.requires_grad_(True)

    if not disable_adapter:
        for _, p in adapter.named_parameters():
            p.requires_grad_(True)
        if bridge_only_debug:
            for n, p in adapter.named_parameters():
                p.requires_grad_(any(k in n for k in ["mem_proj_", "resampler", "memory_out_proj", "memory_ln", "scale_embed", "q2", "q3", "q4"]))


def make_optimizer(pixart, adapter, disable_adapter=False, pixart_lr=1e-5, adapter_lr=3e-5):
    # align semantics with main training: memory_bridge / adapter_backbone / pixart_readout_bridge / pixart_low_lr
    pixart_readout_bridge, pixart_low_lr = [], []
    for n, p in pixart.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in ["adapter_ca_norm_q", "adapter_ca_layers", "adapter_ca_out", "adapter_ca_gate"]):
            pixart_readout_bridge.append(p)
        else:
            pixart_low_lr.append(p)

    groups = []
    if len(pixart_readout_bridge) > 0:
        groups.append({"params": pixart_readout_bridge, "lr": max(5e-5, float(pixart_lr)), "name": "pixart_readout_bridge"})
    if len(pixart_low_lr) > 0:
        groups.append({"params": pixart_low_lr, "lr": min(5e-6, float(pixart_lr)), "name": "pixart_low_lr"})

    if not disable_adapter:
        memory_bridge, adapter_backbone = [], []
        for n, p in adapter.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in n for k in ["mem_proj_", "resampler", "memory_out_proj", "memory_ln", "scale_embed", "q2", "q3", "q4"]):
                memory_bridge.append(p)
            else:
                adapter_backbone.append(p)
        if len(memory_bridge) > 0:
            groups.append({"params": memory_bridge, "lr": max(1e-4, float(adapter_lr)), "name": "memory_bridge"})
        if len(adapter_backbone) > 0:
            groups.append({"params": adapter_backbone, "lr": float(adapter_lr), "name": "adapter_backbone"})

    return torch.optim.AdamW(groups, weight_decay=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixart_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--null_t5_embed_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--init_mode", type=str, default="warm", choices=["warm", "cold"])
    parser.add_argument("--realsr_roots", type=str, default="/data/RealSR/Nikon/Test/4,/data/RealSR/Canon/Test/4")
    parser.add_argument("--pick_index", type=int, default=0)
    parser.add_argument("--roi", type=str, default="")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--disable_adapter", action="store_true")
    parser.add_argument("--bridge_only_debug", action="store_true")
    parser.add_argument("--train_steps", type=int, default=400)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--infer_steps", type=int, default=50)
    parser.add_argument("--pixart_lr", type=float, default=1e-5)
    parser.add_argument("--adapter_lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=4)
    parser.add_argument("--use_lq_init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lq_init_strength", type=float, default=0.3)
    parser.add_argument("--out_dir", type=str, default="outputs/overfit_single_realsr")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    roi = parse_roi(args.roi)
    roots = [x.strip() for x in args.realsr_roots.split(",") if x.strip()]
    ds = SingleRealSROverfitDataset(roots=roots, pick_index=args.pick_index, roi=roi, crop_size=args.crop_size)
    batch = ds[0]

    hr = batch["hr"].unsqueeze(0).to(device)
    lr = batch["lr"].unsqueeze(0).to(device)
    lr_small = batch["lr_small"].unsqueeze(0).to(device)

    # ROI mode bug fix: once sample is cropped to ROI patch, use local patch ROI only.
    if batch["local_roi"] is not None:
        args.local_roi = batch["local_roi"]
    else:
        h, w = hr.shape[-2:]
        args.local_roi = clamp_roi_to_hw(roi, h, w) if roi is not None else None

    pixart = PixArtSigmaSR_XL_2(input_size=64, in_channels=4, out_channels=4).to(device)
    base = torch.load(args.pixart_path, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    if hasattr(pixart, "load_pretrained_weights_with_zero_init"):
        pixart.load_pretrained_weights_with_zero_init(base)
    else:
        load_state_dict_shape_compatible(pixart, base, context="base-pretrain")

    adapter = build_adapter_msm_qca(in_channels=3, hidden_size=1152, injection_layers_map=getattr(pixart, "injection_layers", None)).to(device).float()

    print(f"[Overfit] init_mode={args.init_mode}")
    if args.init_mode == "warm":
        if not args.ckpt_path or (not os.path.exists(args.ckpt_path)):
            raise FileNotFoundError("init_mode=warm requires a valid --ckpt_path")
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        saved_trainable = ckpt.get("pixart_keep", ckpt.get("pixart_trainable", {}))
        has_lora = any(("lora_A" in k) or ("lora_B" in k) for k in saved_trainable.keys())
        lora_rank = int(ckpt["lora_rank"]) if "lora_rank" in ckpt else int(args.lora_rank)
        lora_alpha = int(ckpt["lora_alpha"]) if "lora_alpha" in ckpt else int(args.lora_alpha)
        # same architecture in warm/cold: always materialize LoRA layers
        apply_lora_attn_only(pixart, rank=lora_rank if has_lora else int(args.lora_rank), alpha=lora_alpha if has_lora else int(args.lora_alpha))
        _load_pixart_subset_compatible(pixart, saved_trainable, context="overfit")
        if "adapter" in ckpt:
            load_state_dict_shape_compatible(adapter, ckpt["adapter"], context="overfit-adapter")
    elif args.init_mode == "cold":
        # Cold-start: same architecture, random LoRA/bridge/adapters, no warm weights from best checkpoint.
        apply_lora_attn_only(pixart, rank=int(args.lora_rank), alpha=int(args.lora_alpha))
    else:
        raise ValueError(f"Unknown init_mode={args.init_mode}")

    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device).float().eval()
    vae.enable_slicing()
    null_pack = torch.load(args.null_t5_embed_path, map_location="cpu")
    y_embed = null_pack["y"].to(device)
    h, w = hr.shape[-2:]
    data_info = {
        "img_hw": torch.tensor([[float(h), float(w)]], device=device),
        "aspect_ratio": torch.tensor([float(w) / float(h)], device=device),
    }

    lpips_fn = lpips.LPIPS(net='vgg').to("cpu").eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # Disable EMA/condition dropout/condition noise/augmentation sampling by design.
    configure_trainable(pixart, adapter, disable_adapter=args.disable_adapter, bridge_only_debug=args.bridge_only_debug)
    optimizer = make_optimizer(
        pixart,
        adapter,
        disable_adapter=args.disable_adapter,
        pixart_lr=args.pixart_lr,
        adapter_lr=args.adapter_lr,
    )

    diffusion = IDDPM(str(1000))
    pixart.train()
    adapter.train()

    out_dir = args.out_dir if args.tag == "" else os.path.join(args.out_dir, args.tag)

    print(f"[Overfit] sample={batch['path']} hr={tuple(hr.shape)} lr={tuple(lr.shape)} lr_small={tuple(lr_small.shape)} disable_adapter={args.disable_adapter}")
    if args.local_roi is not None:
        print(f"[Overfit] local ROI={args.local_roi} (x1,y1,x2,y2)")
    print(f"[Overfit] optimizer lrs: pixart_lr={args.pixart_lr}, adapter_lr={args.adapter_lr}")

    for step in range(1, args.train_steps + 1):
        with torch.no_grad():
            z_hr = vae.encode(hr).latent_dist.mean * vae.config.scaling_factor

        t = torch.randint(0, 1000, (z_hr.shape[0],), device=device).long()
        noise = torch.randn_like(z_hr)
        zt = diffusion.q_sample(z_hr, t, noise)

        with torch.no_grad():
            t_embed = pixart.t_embedder(t.to(dtype=compute_dtype))

        cond = None
        if not args.disable_adapter:
            with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=(device == "cuda")):
                cond = adapter(lr_small.to(dtype=compute_dtype), t_embed=t_embed)
            if step == 1 and isinstance(cond, dict) and torch.is_tensor(cond.get("memory_tokens", None)):
                print(f"[ShapeCheck][overfit] memory_tokens={tuple(cond['memory_tokens'].shape)}")

        with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=(device == "cuda")):
            out = pixart(
                x=zt.to(compute_dtype),
                timestep=t,
                y=y_embed,
                aug_level=torch.zeros((zt.shape[0],), device=device, dtype=compute_dtype),
                mask=None,
                data_info=data_info,
                adapter_cond=cond,
                force_drop_ids=torch.ones(zt.shape[0], device=device),
            )
            model_pred = out.float()

            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, z_hr.shape)
            sigma_t = _extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, t, z_hr.shape)
            target_v = alpha_t * noise - sigma_t * z_hr.float()
            loss_v = F.mse_loss(model_pred, target_v)

            z0 = alpha_t * zt.float() - sigma_t * model_pred
            loss_latent_l1 = F.l1_loss(z0, z_hr.float())

            pred = vae.decode(z0 / vae.config.scaling_factor).sample.clamp(-1, 1)
            loss_lr_cons = structure_consistency_loss(pred, lr)
            loss_edge, _, _ = edge_guided_losses(pred, hr)

            wloss = get_fixed_loss_weights()
            loss = loss_v + wloss["latent_l1"] * loss_latent_l1 + wloss["lr_cons"] * loss_lr_cons + wloss["edge_grad"] * loss_edge

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1:
            print(
                f"[Step {step:04d}] loss={float(loss.item()):.6f} "
                f"v={float(loss_v.item()):.6f} z_l1={float(loss_latent_l1.item()):.6f} "
                f"lr_cons={float(loss_lr_cons.item()):.6f} edge={float(loss_edge.item()):.6f}"
            )

        if (step % args.save_every == 0) or (step == args.train_steps):
            pixart.eval()
            adapter.eval()
            with torch.no_grad():
                pred_eval, metrics = run_formal_inference(
                    pixart, adapter, vae, hr, lr, lr_small, y_embed, data_info, args, device, compute_dtype, lpips_fn
                )

            ca_g = getattr(pixart, "_last_adapter_ca_gates", None)
            if torch.is_tensor(ca_g):
                ca_gate_mean = float(ca_g.mean().item())
                ca_gate_norm = float(torch.norm(ca_g, p=2).item())
            else:
                ca_gate_mean = 0.0
                ca_gate_norm = 0.0
            print(
                f"[Eval step {step}] full: PSNR={metrics['full']['psnr']:.4f}, "
                f"SSIM={metrics['full']['ssim']:.4f}, LPIPS={metrics['full']['lpips']:.4f} | "
                f"ca_gate(mean/norm)=({ca_gate_mean:.4f}/{ca_gate_norm:.4f})"
            )
            if metrics["roi"] is not None:
                print(
                    f"[Eval step {step}] roi : PSNR={metrics['roi']['psnr']:.4f}, "
                    f"SSIM={metrics['roi']['ssim']:.4f}, LPIPS={metrics['roi']['lpips']:.4f}"
                )

            save_visuals(out_dir, step, hr, lr, pred_eval, args.local_roi)
            pixart.train()
            adapter.train()


if __name__ == "__main__":
    main()
