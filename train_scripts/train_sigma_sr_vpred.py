# /home/hello/HJT/DiTSR/experiments/train_4090_auto_v8.py
# DiTSR v8 Training Script (Final Corrected Version)
# ------------------------------------------------------------------
# Fixes included:
# 1. [Optim] Fixed parameter grouping (Mutually Exclusive).
# 2. [Process] Unlocked x_embedder learning rate (1e-4).
# 3. [Process] Disabled LPIPS for structure convergence (Stage 1).
# 4. [Structure] Fixed NameError by ensuring Dataset classes are defined.
# ------------------------------------------------------------------

import os
import sys
from pathlib import Path

# ================= 1. Path Setup =================
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import hashlib
import shutil
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import lpips
from diffusers import AutoencoderKL, DDIMScheduler
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

# [Import V8 Model]
from diffusion.model.nets.PixArtSigma_SR import PixArtSigmaSR_XL_2
from diffusion.model.nets.adapter import build_adapter_v7
from diffusion import IDDPM
from diffusion.model.gaussian_diffusion import _extract_into_tensor

BASE_PIXART_SHA256 = None

# Added "aug_embedder" to required keys
V7_REQUIRED_PIXART_KEY_FRAGMENTS = (
    "input_adaln", "adapter_alpha_mlp", "input_res_proj",
    "input_adapter_ln", "style_fusion_mlp", "aug_embedder", "injection_scales"
)
FP32_SAVE_KEY_FRAGMENTS = V7_REQUIRED_PIXART_KEY_FRAGMENTS

def get_required_v7_key_fragments_for_model(model: nn.Module):
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    required = []
    for frag in V7_REQUIRED_PIXART_KEY_FRAGMENTS:
        if any(frag in name for name in trainable_names):
            required.append(frag)
    return tuple(required)

# ================= 2. Hyper-parameters =================
TRAIN_DF2K_HR_DIR = "/data/DF2K/DF2K_train_HR"
TRAIN_DF2K_LR_DIR = "/data/DF2K/DF2K_train_LR_unknown"
TRAIN_REALSR_DIRS = [
    "/data/RealSR/Canon/Train/4",
    "/data/RealSR/Nikon/Train/4",
]
VAL_HR_DIR   = "/data/DF2K/DF2K_valid_HR"
VAL_LR_DIR_CANDIDATES = [
    "/data/DF2K/DF2K_valid_LR_unknown/X4",
    "/data/DF2K/DF2K_valid_LR_bicubic/X4",
]
VAL_LR_DIR = next((p for p in VAL_LR_DIR_CANDIDATES if os.path.exists(p)), None)

PIXART_PATH = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "PixArt-Sigma-XL-2-512-MS.pth")
DIFFUSERS_ROOT = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "pixart_sigma_sdxlvae_T5_diffusers")
VAE_PATH = os.path.join(DIFFUSERS_ROOT, "vae")
NULL_T5_EMBED_PATH = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "null_t5_embed_sigma_300.pth")

OUT_BASE = os.getenv("DTSR_OUT_BASE", os.path.join(PROJECT_ROOT, "experiments_results"))
OUT_DIR = os.path.join(OUT_BASE, "train_sigma_sr_vpred")
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
VIS_DIR  = os.path.join(OUT_DIR, "vis")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
LAST_CKPT_PATH = os.path.join(CKPT_DIR, "last.pth")

DEVICE = "cuda"
COMPUTE_DTYPE = torch.bfloat16
SEED = 3407
DETERMINISTIC = True
FAST_DEV_RUN = os.getenv("FAST_DEV_RUN", "0") == "1"
FAST_TRAIN_STEPS = int(os.getenv("FAST_TRAIN_STEPS", "10"))
FAST_VAL_BATCHES = int(os.getenv("FAST_VAL_BATCHES", "2"))
FAST_VAL_STEPS = int(os.getenv("FAST_VAL_STEPS", "10"))
MAX_TRAIN_STEPS = int(os.getenv("MAX_TRAIN_STEPS", "0"))

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16 
NUM_WORKERS = 8

LR_BASE = 1e-5 
LORA_RANK = 16
LORA_ALPHA = 16
TRAIN_PIXART_X_EMBEDDER = True  # enable concat LR latent path learning in x_embedder
SPARSE_INJECT_RATIO = 1.0
INJECTION_CUTOFF_LAYER = 28
INJECTION_STRATEGY = "full"

# [V8 Augmentation]
COND_AUG_NOISE_RANGE = (0.0, 0.05) 

WARMUP_STEPS = 4000         
RAMP_UP_STEPS = 5000         
TARGET_LPIPS_WEIGHT = 0.50
LPIPS_BASE_WEIGHT = 0.0      
L1_BASE_WEIGHT = 1.0
EDGE_GRAD_WEIGHT = 0.10     # edge-region gradient matching
FLAT_HF_WEIGHT   = 0.05     # flat/defocus HF suppression (Laplacian)
EDGE_Q           = 0.90     # GT edge quantile for normalization
EDGE_POW         = 0.50     # mask sharpening ( <1 boosts weak edges )
EDGE_WARMUP_STEPS = 3000
EDGE_RAMP_STEPS = 4000

VAL_STEPS_LIST = [50]
BEST_VAL_STEPS = 50
PSNR_SWITCH = 24.0
KEEP_TOPK = 2
VAL_MODE = "lr_dir"
VAL_PACK_DIR = os.path.join(PROJECT_ROOT, "valpacks", "df2k_train_like_50_seed3407")
VAL_PACK_LR_DIR_NAME = "lq512"
TRAIN_DEG_MODE = "highorder"
CFG_SCALE = 1.0

# [V8 Change] Default to LQ-Init for validation
USE_LQ_INIT = True 
LQ_INIT_STRENGTH = 0.1

INIT_NOISE_STD = 0.0
USE_ADAPTER_CFDROPOUT = True
COND_DROP_PROB = 0.10
FORCE_DROP_TEXT = True  # validation-time text drop behavior
INJECT_SCALE_REG_LAMBDA = 1e-4
PIXEL_LOSS_T_MAX = 250
PIXEL_LOSS_START_STEP = WARMUP_STEPS

USE_LR_CONSISTENCY = False 
USE_NOISE_CONSISTENCY = False

VAE_TILING = False
DEG_OPS = ["blur", "resize", "noise", "jpeg"]
P_TWO_STAGE = 0.35
RESIZE_SCALE_RANGE = (0.3, 1.8)
NOISE_RANGE = (0.0, 0.05)
BLUR_KERNELS = [7, 9, 11, 13, 15, 21]
JPEG_QUALITY_RANGE = (30, 95)
RESIZE_INTERP_MODES = [transforms.InterpolationMode.NEAREST, transforms.InterpolationMode.BILINEAR, transforms.InterpolationMode.BICUBIC]

# ================= 3. Logic Functions =================
def get_loss_weights(global_step):
    weights = {'mse': 1.0, 'latent_l1': L1_BASE_WEIGHT}
    if global_step < WARMUP_STEPS:
        weights['lpips'] = 0.0
    elif global_step < (WARMUP_STEPS + RAMP_UP_STEPS):
        progress = (global_step - WARMUP_STEPS) / RAMP_UP_STEPS
        weights['lpips'] = LPIPS_BASE_WEIGHT + (TARGET_LPIPS_WEIGHT - LPIPS_BASE_WEIGHT) * progress
    else:
        weights['lpips'] = TARGET_LPIPS_WEIGHT

    if global_step < EDGE_WARMUP_STEPS:
        edge_progress = 0.0
    elif EDGE_RAMP_STEPS <= 0:
        edge_progress = 1.0
    else:
        edge_progress = min(1.0, (global_step - EDGE_WARMUP_STEPS) / EDGE_RAMP_STEPS)
    weights['edge_grad'] = EDGE_GRAD_WEIGHT * edge_progress
    weights['flat_hf'] = FLAT_HF_WEIGHT * edge_progress
    return weights

def rgb01_to_y01(rgb01):
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    return (16.0 + 65.481*r + 128.553*g + 24.966*b) / 255.0

# ----------------- Edge-guided perceptual regularizers (GT-driven) -----------------
# Goal: (1) match gradients where GT has edges, (2) suppress high-frequency hallucinations where GT is flat/defocused.
_SOBEL_X = torch.tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
_LAPLACE = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

def _to_luma01(img_m11: torch.Tensor) -> torch.Tensor:
    # img in [-1,1], returns luma in [0,1], shape [B,1,H,W], float32
    img01 = (img_m11.float() + 1.0) * 0.5
    r = img01[:, 0:1]; g = img01[:, 1:2]; b = img01[:, 2:3]
    luma = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return luma.clamp(0.0, 1.0)

@torch.cuda.amp.autocast(enabled=False)
def edge_mask_from_gt(gt_m11: torch.Tensor, q: float = 0.90, pow_: float = 0.50, eps: float = 1e-6) -> torch.Tensor:
    # Return mask in [0,1], high on GT edges, low on flat/defocus regions.
    x = _to_luma01(gt_m11)
    kx = _SOBEL_X.to(device=x.device); ky = _SOBEL_Y.to(device=x.device)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + eps)
    # Robust normalization using per-image quantile (prevents a few strong edges from saturating everything).
    flat = mag.flatten(1)
    denom = torch.quantile(flat, q, dim=1, keepdim=True).clamp_min(eps)
    m = (flat / denom).view_as(mag).clamp(0.0, 1.0)
    if pow_ != 1.0:
        m = m.pow(pow_)
    return m

@torch.cuda.amp.autocast(enabled=False)
def edge_guided_losses(pred_m11: torch.Tensor, gt_m11: torch.Tensor, q: float = 0.90, pow_: float = 0.50, eps: float = 1e-6):
    # pred/gt in [-1,1], shape [B,3,H,W]
    m = edge_mask_from_gt(gt_m11, q=q, pow_=pow_, eps=eps)  # [B,1,H,W]
    p = _to_luma01(pred_m11); g = _to_luma01(gt_m11)
    kx = _SOBEL_X.to(device=p.device); ky = _SOBEL_Y.to(device=p.device); kl = _LAPLACE.to(device=p.device)
    pgx = F.conv2d(p, kx, padding=1); pgy = F.conv2d(p, ky, padding=1)
    ggx = F.conv2d(g, kx, padding=1); ggy = F.conv2d(g, ky, padding=1)
    # (A) Edge matching: only care where GT has edges (prevents "inventing edges" in defocus).
    loss_edge = (m * (pgx - ggx).abs() + m * (pgy - ggy).abs()).mean()
    # (B) HF suppression on flat regions: penalize Laplacian energy where GT is flat/defocused.
    plap = F.conv2d(p, kl, padding=1)
    loss_flat_hf = ((1.0 - m) * plap.abs()).mean()
    return loss_edge, loss_flat_hf, m
# -------------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC: torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed); np.random.seed(worker_seed); torch.manual_seed(worker_seed)

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

def mask_adapter_cond(cond, keep_mask: torch.Tensor):
    if cond is None: return None
    if not torch.is_tensor(keep_mask): keep_mask = torch.tensor(keep_mask)
    def _find_device_dtype(x):
        if torch.is_tensor(x): return x.device, x.dtype
        if isinstance(x, (list, tuple)):
            for item in x:
                found = _find_device_dtype(item)
                if found is not None: return found
        return None
    found = _find_device_dtype(cond)
    if found is None: return cond
    dev, _ = found
    keep_mask = keep_mask.to(device=dev, dtype=torch.float32)
    def _mask(x: torch.Tensor):
        m = keep_mask
        while m.ndim < x.ndim: m = m.unsqueeze(-1)
        return x * m.to(dtype=x.dtype)
    if torch.is_tensor(cond): return _mask(cond)
    if isinstance(cond, dict):
        return {k: (_mask(v) if torch.is_tensor(v) else v) for k, v in cond.items()}
    if isinstance(cond, (list, tuple)):
        if len(cond) == 2 and isinstance(cond[0], list) and torch.is_tensor(cond[1]):
            spatial = [_mask(c) for c in cond[0]]
            style = _mask(cond[1])
            return (spatial, style)
        masked = []
        for c in cond:
            if torch.is_tensor(c): masked.append(_mask(c))
            elif isinstance(c, list): masked.append([_mask(ci) if torch.is_tensor(ci) else ci for ci in c])
            else: masked.append(c)
        return masked if isinstance(cond, list) else tuple(masked)
    return cond

def file_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""): sha.update(chunk)
    return sha.hexdigest()

def _should_keep_fp32_on_save(param_name: str) -> bool:
    return any(tag in param_name for tag in FP32_SAVE_KEY_FRAGMENTS)

def collect_trainable_state_dict(model: nn.Module):
    state = {}
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        tensor = param.detach().cpu()
        if _should_keep_fp32_on_save(name): tensor = tensor.float()
        state[name] = tensor
    return state

def validate_v7_trainable_state_keys(trainable_sd: dict, required_fragments):
    keys = list(trainable_sd.keys())
    missing = []
    counts = {}
    for frag in required_fragments:
        c = sum(1 for k in keys if frag in k)
        counts[frag] = c
        if c == 0: missing.append(frag)
    if missing:
        raise RuntimeError("v7 trainable checkpoint validation failed: " + ", ".join(missing))
    return counts

def compute_injection_scale_reg(model: nn.Module, lambda_reg: float = 1e-4):
    if lambda_reg <= 0 or not hasattr(model, "injection_scales") or not hasattr(model, "injection_layers"):
        return torch.tensor(0.0, device=DEVICE)
    depth = max(1, len(getattr(model, "blocks", [])) - 1)
    reg = torch.tensor(0.0, device=DEVICE)
    for i, p_scale in enumerate(model.injection_scales):
        lid = int(model.injection_layers[i])
        u = float(lid) / float(depth)
        reg = reg + (u * u) * (F.softplus(p_scale) ** 2).mean()
    return reg * float(lambda_reg)

def log_injection_scale_stats(model: nn.Module, prefix: str = "[InjectScale]"):
    if not hasattr(model, "injection_scales"):
        return
    vals = []
    for p in model.injection_scales:
        vals.append(float(F.softplus(p.detach().float()).mean().item()))
    if len(vals) == 0:
        return
    k = min(5, len(vals))
    front_mean = float(np.mean(vals[:k]))
    back_mean = float(np.mean(vals[-k:]))
    print(f"{prefix} front{k}_mean={front_mean:.4f} back{k}_mean={back_mean:.4f} min={min(vals):.4f} max={max(vals):.4f} (softplus)")

def get_config_snapshot():
    return {
        "batch_size": BATCH_SIZE,
        "lr_base": LR_BASE,
        "lora_rank": LORA_RANK,
        "sparse_inject_ratio": SPARSE_INJECT_RATIO,
        "lr_latent_noise_std": INIT_NOISE_STD,
        "loss_weights": "Dynamic",
        "seed": SEED,
    }

# ================= 4. Data Pipeline =================
class DegradationPipeline:
    def __init__(self, crop_size=512):
        self.crop_size = crop_size
        self.blur_kernels = BLUR_KERNELS
        self.blur_sigma_range = (0.2, 2.0)
        self.aniso_sigma_range = (0.2, 2.5)
        self.aniso_theta_range = (0.0, math.pi)
        self.noise_range = NOISE_RANGE
        self.downscale_factor = 0.25 

    def _sample_uniform(self, low, high, generator):
        if generator is None: return float(random.uniform(low, high))
        return float(low + (high - low) * torch.rand((), generator=generator).item())

    def _sample_int(self, low, high, generator):
        if generator is None: return int(random.randint(low, high))
        return int(torch.randint(low, high + 1, (1,), generator=generator).item())

    def _sample_choice(self, choices, generator):
        if generator is None: return random.choice(choices)
        idx = int(torch.randint(0, len(choices), (1,), generator=generator).item())
        return choices[idx]

    def _build_aniso_kernel(self, k, sigma_x, sigma_y, theta, device, dtype):
        ax = torch.arange(-(k // 2), k // 2 + 1, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        c, s = math.cos(theta), math.sin(theta)
        x_rot = c * xx + s * yy
        y_rot = -s * xx + c * yy
        kernel = torch.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))
        kernel = kernel / kernel.sum()
        return kernel

    def _apply_aniso_blur(self, img, k, sigma_x, sigma_y, theta):
        kernel = self._build_aniso_kernel(k, sigma_x, sigma_y, theta, img.device, img.dtype)
        kernel = kernel.view(1, 1, k, k)
        weight = kernel.repeat(img.shape[0], 1, 1, 1)
        img = img.unsqueeze(0)
        img = F.conv2d(img, weight, padding=k // 2, groups=img.shape[1])
        return img.squeeze(0)

    def _apply_jpeg(self, img, quality):
        img = img.detach().to(torch.float32)
        img_np = (img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img_np).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        out = TF.to_tensor(out).to(img.device, dtype=img.dtype)
        return out

    def _shuffle_ops(self, generator):
        ops = list(DEG_OPS)
        if generator is None: random.shuffle(ops)
        else:
            for i in range(len(ops) - 1, 0, -1):
                j = int(torch.randint(0, i + 1, (1,), generator=generator).item())
                ops[i], ops[j] = ops[j], ops[i]
        return ops

    def _sample_stage_params(self, generator):
        blur_applied = bool(self._sample_uniform(0.0, 1.0, generator) < 0.9)
        blur_is_aniso = bool(self._sample_uniform(0.0, 1.0, generator) < 0.5)
        if blur_applied:
            k_size = self._sample_choice(self.blur_kernels, generator)
            if blur_is_aniso:
                sigma_x = self._sample_uniform(*self.aniso_sigma_range, generator)
                sigma_y = self._sample_uniform(*self.aniso_sigma_range, generator)
                theta = self._sample_uniform(*self.aniso_theta_range, generator)
                sigma = 0.0
            else:
                sigma = self._sample_uniform(*self.blur_sigma_range, generator)
                sigma_x = 0.0; sigma_y = 0.0; theta = 0.0
        else:
            k_size = 0; sigma = 0.0; sigma_x = 0.0; sigma_y = 0.0; theta = 0.0
        resize_scale = self._sample_uniform(*RESIZE_SCALE_RANGE, generator)
        resize_interp = self._sample_choice(RESIZE_INTERP_MODES, generator)
        resize_interp_idx = RESIZE_INTERP_MODES.index(resize_interp)
        noise_std = self._sample_uniform(*self.noise_range, generator)
        jpeg_quality = self._sample_int(*JPEG_QUALITY_RANGE, generator)
        return {
            "blur_applied": blur_applied,
            "k_size": k_size,
            "sigma": sigma, "sigma_x": sigma_x, "sigma_y": sigma_y, "theta": theta,
            "resize_scale": resize_scale,
            "resize_interp_idx": resize_interp_idx, "resize_interp": resize_interp,
            "noise_std": noise_std,
            "jpeg_quality": jpeg_quality,
        }

    def __call__(self, hr_tensor, return_meta: bool = False, meta=None, generator=None):
        img = (hr_tensor + 1.0) * 0.5
        if meta is None:
            use_two_stage = bool(self._sample_uniform(0.0, 1.0, generator) < P_TWO_STAGE)
            ops_stage1 = self._shuffle_ops(generator)
            ops_stage2 = self._shuffle_ops(generator) if use_two_stage else []
            stage1 = self._sample_stage_params(generator)
            stage2 = self._sample_stage_params(generator) if use_two_stage else None
        else:
            use_two_stage = bool(int(meta.get("use_two_stage", torch.tensor(0)).item()))
            ops_stage1 = [op for op in str(meta.get("ops_stage1", ",".join(DEG_OPS))).split(",") if op]
            ops_stage2 = [op for op in str(meta.get("ops_stage2", "")).split(",") if op] if use_two_stage else []
            stage1 = {
                "blur_applied": bool(int(meta["stage1_blur_applied"].item())),
                "k_size": int(meta["stage1_k_size"].item()),
                "sigma": float(meta["stage1_sigma"].item()),
                "sigma_x": float(meta["stage1_sigma_x"].item()),
                "sigma_y": float(meta["stage1_sigma_y"].item()),
                "theta": float(meta["stage1_theta"].item()),
                "resize_scale": float(meta["stage1_resize_scale"].item()),
                "resize_interp_idx": int(meta["stage1_resize_interp"].item()),
                "resize_interp": RESIZE_INTERP_MODES[int(meta["stage1_resize_interp"].item())],
                "noise_std": float(meta["stage1_noise_std"].item()),
                "jpeg_quality": int(meta["stage1_jpeg_quality"].item()),
                "noise": meta.get("stage1_noise", None),
            }
            stage2 = None
            if use_two_stage:
                stage2 = {
                    "blur_applied": bool(int(meta["stage2_blur_applied"].item())),
                    "k_size": int(meta["stage2_k_size"].item()),
                    "sigma": float(meta["stage2_sigma"].item()),
                    "sigma_x": float(meta["stage2_sigma_x"].item()),
                    "sigma_y": float(meta["stage2_sigma_y"].item()),
                    "theta": float(meta["stage2_theta"].item()),
                    "resize_scale": float(meta["stage2_resize_scale"].item()),
                    "resize_interp_idx": int(meta["stage2_resize_interp"].item()),
                    "resize_interp": RESIZE_INTERP_MODES[int(meta["stage2_resize_interp"].item())],
                    "noise_std": float(meta["stage2_noise_std"].item()),
                    "jpeg_quality": int(meta["stage2_jpeg_quality"].item()),
                    "noise": meta.get("stage2_noise", None),
                }

        def apply_ops(img_in, ops, params):
            out = img_in
            stage_noise = None
            for op in ops:
                if op == "blur" and params["blur_applied"]:
                    if params["sigma_x"] > 0 and params["sigma_y"] > 0:
                        out = self._apply_aniso_blur(out, params["k_size"], params["sigma_x"], params["sigma_y"], params["theta"])
                    else: out = TF.gaussian_blur(out, params["k_size"], [params["sigma"], params["sigma"]])
                elif op == "resize":
                    mid_h = max(1, int(round(self.crop_size * params["resize_scale"])))
                    mid_w = max(1, int(round(self.crop_size * params["resize_scale"])))
                    out = TF.resize(out, [mid_h, mid_w], interpolation=params["resize_interp"], antialias=True)
                elif op == "noise":
                    if params["noise_std"] > 0:
                        if meta is None:
                            if generator is None: noise = torch.randn_like(out)
                            else: noise = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator)
                        else:
                            noise = params.get("noise")
                            if noise is None: noise = torch.zeros_like(out)
                            else: noise = noise.to(out.device, dtype=out.dtype)
                        stage_noise = noise
                        out = (out + noise * params["noise_std"]).clamp(0.0, 1.0)
                    else: stage_noise = torch.zeros_like(out)
                elif op == "jpeg": out = self._apply_jpeg(out, params["jpeg_quality"])
            if stage_noise is None: stage_noise = torch.zeros_like(out)
            return out, stage_noise

        lr_small, stage1_noise = apply_ops(img, ops_stage1, stage1)
        stage2_noise = torch.zeros_like(lr_small)
        if use_two_stage: lr_small, stage2_noise = apply_ops(lr_small, ops_stage2, stage2)

        down_h = int(self.crop_size * self.downscale_factor)
        down_w = int(self.crop_size * self.downscale_factor)
        lr_small = TF.resize(lr_small, [down_h, down_w], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        lr_out = TF.resize(lr_small, [self.crop_size, self.crop_size], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        lr_out = (lr_out * 2.0 - 1.0).clamp(-1.0, 1.0)

        # FIXED: Strictly return 2 values
        if return_meta:
            meta_out = {
                "stage1_blur_applied": torch.tensor(int(stage1["blur_applied"]), dtype=torch.int64),
                "stage1_k_size": torch.tensor(int(stage1["k_size"]), dtype=torch.int64),
                "stage1_sigma": torch.tensor(float(stage1["sigma"]), dtype=torch.float32),
                "stage1_sigma_x": torch.tensor(float(stage1["sigma_x"]), dtype=torch.float32),
                "stage1_sigma_y": torch.tensor(float(stage1["sigma_y"]), dtype=torch.float32),
                "stage1_theta": torch.tensor(float(stage1["theta"]), dtype=torch.float32),
                "stage1_noise_std": torch.tensor(float(stage1["noise_std"]), dtype=torch.float32),
                "stage1_noise": stage1_noise.detach().cpu().float(),
                "stage1_resize_scale": torch.tensor(float(stage1["resize_scale"]), dtype=torch.float32),
                "stage1_resize_interp": torch.tensor(int(stage1["resize_interp_idx"]), dtype=torch.int64),
                "stage1_jpeg_quality": torch.tensor(int(stage1["jpeg_quality"]), dtype=torch.int64),
                "stage2_blur_applied": torch.tensor(int(stage2["blur_applied"]), dtype=torch.int64) if stage2 else torch.tensor(0, dtype=torch.int64),
                "stage2_k_size": torch.tensor(int(stage2["k_size"]), dtype=torch.int64) if stage2 else torch.tensor(0, dtype=torch.int64),
                "stage2_sigma": torch.tensor(float(stage2["sigma"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_sigma_x": torch.tensor(float(stage2["sigma_x"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_sigma_y": torch.tensor(float(stage2["sigma_y"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_theta": torch.tensor(float(stage2["theta"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_noise_std": torch.tensor(float(stage2["noise_std"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_noise": stage2_noise.detach().cpu().float(),
                "stage2_resize_scale": torch.tensor(float(stage2["resize_scale"]), dtype=torch.float32) if stage2 else torch.tensor(0.0, dtype=torch.float32),
                "stage2_resize_interp": torch.tensor(int(stage2["resize_interp_idx"]), dtype=torch.int64) if stage2 else torch.tensor(0, dtype=torch.int64),
                "stage2_jpeg_quality": torch.tensor(int(stage2["jpeg_quality"]), dtype=torch.int64) if stage2 else torch.tensor(0, dtype=torch.int64),
                "use_two_stage": torch.tensor(int(use_two_stage), dtype=torch.int64),
                "ops_stage1": ",".join(ops_stage1),
                "ops_stage2": ",".join(ops_stage2),
                "down_h": torch.tensor(int(down_h), dtype=torch.int64),
                "down_w": torch.tensor(int(down_w), dtype=torch.int64),
            }
            return lr_out, meta_out
        return lr_out

# ================= 6. Datasets (Correctly Placed BEFORE Main) =================
def _scan_images(root):
    root_p = Path(root)
    if not root_p.exists():
        return []
    out = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        out.extend(root_p.rglob(ext))
    return sorted(out)

def _normalize_pair_stem(stem: str) -> str:
    key = stem.lower()
    for tok in ("_hr", "_lr4", "_lr", "x4"):
        key = key.replace(tok, "")
    return key

def build_paired_file_list(df2k_hr_root: str, df2k_lr_root: str, realsr_roots):
    pairs = []
    df2k_hr = _scan_images(df2k_hr_root)
    if df2k_hr and os.path.exists(df2k_lr_root):
        lr_map = {}
        for p in _scan_images(df2k_lr_root):
            lr_map.setdefault(_normalize_pair_stem(p.stem), []).append(str(p))
        for hr_path in df2k_hr:
            key = _normalize_pair_stem(hr_path.stem)
            lr_cands = lr_map.get(key, [])
            if not lr_cands:
                continue
            best_lr = sorted(lr_cands, key=lambda x: ("lr4" not in x.lower() and "x4" not in x.lower(), len(x)))[0]
            pairs.append((best_lr, str(hr_path)))

    for root in realsr_roots:
        for hr_path in _scan_images(root):
            stem = hr_path.stem
            if "_hr" not in stem.lower():
                continue
            lr_name = stem.replace("_HR", "_LR4").replace("_hr", "_lr4") + hr_path.suffix
            lr_path = hr_path.with_name(lr_name)
            if lr_path.exists():
                pairs.append((str(lr_path), str(hr_path)))
    return sorted(pairs)

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

class DF2K_Online_Dataset(Dataset):
    def __init__(self, crop_size=512, is_train=True, scale=4):
        self.pairs = build_paired_file_list(TRAIN_DF2K_HR_DIR, TRAIN_DF2K_LR_DIR, TRAIN_REALSR_DIRS)
        if len(self.pairs) == 0:
            raise RuntimeError("No LR/HR pairs found from DF2K/RealSR training roots.")
        self.crop_size = crop_size
        self.lr_patch = crop_size // scale
        self.scale = scale
        self.is_train = is_train
        self.norm = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        self.to_tensor = transforms.ToTensor()
        self.epoch = 0

    def set_epoch(self, epoch: int): self.epoch = int(epoch)
    def _make_generator(self, idx: int):
        gen = torch.Generator(); seed = SEED + (self.epoch * 1_000_000) + int(idx); gen.manual_seed(seed)
        return gen
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        try:
            lr_path, hr_path = self.pairs[idx]
            lr_pil = Image.open(lr_path).convert("RGB")
            hr_pil = Image.open(hr_path).convert("RGB")
        except: return self.__getitem__((idx + 1) % len(self))
        gen = None
        lr_aligned, hr_aligned = center_crop_aligned_pair(lr_pil, hr_pil, scale=self.scale)

        if self.is_train:
            gen = self._make_generator(idx)
            w_l, h_l = lr_aligned.size
            if h_l < self.lr_patch or w_l < self.lr_patch:
                lr_aligned = TF.resize(lr_aligned, (self.lr_patch, self.lr_patch), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
                hr_aligned = TF.resize(hr_aligned, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
                x = 0
                y = 0
            else:
                max_x = w_l - self.lr_patch
                max_y = h_l - self.lr_patch
                x = int(torch.randint(0, max_x + 1, (1,), generator=gen).item())
                y = int(torch.randint(0, max_y + 1, (1,), generator=gen).item())
            lr_crop = TF.crop(lr_aligned, y, x, self.lr_patch, self.lr_patch)
            hr_crop = TF.crop(hr_aligned, y * self.scale, x * self.scale, self.crop_size, self.crop_size)
            if torch.rand(1, generator=gen).item() < 0.5:
                lr_crop = TF.hflip(lr_crop)
                hr_crop = TF.hflip(hr_crop)
            k = int(torch.randint(0, 4, (1,), generator=gen).item())
            if k:
                angle = 90 * k
                lr_crop = TF.rotate(lr_crop, angle)
                hr_crop = TF.rotate(hr_crop, angle)
        else:
            lr_crop = TF.center_crop(lr_aligned, (self.lr_patch, self.lr_patch))
            hr_crop = TF.center_crop(hr_aligned, (self.crop_size, self.crop_size))

        lr_up = TF.resize(lr_crop, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        hr_tensor = self.norm(self.to_tensor(hr_crop))
        lr_tensor = self.norm(self.to_tensor(lr_up))
        return {"hr": hr_tensor, "lr": lr_tensor, "path": hr_path}

class DF2K_Val_Fixed_Dataset(Dataset):
    def __init__(self, hr_root, lr_root=None, crop_size=512):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_root, "*.png")))
        self.lr_root = lr_root; self.crop_size = crop_size
        self.norm = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3); self.to_tensor = transforms.ToTensor()
    def __len__(self): return len(self.hr_paths)
    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]; hr_pil = Image.open(hr_path).convert("RGB")
        lr_crop = None
        if self.lr_root:
            base = os.path.basename(hr_path); lr_name = base.replace(".png", "x4.png")
            lr_p = os.path.join(self.lr_root, lr_name)
            if os.path.exists(lr_p):
                lr_pil = Image.open(lr_p).convert("RGB")
                lr_aligned, hr_aligned = center_crop_aligned_pair(lr_pil, hr_pil, scale=4)
                lr_crop = TF.center_crop(lr_aligned, (self.crop_size//4, self.crop_size//4))
                hr_crop = TF.center_crop(hr_aligned, (self.crop_size, self.crop_size))
                lr_crop = TF.resize(lr_crop, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        if lr_crop is None:
            hr_crop = TF.center_crop(hr_pil, (self.crop_size, self.crop_size))
            w, h = hr_crop.size; lr_small = hr_crop.resize((w//4, h//4), Image.BICUBIC)
            lr_crop = lr_small.resize((w, h), Image.BICUBIC)
        hr_tensor = self.norm(self.to_tensor(hr_crop)); lr_tensor = self.norm(self.to_tensor(lr_crop))
        return {"hr": hr_tensor, "lr": lr_tensor, "path": hr_path}

class DF2K_Val_Degraded_Dataset(Dataset):
    def __init__(self, hr_root, crop_size=512, seed=3407, deg_mode="highorder"):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_root, "*.png")))
        self.crop_size = crop_size; self.norm = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        self.to_tensor = transforms.ToTensor(); self.pipeline = DegradationPipeline(crop_size)
        self.seed = int(seed); self.deg_mode = deg_mode
    def __len__(self): return len(self.hr_paths)
    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]; hr_pil = Image.open(hr_path).convert("RGB")
        hr_crop = TF.center_crop(hr_pil, (self.crop_size, self.crop_size))
        hr_tensor = self.norm(self.to_tensor(hr_crop))
        if self.deg_mode == "bicubic":
            lr_small = TF.resize((hr_tensor + 1.0) * 0.5, (self.crop_size // 4, self.crop_size // 4), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            lr_tensor = TF.resize(lr_small, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            lr_tensor = (lr_tensor * 2.0 - 1.0).clamp(-1.0, 1.0)
        else:
            gen = torch.Generator(); gen.manual_seed(self.seed + idx)
            # Ensure only 2 values unpacked
            lr_tensor, _ = self.pipeline(hr_tensor, return_meta=True, generator=gen) # Ignore meta here
        return {"hr": hr_tensor, "lr": lr_tensor, "path": hr_path}

class ValPackDataset(Dataset):
    def __init__(self, pack_dir: str, lr_dir_name: str = "lq512", crop_size: int = 512):
        self.pack_dir = Path(pack_dir); self.hr_dir = self.pack_dir / "gt512"; self.lr_dir = self.pack_dir / lr_dir_name
        if not self.hr_dir.is_dir(): raise FileNotFoundError(f"gt512 dir not found: {self.hr_dir}")
        if not self.lr_dir.is_dir(): raise FileNotFoundError(f"LR dir not found: {self.lr_dir}")
        self.crop_size = crop_size; self.norm = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        self.to_tensor = transforms.ToTensor(); self.hr_paths = sorted(list(self.hr_dir.glob("*.png")))
    def __len__(self): return len(self.hr_paths)
    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]; name = hr_path.stem; lr_path = self.lr_dir / f"{name}.png"
        if not lr_path.is_file(): raise FileNotFoundError(f"LR image missing: {lr_path}")
        hr_pil = Image.open(hr_path).convert("RGB"); lr_pil = Image.open(lr_path).convert("RGB")
        hr_crop = TF.center_crop(hr_pil, (self.crop_size, self.crop_size)); lr_crop = TF.center_crop(lr_pil, (self.crop_size, self.crop_size))
        if lr_crop.size != (self.crop_size, self.crop_size):
            lr_crop = TF.resize(lr_crop, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        hr_tensor = self.norm(self.to_tensor(hr_crop)); lr_tensor = self.norm(self.to_tensor(lr_crop))
        return {"hr": hr_tensor, "lr": lr_tensor, "path": str(hr_path)}

# ================= 7. LoRA =================
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base; self.scaling = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False, dtype=torch.float32)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, dtype=torch.float32)
        self.lora_A.to(base.weight.device); self.lora_B.to(base.weight.device)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5)); nn.init.zeros_(self.lora_B.weight)
        self.base.weight.requires_grad = False
        if self.base.bias is not None: self.base.bias.requires_grad = False
    def forward(self, x):
        out = self.base(x)
        delta = self.lora_B(self.lora_A(x.float())) * self.scaling
        return out + delta.to(out.dtype)

def apply_lora(model, rank=64, alpha=64):
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(key in name for key in ("qkv", "proj", "to_q", "to_k", "to_v", "q_linear", "kv_linear")):
             parent = model.get_submodule(name.rsplit('.', 1)[0]); child = name.rsplit('.', 1)[1]
             setattr(parent, child, LoRALinear(module, rank, alpha)); cnt += 1
    print(f"✅ LoRA applied to {cnt} layers.")


def configure_pixart_trainable_params(pixart: nn.Module, inject_gate_keys, train_x_embedder: bool = False):
    # Freeze everything first, then unfreeze only the intended trainable subset.
    for p in pixart.parameters():
        p.requires_grad_(False)

    trainable_names = []
    for n, p in pixart.named_parameters():
        is_lora = ("lora_A" in n) or ("lora_B" in n)
        is_inject = any(k in n for k in inject_gate_keys)
        is_x_embed = train_x_embedder and ("x_embedder" in n)
        if is_lora or is_inject or is_x_embed:
            p.requires_grad_(True)
            trainable_names.append(n)

    if len(trainable_names) == 0:
        raise RuntimeError("No PixArt trainable parameters selected. Check freezing whitelist.")
    print(f"✅ PixArt trainable configured: {len(trainable_names)} tensors (x_embedder={train_x_embedder})")

# ================= 8. Checkpointing =================
def should_keep_ckpt(psnr_v, lpips_v):
    if not math.isfinite(psnr_v): return (999, float("inf"))
    if psnr_v >= PSNR_SWITCH and math.isfinite(lpips_v): return (0, lpips_v)
    return (1, -psnr_v)

def atomic_torch_save(state, path):
    tmp = path + ".tmp"
    try:
        torch.save(state, tmp); os.replace(tmp, path); return True, "zip"
    except Exception as e_zip:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass
        try:
            torch.save(state, tmp, _use_new_zipfile_serialization=False); os.replace(tmp, path); return True, f"legacy ({e_zip})"
        except Exception as e_old:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except Exception: pass
            return False, f"zip_error={e_zip}; legacy_error={e_old}"

def save_smart(epoch, global_step, pixart, adapter, optimizer, best_records, metrics, dl_gen):
    global BASE_PIXART_SHA256
    psnr_v, ssim_v, lpips_v = metrics; priority, score = should_keep_ckpt(psnr_v, lpips_v)
    current_record = {"path": None, "epoch": epoch, "priority": priority, "score": score, "psnr": psnr_v, "lpips": lpips_v}
    save_as_best = False
    if len(best_records) < KEEP_TOPK: save_as_best = True
    else:
        worst_record = best_records[-1]
        if (priority < worst_record['priority']) or (priority == worst_record['priority'] and score < worst_record['score']): save_as_best = True

    ckpt_name = None
    if save_as_best:
        ckpt_name = f"epoch{epoch+1:03d}_psnr{psnr_v:.2f}_lp{lpips_v:.4f}.pth"
        ckpt_path = os.path.join(CKPT_DIR, ckpt_name); current_record['path'] = ckpt_path

    if BASE_PIXART_SHA256 is None and os.path.exists(PIXART_PATH):
        try: BASE_PIXART_SHA256 = file_sha256(PIXART_PATH)
        except Exception as e: print(f"⚠️ Base PixArt hash failed (non-fatal): {e}"); BASE_PIXART_SHA256 = None
    pixart_sd = collect_trainable_state_dict(pixart); required_frags = get_required_v7_key_fragments_for_model(pixart)
    v7_key_counts = validate_v7_trainable_state_keys(pixart_sd, required_frags)
    lora_key_count = sum(("lora_A" in k or "lora_B" in k) for k in pixart_sd.keys())
    if lora_key_count == 0:
        raise RuntimeError("No LoRA keys found in pixart_trainable. Check apply_lora() and requires_grad flags.")
    print(f"✅ LoRA save check: {lora_key_count} tensors")
    print("✅ v7 save check:", ", ".join([f"{k}={v}" for k, v in v7_key_counts.items()]))

    next_best_records = list(best_records)
    if save_as_best:
        next_best_records.append(current_record)
        next_best_records.sort(key=lambda x: (x['priority'], x['score']))
        next_best_records = next_best_records[:KEEP_TOPK]

    state = {
        "epoch": epoch, "step": global_step, "adapter": {k: v.detach().float().cpu() for k, v in adapter.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "rng_state": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "numpy": np.random.get_state(), "python": random.getstate()},
        "dl_gen_state": dl_gen.get_state(), "pixart_trainable": pixart_sd, "best_records": next_best_records, "config_snapshot": get_config_snapshot(), "base_pixart_sha256": BASE_PIXART_SHA256, "env_info": {"torch": torch.__version__, "numpy": np.__version__},
    }
    last_path = LAST_CKPT_PATH; ok_last, msg_last = atomic_torch_save(state, last_path)
    if ok_last: print(f"💾 Saved last checkpoint to {last_path} [{msg_last}]")
    else: print(f"❌ Failed to save last.pth: {msg_last}")

    best_saved = False
    if save_as_best and current_record["path"]:
        try:
            if ok_last and os.path.exists(last_path):
                shutil.copy2(last_path, current_record["path"]); print(f"🏆 New Best Model! Copied from last.pth to {ckpt_name}")
                best_saved = True
            else:
                ok_best, msg_best = atomic_torch_save(state, current_record["path"])
                if ok_best:
                    print(f"🏆 New Best Model! Saved to {ckpt_name} [{msg_best}]")
                    best_saved = True
                else:
                    print(f"❌ Failed to save best checkpoint: {msg_best}")
        except Exception as e:
            print(f"❌ Failed to save best checkpoint: {e}")

    if best_saved:
        old_paths = {rec.get('path') for rec in best_records if rec.get('path')}
        keep_paths = {rec.get('path') for rec in next_best_records if rec.get('path')}
        for stale in sorted(old_paths - keep_paths):
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                    print(f"🗑️ Removed old best: {os.path.basename(stale)}")
                except Exception:
                    pass
        return next_best_records

    return best_records

def _strict_load_pixart_trainable_subset(pixart: nn.Module, saved_trainable: dict, context: str):
    expected = {k for k, p in pixart.named_parameters() if p.requires_grad}
    saved = set(saved_trainable.keys())
    missing = sorted(expected - saved)
    unexpected = sorted(saved - expected)
    if missing or unexpected:
        msg = [f"{context}: pixart_trainable key mismatch."]
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
        raise RuntimeError(f"{context}: pixart_trainable shape mismatch count={len(bad_shapes)}. {preview}")

    for k in sorted(expected):
        curr[k] = saved_trainable[k].to(dtype=curr[k].dtype)
    pixart.load_state_dict(curr, strict=False)

def resume(pixart, adapter, optimizer, dl_gen):
    if not os.path.exists(LAST_CKPT_PATH): return 0, 0, []
    print(f"📥 Resuming from {LAST_CKPT_PATH}...")
    ckpt = torch.load(LAST_CKPT_PATH, map_location="cpu")
    saved_trainable = ckpt.get("pixart_trainable", {})
    required_frags = get_required_v7_key_fragments_for_model(pixart)
    missing_required = [frag for frag in required_frags if not any(frag in k for k in saved_trainable.keys())]
    if missing_required: raise RuntimeError("Checkpoint is missing required v7 trainable keys: " + ", ".join(missing_required))

    adapter_sd = ckpt.get("adapter", {})
    try:
        adapter.load_state_dict(adapter_sd, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"Adapter strict load failed during resume: {e}") from e

    _strict_load_pixart_trainable_subset(pixart, saved_trainable, context="resume")
    optimizer.load_state_dict(ckpt["optimizer"])
    rs = ckpt.get("rng_state", None)
    if rs is not None:
        try:
            if rs.get("torch") is not None: torch.set_rng_state(rs["torch"])
            if torch.cuda.is_available() and rs.get("cuda") is not None: torch.cuda.set_rng_state_all(rs["cuda"])
            if rs.get("numpy") is not None: np.random.set_state(rs["numpy"])
            if rs.get("python") is not None: random.setstate(rs["python"])
        except Exception as e: print(f"⚠️ RNG restore failed (non-fatal): {e}")
    dl_state = ckpt.get("dl_gen_state", None)
    if dl_state is not None:
        try: dl_gen.set_state(dl_state)
        except Exception as e: print(f"⚠️ DataLoader generator restore failed (non-fatal): {e}")
    return ckpt["epoch"]+1, ckpt["step"], ckpt.get("best_records", [])

# ================= 9. Validation =================
@torch.no_grad()
def validate(epoch, pixart, adapter, vae, val_loader, y_embed, data_info, lpips_fn_val_cpu):
    print(f"🔎 Validating Epoch {epoch+1}...")
    pixart.eval(); adapter.eval()
    results = {}
    val_gen = torch.Generator(device=DEVICE); val_gen.manual_seed(SEED)
    
    # [V8 Change] Validation uses V-Prediction scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear",
        clip_sample=False, prediction_type="v_prediction", set_alpha_to_one=False,
    )

    # Validation LPIPS must stay on CPU to reduce GPU memory peak.
    try:
        _lpips_dev = next(lpips_fn_val_cpu.parameters()).device
        if _lpips_dev.type != "cpu":
            raise RuntimeError(f"lpips_fn_val_cpu must be on CPU, got {_lpips_dev}")
    except StopIteration:
        pass
    
    steps_list = [FAST_VAL_STEPS] if FAST_DEV_RUN else VAL_STEPS_LIST
    for steps in steps_list:
        scheduler.set_timesteps(steps, device=DEVICE)
        psnrs, ssims, lpipss = [], [], []; vis_done = False
        for batch in tqdm(val_loader, desc=f"Val@{steps}"):
            hr = batch["hr"].to(DEVICE); lr = batch["lr"].to(DEVICE)
            z_hr = vae.encode(hr).latent_dist.mean * vae.config.scaling_factor
            z_lr = vae.encode(lr).latent_dist.mean * vae.config.scaling_factor
            
            if USE_LQ_INIT: latents, run_timesteps = get_lq_init_latents(z_lr.to(COMPUTE_DTYPE), scheduler, steps, val_gen, LQ_INIT_STRENGTH, COMPUTE_DTYPE)
            else: latents = randn_like_with_generator(z_hr, val_gen); run_timesteps = scheduler.timesteps
            
            cond = adapter(z_lr.float())
            aug_level = torch.zeros((latents.shape[0],), device=DEVICE, dtype=COMPUTE_DTYPE)
            
            for t in run_timesteps:
                t_b = torch.tensor([t], device=DEVICE).expand(latents.shape[0])
                with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                    if FORCE_DROP_TEXT: drop_uncond = torch.ones(latents.shape[0], device=DEVICE); drop_cond = torch.ones(latents.shape[0], device=DEVICE)
                    else: drop_uncond = torch.ones(latents.shape[0], device=DEVICE); drop_cond = torch.zeros(latents.shape[0], device=DEVICE)
                    lr_ref = z_lr.to(COMPUTE_DTYPE)
                    model_in = torch.cat([latents.to(COMPUTE_DTYPE), lr_ref], dim=1)
                    cond_zero = mask_adapter_cond(cond, torch.zeros((latents.shape[0],), device=DEVICE))
                    if CFG_SCALE == 1.0:
                        out = pixart(x=model_in, timestep=t_b, y=y_embed, aug_level=aug_level, mask=None, data_info=data_info, adapter_cond=cond, injection_mode="hybrid", force_drop_ids=drop_cond)
                        if out.shape[1] == 8:
                            out, _ = out.chunk(2, dim=1)
                    else:
                        out_uncond = pixart(x=model_in, timestep=t_b, y=y_embed, aug_level=aug_level, mask=None, data_info=data_info, adapter_cond=cond_zero, injection_mode="hybrid", force_drop_ids=drop_uncond)
                        out_cond = pixart(x=model_in, timestep=t_b, y=y_embed, aug_level=aug_level, mask=None, data_info=data_info, adapter_cond=cond, injection_mode="hybrid", force_drop_ids=drop_cond)
                        if out_uncond.shape[1] == 8: out_uncond, _ = out_uncond.chunk(2, dim=1)
                        if out_cond.shape[1] == 8: out_cond, _ = out_cond.chunk(2, dim=1)
                        out = out_uncond + CFG_SCALE * (out_cond - out_uncond)
                latents = scheduler.step(out.float(), t, latents.float()).prev_sample
            pred = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)
            p01 = (pred + 1) / 2; h01 = (hr + 1) / 2
            py = rgb01_to_y01(p01)[..., 4:-4, 4:-4]; hy = rgb01_to_y01(h01)[..., 4:-4, 4:-4]
            if "psnr" in globals():
                psnrs.append(psnr(py, hy, data_range=1.0).item()); ssims.append(ssim(py, hy, data_range=1.0).item())

            pred_cpu = pred.detach().to("cpu", dtype=torch.float32)
            hr_cpu = hr.detach().to("cpu", dtype=torch.float32)
            lpipss.append(lpips_fn_val_cpu(pred_cpu, hr_cpu).mean().item())
            del pred_cpu, hr_cpu

            if not vis_done:
                save_path = os.path.join(VIS_DIR, f"epoch{epoch+1:03d}_steps{steps}.png")
                lr_np = (lr[0].cpu().float().numpy().transpose(1,2,0) + 1) / 2
                hr_np = (hr[0].cpu().float().numpy().transpose(1,2,0) + 1) / 2
                pr_np = (pred[0].cpu().float().numpy().transpose(1,2,0) + 1) / 2
                plt.figure(figsize=(12,4))
                plt.subplot(1,3,1); plt.imshow(np.clip(lr_np, 0, 1)); plt.title("Input LR"); plt.axis("off")
                plt.subplot(1,3,2); plt.imshow(np.clip(hr_np, 0, 1)); plt.title("GT"); plt.axis("off")
                plt.subplot(1,3,3); plt.imshow(np.clip(pr_np, 0, 1)); plt.title(f"Pred @{steps}"); plt.axis("off")
                plt.savefig(save_path, bbox_inches="tight"); plt.close(); vis_done = True
            if FAST_DEV_RUN and len(psnrs) >= FAST_VAL_BATCHES: break
        res = (float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(lpipss)))
        results[int(steps)] = res
        print(f"[VAL@{steps}] Ep{epoch+1}: PSNR={res[0]:.2f} | SSIM={res[1]:.4f} | LPIPS={res[2]:.4f}")
    pixart.train(); adapter.train()
    return results

# ================= 10. Main =================
def main():
    seed_everything(SEED); dl_gen = torch.Generator(); dl_gen.manual_seed(SEED)
    required_paths = [PIXART_PATH, VAE_PATH, NULL_T5_EMBED_PATH]
    for pth in required_paths:
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Required pretrained path missing: {pth}")

    train_ds = DF2K_Online_Dataset(crop_size=512, is_train=True, scale=4)
    if VAL_MODE == "valpack":
        val_ds = ValPackDataset(VAL_PACK_DIR, lr_dir_name=VAL_PACK_LR_DIR_NAME, crop_size=512)
        print(f"[VAL] mode=valpack path={VAL_PACK_DIR}/{VAL_PACK_LR_DIR_NAME}")
    elif VAL_MODE == "train_like":
        val_ds = DF2K_Val_Degraded_Dataset(VAL_HR_DIR, crop_size=512, seed=SEED, deg_mode=TRAIN_DEG_MODE)
        print(f"[VAL] mode=train_like deg_mode={TRAIN_DEG_MODE}")
    elif VAL_MODE == "lr_dir" and VAL_LR_DIR is not None:
        val_ds = DF2K_Val_Fixed_Dataset(VAL_HR_DIR, lr_root=VAL_LR_DIR, crop_size=512)
        print(f"[VAL] mode=lr_dir lr_root={VAL_LR_DIR}")
    else:
        val_ds = DF2K_Val_Fixed_Dataset(VAL_HR_DIR, lr_root=None, crop_size=512)
        print("[VAL] mode=fallback_bicubic_from_hr (no paired VAL_LR_DIR found)")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, worker_init_fn=seed_worker, generator=dl_gen)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    pixart = PixArtSigmaSR_XL_2(
        input_size=64, in_channels=8, sparse_inject_ratio=SPARSE_INJECT_RATIO,
        injection_cutoff_layer=INJECTION_CUTOFF_LAYER, injection_strategy=INJECTION_STRATEGY,
    ).to(DEVICE)
    base = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in base: base = base["state_dict"]
    if "pos_embed" in base: del base["pos_embed"]
    if "x_embedder.proj.weight" in base and base["x_embedder.proj.weight"].shape[1] == 4:
        w4 = base["x_embedder.proj.weight"]
        w8 = torch.zeros((w4.shape[0], 8, w4.shape[2], w4.shape[3]), dtype=w4.dtype)
        w8[:, :4] = w4; w8[:, 4:] = w4 * 0.5; base["x_embedder.proj.weight"] = w8
    if hasattr(pixart, "load_pretrained_weights_with_zero_init"):
        pixart.load_pretrained_weights_with_zero_init(base)
    else:
        missing, unexpected = pixart.load_state_dict(base, strict=False)
        print(f"[Load] missing={len(missing)} unexpected={len(unexpected)}")
    apply_lora(pixart, LORA_RANK, LORA_ALPHA)
    inject_gate_keys = (
        "adapter_alpha_mlp", "input_adaln", "input_res_proj",
        "style_fusion_mlp", "input_adapter_ln", "aug_embedder", "injection_scales"
    )
    configure_pixart_trainable_params(pixart, inject_gate_keys=inject_gate_keys, train_x_embedder=TRAIN_PIXART_X_EMBEDDER)
    pixart.train()

    adapter = build_adapter_v7(in_channels=4, hidden_size=1152, injection_layers_map=getattr(pixart, "injection_layer_to_level", getattr(pixart, "injection_layers", None))).to(DEVICE).float().train()
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE).float().eval()
    vae.enable_slicing()
    if VAE_TILING and hasattr(vae, "enable_tiling"): vae.enable_tiling()

    lpips_fn_val_cpu = lpips.LPIPS(net='vgg').to("cpu").eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in lpips_fn_val_cpu.parameters(): p.requires_grad_(False)
    print("✅ Validation LPIPS is on CPU.")
    lpips_fn_train = None  # lazy-init on GPU when/if training LPIPS becomes active

    if not os.path.exists(NULL_T5_EMBED_PATH):
        raise FileNotFoundError(f"Null T5 embed not found: {NULL_T5_EMBED_PATH}")
    null_pack = torch.load(NULL_T5_EMBED_PATH, map_location="cpu")
    if "y" not in null_pack:
        raise KeyError(f"Invalid null T5 embed file (missing key 'y'): {NULL_T5_EMBED_PATH}")
    y = null_pack["y"].to(DEVICE)
    if y.ndim != 4:
        raise RuntimeError(f"Invalid y shape from offline null T5 embed: {tuple(y.shape)} (expected [1,1,L,C])")
    print(f"✅ Loaded offline null T5 embedding: y.shape={tuple(y.shape)}")

    d_info = {"img_hw": torch.tensor([[512.,512.]]).to(DEVICE), "aspect_ratio": torch.tensor([1.]).to(DEVICE)}

    # [FIXED HERE] Process-Corrected Optimizer Grouping (Mutually Exclusive)
    adapter_params = list(adapter.parameters())
    
    # 1. Select params for trainable PixArt subset (LoRA + injection modules)
    embedder_params = []
    inject_gate_params = []
    lora_params = []
    other_pixart_params = []

    for n, p in pixart.named_parameters():
        if not p.requires_grad:
            continue
        if 'x_embedder' in n:
            embedder_params.append(p)
        elif any(k in n for k in inject_gate_keys):
            inject_gate_params.append(p)
        elif ('lora_A' in n) or ('lora_B' in n):
            lora_params.append(p)
        else:
            other_pixart_params.append(p)

    # 3. Create Optimizer Groups
    optim_groups = [
        {"params": adapter_params, "lr": 1e-4},
        {"params": inject_gate_params, "lr": 1e-4, "weight_decay": 0.0},
        {"params": lora_params, "lr": 1e-5},
    ]
    if TRAIN_PIXART_X_EMBEDDER and len(embedder_params) > 0:
        optim_groups.append({"params": embedder_params, "lr": 1e-4})
    optimizer = torch.optim.AdamW(optim_groups)

    # 2. Clipper needs flat tensor list
    params_to_clip = adapter_params + inject_gate_params + lora_params + embedder_params

    # Sanity checks: ensure pixart trainable params are fully and uniquely covered.
    pixart_trainable = [p for p in pixart.parameters() if p.requires_grad]
    grouped = embedder_params + inject_gate_params + lora_params + other_pixart_params
    if len({id(p) for p in grouped}) != len(grouped):
        raise RuntimeError("Optimizer grouping has duplicate PixArt params across groups.")
    if {id(p) for p in grouped} != {id(p) for p in pixart_trainable}:
        raise RuntimeError("Optimizer grouping does not exactly cover PixArt trainable params.")
    if len(inject_gate_params) == 0 or len(lora_params) == 0:
        print(f"⚠️ Optimizer group warning: inject_gate={len(inject_gate_params)}, lora={len(lora_params)}, x_embedder={len(embedder_params)}")
    if len(other_pixart_params) != 0:
        raise RuntimeError(f"Unexpected trainable backbone params outside whitelist: {len(other_pixart_params)}")

    # [V8 Change] Switch to V-Prediction logic manually in loop (since IDDPM is epsilon based)
    # We will manually calculate v_target and loss.
    # Note: IDDPM class is kept for schedule utils, but we bypass its loss function.
    diffusion = IDDPM(str(1000))
    ep_start, step, best = resume(pixart, adapter, optimizer, dl_gen)

    print("🚀 DiT-SR V8 Training Started (V-Pred, Aug, Copy-Init).")
    max_steps = MAX_TRAIN_STEPS if MAX_TRAIN_STEPS > 0 else (FAST_TRAIN_STEPS if FAST_DEV_RUN else None)

    for epoch in range(ep_start, 1000):
        if max_steps is not None and step >= max_steps: break
        train_ds.set_epoch(epoch)
        pbar = tqdm(train_loader, dynamic_ncols=True, desc=f"Ep{epoch+1}")
        accum_micro_steps = 0
        reached_max_steps = False
        for i, batch in enumerate(pbar):
            if max_steps is not None and step >= max_steps:
                reached_max_steps = True
                break
            hr = batch['hr'].to(DEVICE); lr = batch['lr'].to(DEVICE)
            with torch.no_grad():
                zh = vae.encode(hr).latent_dist.mean * vae.config.scaling_factor
                zl = vae.encode(lr).latent_dist.mean * vae.config.scaling_factor

            # [V8 Logic] V-Prediction Training
            t = torch.randint(0, 1000, (zh.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(zh)
            zt = diffusion.q_sample(zh, t, noise)
            
            # [V8 Logic] Conditioning Augmentation
            # 1. Sample noise level for LR
            aug_noise_level = torch.rand(zh.shape[0], device=DEVICE) * (COND_AUG_NOISE_RANGE[1] - COND_AUG_NOISE_RANGE[0]) + COND_AUG_NOISE_RANGE[0]
            # 2. Add noise to LR
            zlr_aug = zl.float() + torch.randn_like(zl) * aug_noise_level[:, None, None, None]
            # 3. Augmentation Level for embedding (mapped to 0-1000 for embedding)
            aug_level_emb = (aug_noise_level * 1000.0).float()

            cond = adapter(zlr_aug.float()) # Adapter sees augmented LR
            cond_in = cond
            if USE_ADAPTER_CFDROPOUT and COND_DROP_PROB > 0:
                keep = (torch.rand((zt.shape[0],), device=DEVICE) >= COND_DROP_PROB).float()
                cond_in = mask_adapter_cond(cond, keep)

            with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                drop_uncond = torch.ones(zt.shape[0], device=DEVICE)
                # Pass zlr_aug to concat as well (Consistency!)
                kwargs = dict(x=torch.cat([zt, zlr_aug.to(zt.dtype)], dim=1), timestep=t, y=y, aug_level=aug_level_emb, data_info=d_info, adapter_cond=cond_in, injection_mode="hybrid")
                kwargs["force_drop_ids"] = drop_uncond
                
                out = pixart(**kwargs)
                if out.shape[1] == 8: out, _ = out.chunk(2, dim=1)
                model_pred = out.float()

                # [V8 Logic] Calculate V-Target
                # v = alpha * epsilon - sigma * x0
                alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, zh.shape)
                sigma_t = _extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, t, zh.shape)
                target_v = alpha_t * noise - sigma_t * zh.float()
                
                # [V8 Logic] Min-SNR-Gamma Weighting
                # SNR = alpha^2 / sigma^2
                snr = (alpha_t**2) / (sigma_t**2)
                # Min-SNR-Gamma (gamma=5 is standard for V-pred)
                gamma = 5.0
                min_snr_gamma = torch.min(snr, torch.tensor(gamma, device=DEVICE))
                
                # Loss = MSE(pred_v, target_v) * Min-SNR / SNR (simplified weighting often just min(snr, gamma) / snr_something, but standard implementation is simpler:
                # For v-prediction, standard MSE is weighted by SNR/(SNR+1) effectively.
                # Here we use simplified MSE on V directly, which is robust.
                # Optional: Add Min-SNR weighting:
                loss_weights = min_snr_gamma / snr
                loss_v = (F.mse_loss(model_pred, target_v, reduction='none').mean(dim=[1,2,3]) * loss_weights.view(-1)).mean()

                # Reconstruct x0 for other losses (x0 = alpha * zt - sigma * v)
                z0 = alpha_t * zt.float() - sigma_t * model_pred

                loss_latent_l1 = F.l1_loss(z0, zh.float())

                w = get_loss_weights(step)
                
                # Calculate pixel-space losses
                loss_edge = torch.tensor(0.0, device=DEVICE)
                loss_flat_hf = torch.tensor(0.0, device=DEVICE)
                loss_lpips = torch.tensor(0.0, device=DEVICE)

                need_pixel_loss = (w['lpips'] > 0) or (w['edge_grad'] > 0) or (w['flat_hf'] > 0)
                allow_by_t = int(t[0].item()) <= PIXEL_LOSS_T_MAX
                allow_by_step = step >= PIXEL_LOSS_START_STEP
                calc_pixel_loss = need_pixel_loss and allow_by_t and allow_by_step

                if calc_pixel_loss:
                    top = torch.randint(0, 25, (1,), device=DEVICE).item() 
                    left = torch.randint(0, 25, (1,), device=DEVICE).item()
                    z0_crop = z0[..., top:top+40, left:left+40]
                    img_p_raw = vae.decode(z0_crop/vae.config.scaling_factor).sample.clamp(-1,1)
                    img_p_valid = img_p_raw[..., 32:-32, 32:-32]
                    y0 = top * 8 + 32; x0 = left * 8 + 32
                    img_t_valid = hr[..., y0:y0+256, x0:x0+256].clamp(-1, 1)
                    
                    if w['lpips'] > 0:
                        if lpips_fn_train is None:
                            lpips_fn_train = lpips.LPIPS(net='vgg').to(DEVICE).eval()
                            for p in lpips_fn_train.parameters():
                                p.requires_grad_(False)
                            print("✅ Training LPIPS initialized on GPU (lazy init).")
                        loss_lpips = lpips_fn_train(img_p_valid, img_t_valid).mean()

                    if w['edge_grad'] > 0 or w['flat_hf'] > 0:
                        loss_edge, loss_flat_hf, _ = edge_guided_losses(
                            img_p_valid, img_t_valid, q=EDGE_Q, pow_=EDGE_POW
                        )
                    

                inject_reg = compute_injection_scale_reg(pixart, INJECT_SCALE_REG_LAMBDA)
                loss = (
                    loss_v 
                    + w['latent_l1']*loss_latent_l1
                    + w['lpips']*loss_lpips
                    + w['edge_grad'] * loss_edge + w['flat_hf'] * loss_flat_hf
                    + inject_reg
                ) / GRAD_ACCUM_STEPS

            loss.backward()
            accum_micro_steps += 1

            if accum_micro_steps == GRAD_ACCUM_STEPS:
                torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                accum_micro_steps = 0

            if i % 10 == 0:
                pbar.set_postfix({
                    'v_loss': f"{loss_v:.3f}",
                    'lat_l1': f"{loss_latent_l1:.3f}",
                    'lp': f"{loss_lpips:.3f}",
                    'edge': f"{loss_edge.item():.3f}",
                    'flat_hf': f"{loss_flat_hf.item():.3f}",
                    'w_edge': f"{w['edge_grad']:.3f}",
                    'w_flat': f"{w['flat_hf']:.3f}",
                    'ireg': f"{inject_reg.item():.4f}",
                })

        if accum_micro_steps > 0 and not reached_max_steps:
            torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            accum_micro_steps = 0

        log_injection_scale_stats(pixart, prefix=f"[InjectScale][Ep{epoch+1}]")
        val_dict = validate(epoch, pixart, adapter, vae, val_loader, y, d_info, lpips_fn_val_cpu)
        if int(BEST_VAL_STEPS) in val_dict: metrics = val_dict[int(BEST_VAL_STEPS)]
        else: metrics = next(iter(val_dict.values()))
        best = save_smart(epoch, step, pixart, adapter, optimizer, best, metrics, dl_gen)

if __name__ == "__main__":
    main()
