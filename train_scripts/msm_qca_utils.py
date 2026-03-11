import glob
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


DEFAULT_ADAPTER_CA_BLOCK_IDS = [14, 18, 22, 26]
DEFAULT_MEMORY_TOKEN_COUNTS = [64, 32, 16]
DEFAULT_RESAMPLER_DIM = 512
DEFAULT_RESAMPLER_DEPTH = 2
DEFAULT_RESAMPLER_HEADS = 8


DEFAULT_DATA_CONFIG = {
    "train_df2k_hr_dir": os.getenv("TRAIN_DF2K_HR_DIR", "/data/DF2K/DF2K_train_HR"),
    "train_df2k_lr_dir": os.getenv("TRAIN_DF2K_LR_DIR", "/data/DF2K/DF2K_train_LR_unknown"),
    "train_realsr_roots": [
        x.strip()
        for x in os.getenv("TRAIN_REALSR_ROOTS", "/data/RealSR/Canon/Train/4,/data/RealSR/Nikon/Train/4").split(",")
        if x.strip()
    ],
    "val_realsr_roots": [
        x.strip()
        for x in os.getenv("VAL_REALSR_ROOTS", "/data/RealSR/Nikon/Test/4,/data/RealSR/Canon/Test/4").split(",")
        if x.strip()
    ],
    "crop_size": int(os.getenv("CROP_SIZE", "512")),
    "scale": int(os.getenv("SCALE", "4")),
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


class DF2KOnlineDataset(Dataset):
    def __init__(self, crop_size=512, is_train=True, scale=4, df2k_hr_dir="", df2k_lr_dir="", realsr_roots=None, seed=3407):
        self.pairs = build_paired_file_list(df2k_hr_dir, df2k_lr_dir, realsr_roots or [])
        if len(self.pairs) == 0:
            raise RuntimeError("No LR/HR pairs found from DF2K/RealSR training roots.")
        self.crop_size = int(crop_size)
        self.lr_patch = self.crop_size // int(scale)
        self.scale = int(scale)
        self.is_train = bool(is_train)
        self.seed = int(seed)
        self.norm = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        self.to_tensor = transforms.ToTensor()
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _make_generator(self, idx: int):
        gen = torch.Generator()
        gen.manual_seed(self.seed + (self.epoch * 1_000_000) + int(idx))
        return gen

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        try:
            lr_path, hr_path = self.pairs[idx]
            lr_pil = Image.open(lr_path).convert("RGB")
            hr_pil = Image.open(hr_path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        lr_aligned, hr_aligned = center_crop_aligned_pair(lr_pil, hr_pil, scale=self.scale)
        if self.is_train:
            gen = self._make_generator(idx)
            w_l, h_l = lr_aligned.size
            if h_l < self.lr_patch or w_l < self.lr_patch:
                lr_aligned = TF.resize(lr_aligned, (self.lr_patch, self.lr_patch), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
                hr_aligned = TF.resize(hr_aligned, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
                x, y = 0, 0
            else:
                x = int(torch.randint(0, w_l - self.lr_patch + 1, (1,), generator=gen).item())
                y = int(torch.randint(0, h_l - self.lr_patch + 1, (1,), generator=gen).item())
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
        return {
            "hr": self.norm(self.to_tensor(hr_crop)),
            "lr": self.norm(self.to_tensor(lr_up)),
            "lr_small": self.norm(self.to_tensor(lr_crop)),
            "path": hr_path,
        }


class RealSRPairedDataset(Dataset):
    def __init__(self, roots, crop_size=512, scale=4):
        self.crop_size = int(crop_size)
        self.scale = int(scale)
        self.norm = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        self.to_tensor = transforms.ToTensor()
        self.pairs = []
        for root in roots:
            if not os.path.isdir(str(root)):
                continue
            for hr_path in sorted(glob.glob(os.path.join(str(root), "*_HR.png"))):
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
        lr_aligned, hr_aligned = center_crop_aligned_pair(lr_pil, hr_pil, scale=self.scale)
        lr_small_pil = TF.center_crop(lr_aligned, (self.crop_size // self.scale, self.crop_size // self.scale))
        hr_crop = TF.center_crop(hr_aligned, (self.crop_size, self.crop_size))
        lr_up_pil = TF.resize(lr_small_pil, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        return {
            "hr": self.norm(self.to_tensor(hr_crop)),
            "lr": self.norm(self.to_tensor(lr_up_pil)),
            "lr_small": self.norm(self.to_tensor(lr_small_pil)),
            "path": hr_path,
        }


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
    m = re.search(r"blocks\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _lora_target_kind(module_name: str):
    if ("attn.qkv" in module_name) or ("attn.proj" in module_name):
        return "attn"
    return "other"


def apply_lora_attn_only(model: nn.Module, rank: int = 4, alpha: int = 4):
    cnt = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        bid = _block_id_from_name(name)
        if bid is None or not (0 <= bid <= 27):
            continue
        if _lora_target_kind(name) != "attn":
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        child = name.rsplit(".", 1)[1]
        setattr(parent, child, LoRALinear(module, int(rank), alpha=float(alpha)))
        cnt += 1
    print(f"✅ Attention LoRA applied to {cnt} layers (rank={rank}, alpha={alpha}).")


def configure_trainable_msm_qca(pixart: nn.Module, adapter: nn.Module, disable_adapter: bool = False, bridge_only_debug: bool = False):
    for _, p in pixart.named_parameters():
        p.requires_grad_(False)
    for _, p in adapter.named_parameters():
        p.requires_grad_(not disable_adapter)

    allow = ["adapter_ca_norm_q", "adapter_ca_layers", "adapter_ca_out", "adapter_ca_gate", "final_layer", "lora_A", "lora_B"]
    for n, p in pixart.named_parameters():
        if any(k in n for k in allow):
            p.requires_grad_(True)

    if (not disable_adapter) and bridge_only_debug:
        bridge_keys = ["mem_proj_", "resampler", "memory_out_proj", "memory_ln", "scale_embed", "q2", "q3", "q4"]
        for n, p in adapter.named_parameters():
            p.requires_grad_(any(k in n for k in bridge_keys))


def build_optimizer_msm_qca(
    pixart: nn.Module,
    adapter: nn.Module,
    *,
    disable_adapter: bool = False,
    memory_bridge_lr: float = 1e-4,
    adapter_backbone_lr: float = 3e-5,
    pixart_readout_bridge_lr: float = 5e-5,
    pixart_low_lr: float = 5e-6,
    weight_decay: float = 0.01,
):
    pixart_readout_bridge, pixart_low_lr = [], []
    for n, p in pixart.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in ["adapter_ca_norm_q", "adapter_ca_layers", "adapter_ca_out", "adapter_ca_gate"]):
            pixart_readout_bridge.append(p)
        else:
            pixart_low_lr.append(p)

    groups = []
    if pixart_readout_bridge:
        groups.append({"params": pixart_readout_bridge, "lr": float(pixart_readout_bridge_lr), "weight_decay": weight_decay, "name": "pixart_readout_bridge"})
    if pixart_low_lr:
        groups.append({"params": pixart_low_lr, "lr": float(pixart_low_lr), "weight_decay": weight_decay, "name": "pixart_low_lr"})

    if not disable_adapter:
        memory_bridge, adapter_backbone = [], []
        for n, p in adapter.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in n for k in ["mem_proj_", "resampler", "memory_out_proj", "memory_ln", "scale_embed", "q2", "q3", "q4"]):
                memory_bridge.append(p)
            else:
                adapter_backbone.append(p)
        if memory_bridge:
            groups.append({"params": memory_bridge, "lr": float(memory_bridge_lr), "weight_decay": weight_decay, "name": "memory_bridge"})
        if adapter_backbone:
            groups.append({"params": adapter_backbone, "lr": float(adapter_backbone_lr), "weight_decay": weight_decay, "name": "adapter_backbone"})

    for g in groups:
        print(f"[OptimGroup] {g['name']}: lr={g['lr']} params={sum(p.numel() for p in g['params'])}")

    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def optimizer_group_lrs(
    *,
    memory_bridge_lr: float = 1e-4,
    adapter_backbone_lr: float = 3e-5,
    pixart_readout_bridge_lr: float = 5e-5,
    pixart_low_lr: float = 5e-6,
):
    return {
        "memory_bridge": float(memory_bridge_lr),
        "adapter_backbone": float(adapter_backbone_lr),
        "pixart_readout_bridge": float(pixart_readout_bridge_lr),
        "pixart_low_lr": float(pixart_low_lr),
    }


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
    return missing, unexpected, skipped


def load_pixart_subset_compatible(pixart: nn.Module, saved_trainable: dict, context: str):
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
def edge_guided_losses(pred_m11: torch.Tensor, gt_m11: torch.Tensor, q: float = 0.90, pow_: float = 0.50, eps: float = 1e-6):
    x = _to_luma01(gt_m11)
    gx = F.conv2d(x, _SOBEL_X.to(x.device), padding=1)
    gy = F.conv2d(x, _SOBEL_Y.to(x.device), padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + eps)
    flat = mag.flatten(1)
    denom = torch.quantile(flat, q, dim=1, keepdim=True).clamp_min(eps)
    m = (flat / denom).view_as(mag).clamp(0.0, 1.0).pow(pow_)

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
        "mse": 1.0,
        "latent_l1": 0.10,
        "lpips": 0.0,
        "edge_grad": 0.01,
        "flat_hf": 0.0,
        "lr_cons": 0.05,
    }


def sample_t(batch: int, device: str, step: int, *, mode: str = "power", power: float = 2.5, tmin: int = 0, tmax: int = 999, two_stage_switch: int = 15000) -> torch.Tensor:
    tmin = int(max(0, tmin))
    tmax = int(min(999, tmax))
    if tmax < tmin:
        tmax = tmin

    mode = str(mode).lower()
    if mode == "uniform":
        return torch.randint(tmin, tmax + 1, (batch,), device=device).long()
    if mode == "power":
        u = torch.rand((batch,), device=device)
        span = float(max(1, tmax - tmin))
        t = torch.floor((u ** float(power)) * span + tmin)
        return t.clamp(tmin, tmax).long()
    if mode == "two_stage":
        if int(step) < int(two_stage_switch):
            u = torch.rand((batch,), device=device)
            span = float(max(1, tmax - tmin))
            t = torch.floor((u ** float(power)) * span + tmin)
            return t.clamp(tmin, tmax).long()
        return torch.randint(tmin, tmax + 1, (batch,), device=device).long()
    raise ValueError(f"Unknown t-sample mode: {mode}")


def decode_vae_sample_checkpointed(vae: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    def _decode_fn(z):
        return vae.decode(z).sample
    return torch.utils.checkpoint.checkpoint(_decode_fn, latents, use_reentrant=False)

def build_msm_qca_config(*, adapter_ca_block_ids, memory_token_counts, resampler_dim, resampler_depth, resampler_heads, batch_size, grad_accum_steps, max_train_steps, dataset_name, crop_size, scale, optimizer_lrs):
    return {
        "model_variant": "msm_qca",
        "adapter_ca_block_ids": list(adapter_ca_block_ids),
        "memory_token_counts": {
            "n2": int(memory_token_counts[0]),
            "n3": int(memory_token_counts[1]),
            "n4": int(memory_token_counts[2]),
        },
        "resampler_dim": int(resampler_dim),
        "resampler_depth": int(resampler_depth),
        "resampler_heads": int(resampler_heads),
        "train_hparams": {
            "batch_size": int(batch_size),
            "grad_accum_steps": int(grad_accum_steps),
            "max_train_steps": int(max_train_steps),
        },
        "dataset": {
            "name": str(dataset_name),
            "crop_size": int(crop_size),
            "scale": int(scale),
        },
        "optimizer_group_lrs": dict(optimizer_lrs),
    }


def assert_msm_qca_config_compatible(current_cfg: dict, ckpt_cfg: dict, context: str):
    if not ckpt_cfg:
        print(f"[{context}] ⚠️ ckpt missing msm_qca_config; skip strict meta check")
        return
    mismatches = []

    def _cmp(name, a, b):
        if a != b:
            mismatches.append((name, a, b))

    _cmp("adapter_ca_block_ids", list(current_cfg.get("adapter_ca_block_ids", [])), list(ckpt_cfg.get("adapter_ca_block_ids", [])))
    _cmp("memory_token_counts", dict(current_cfg.get("memory_token_counts", {})), dict(ckpt_cfg.get("memory_token_counts", {})))
    _cmp("resampler_dim", int(current_cfg.get("resampler_dim", -1)), int(ckpt_cfg.get("resampler_dim", -1)))
    _cmp("resampler_depth", int(current_cfg.get("resampler_depth", -1)), int(ckpt_cfg.get("resampler_depth", -1)))
    _cmp("resampler_heads", int(current_cfg.get("resampler_heads", -1)), int(ckpt_cfg.get("resampler_heads", -1)))
    if mismatches:
        detail = "; ".join([f"{k}: current={a} ckpt={b}" for k, a, b in mismatches])
        raise RuntimeError(f"[{context}] MSM-QCA config mismatch: {detail}")
    print(f"[{context}] ✅ msm_qca_config compatible")
