"""MSM-QCA main training entrypoint (independent from legacy dualstream shell)."""
import os
import sys
import random
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import AutoencoderKL

from diffusion import IDDPM
from diffusion.model.gaussian_diffusion import _extract_into_tensor
from diffusion.model.nets.PixArtSigma_SR import PixArtSigmaSR_XL_2
from diffusion.model.nets.adapter import build_adapter_msm_qca
from diffusion.model.utils import set_grad_checkpoint

# reuse only data/util pieces from legacy, but no legacy main/config shell
from train_scripts.train_sigma_sr_vpred_dualstream import (
    DF2K_Online_Dataset,
    seed_worker,
    load_state_dict_shape_compatible,
    LoRALinear,
    _block_id_from_name,
    _lora_target_kind,
    structure_consistency_loss,
    edge_guided_losses,
)

ADAPTER_CA_BLOCK_IDS = [14, 18, 22, 26]
MEMORY_TOKEN_COUNTS = [64, 32, 16]
RESAMPLER_DIM = 512
RESAMPLER_DEPTH = 2
RESAMPLER_HEADS = 8

SEED = int(os.getenv("SEED", "3407"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))
MAX_TRAIN_STEPS = int(os.getenv("MAX_TRAIN_STEPS", "20000"))
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "16"))

PRETRAINED_ROOT = os.getenv("DTSR_PRETRAINED_ROOT", "/home/hello/HJT/PixArt-sigma/output/pretrained_models")
PIXART_PATH = os.path.join(PRETRAINED_ROOT, "PixArt-Sigma-XL-2-512-MS.pth")
DIFFUSERS_ROOT = os.path.join(PRETRAINED_ROOT, "pixart_sigma_sdxlvae_T5_diffusers")
VAE_PATH = os.path.join(DIFFUSERS_ROOT, "vae")
NULL_T5_EMBED_PATH = os.path.join(PRETRAINED_ROOT, "null_t5_embed_sigma_300.pth")

OUT_BASE = os.getenv("DTSR_OUT_BASE", os.path.join(PROJECT_ROOT, "experiments_results"))
OUT_DIR = os.path.join(OUT_BASE, "train_sigma_sr_msm_qca")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
VIS_DIR = os.path.join(OUT_DIR, "vis")
LAST_CKPT_PATH = os.path.join(CKPT_DIR, "last.pth")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

LORA_RANK = int(os.getenv("LORA_RANK", "4"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "4"))


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_lora(model):
    cnt = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        bid = _block_id_from_name(name)
        if bid is None or not (0 <= bid <= 27):
            continue
        kind = _lora_target_kind(name)
        if kind != "attn":
            continue
        rank = int(LORA_RANK)
        parent = model.get_submodule(name.rsplit('.', 1)[0])
        child = name.rsplit('.', 1)[1]
        setattr(parent, child, LoRALinear(module, rank, alpha=float(LORA_ALPHA)))
        cnt += 1
    print(f"✅ Attention LoRA applied to {cnt} layers (rank={LORA_RANK}, alpha={LORA_ALPHA}).")


def configure_trainables(pixart: nn.Module, adapter: nn.Module):
    for _, p in pixart.named_parameters():
        p.requires_grad_(False)
    for _, p in adapter.named_parameters():
        p.requires_grad_(True)

    for n, p in pixart.named_parameters():
        if any(k in n for k in ["adapter_ca_norm_q", "adapter_ca_layers", "adapter_ca_out", "adapter_ca_gate", "final_layer", "lora_A", "lora_B"]):
            p.requires_grad_(True)


def build_optimizer(pixart: nn.Module, adapter: nn.Module):
    memory_bridge, adapter_backbone = [], []
    for n, p in adapter.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in ["mem_proj_", "resampler", "memory_out_proj", "memory_ln", "scale_embed", "q2", "q3", "q4"]):
            memory_bridge.append(p)
        else:
            adapter_backbone.append(p)

    pixart_readout_bridge, pixart_low_lr = [], []
    for n, p in pixart.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in ["adapter_ca_norm_q", "adapter_ca_layers", "adapter_ca_out", "adapter_ca_gate"]):
            pixart_readout_bridge.append(p)
        else:
            pixart_low_lr.append(p)

    optim_groups = []
    if memory_bridge:
        optim_groups.append({"params": memory_bridge, "lr": 1e-4, "weight_decay": 0.01, "name": "memory_bridge"})
    if adapter_backbone:
        optim_groups.append({"params": adapter_backbone, "lr": 3e-5, "weight_decay": 0.01, "name": "adapter_backbone"})
    if pixart_readout_bridge:
        optim_groups.append({"params": pixart_readout_bridge, "lr": 5e-5, "weight_decay": 0.01, "name": "pixart_readout_bridge"})
    if pixart_low_lr:
        optim_groups.append({"params": pixart_low_lr, "lr": 5e-6, "weight_decay": 0.01, "name": "pixart_low_lr"})

    for g in optim_groups:
        print(f"[OptimGroup] {g['name']}: lr={g['lr']} params={sum(p.numel() for p in g['params'])}")

    return torch.optim.AdamW(optim_groups)


def save_last(step, pixart, adapter, optimizer):
    sd = {
        "step": int(step),
        "pixart_keep": {k: v.detach().cpu().float() for k, v in pixart.state_dict().items()},
        "adapter": {k: v.detach().cpu().float() for k, v in adapter.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "lora_rank": int(LORA_RANK),
        "lora_alpha": int(LORA_ALPHA),
        "msm_qca_config": {
            "model_variant": "msm_qca",
            "adapter_ca_block_ids": list(ADAPTER_CA_BLOCK_IDS),
            "memory_token_counts": {"n2": MEMORY_TOKEN_COUNTS[0], "n3": MEMORY_TOKEN_COUNTS[1], "n4": MEMORY_TOKEN_COUNTS[2]},
            "resampler_dim": RESAMPLER_DIM,
            "resampler_depth": RESAMPLER_DEPTH,
            "resampler_heads": RESAMPLER_HEADS,
            "optimizer_group_lrs": {
                "memory_bridge": 1e-4,
                "adapter_backbone": 3e-5,
                "pixart_readout_bridge": 5e-5,
                "pixart_low_lr": 5e-6,
            },
        },
    }
    torch.save(sd, LAST_CKPT_PATH)


def main():
    seed_everything(SEED)

    train_ds = DF2K_Online_Dataset(crop_size=512, is_train=True, scale=4)
    dl_gen = torch.Generator(); dl_gen.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, worker_init_fn=seed_worker, generator=dl_gen)

    pixart = PixArtSigmaSR_XL_2(input_size=64, in_channels=4, out_channels=4, adapter_ca_block_ids=ADAPTER_CA_BLOCK_IDS).to(DEVICE)
    set_grad_checkpoint(pixart, use_fp32_attention=False, gc_step=1)

    base = torch.load(PIXART_PATH, map_location="cpu")
    if "state_dict" in base:
        base = base["state_dict"]
    if "pos_embed" in base:
        del base["pos_embed"]
    if hasattr(pixart, "load_pretrained_weights_with_zero_init"):
        pixart.load_pretrained_weights_with_zero_init(base)
    else:
        load_state_dict_shape_compatible(pixart, base, context="base-pretrain")

    adapter = build_adapter_msm_qca(hidden_size=1152).to(DEVICE).train()
    apply_lora(pixart)

    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE).float().eval()
    vae.enable_slicing()

    null_pack = torch.load(NULL_T5_EMBED_PATH, map_location="cpu")
    y = null_pack["y"].to(DEVICE)
    d_info = {"img_hw": torch.tensor([[512., 512.]], device=DEVICE), "aspect_ratio": torch.tensor([1.], device=DEVICE)}

    diffusion = IDDPM(str(1000))

    configure_trainables(pixart, adapter)
    optimizer = build_optimizer(pixart, adapter)
    print(f"[MSM-QCA Mainline] out_dir={OUT_DIR}")

    step = 0
    accum = 0
    optimizer.zero_grad()
    pbar = tqdm(total=MAX_TRAIN_STEPS, desc="train_msm_qca")
    while step < MAX_TRAIN_STEPS:
        for batch in train_loader:
            hr = batch['hr'].to(DEVICE)
            lr = batch['lr'].to(DEVICE)
            lr_small = batch['lr_small'].to(DEVICE, dtype=COMPUTE_DTYPE)

            with torch.no_grad():
                zh = vae.encode(hr).latent_dist.mean * vae.config.scaling_factor

            t = torch.randint(0, 1000, (zh.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(zh)
            zt = diffusion.q_sample(zh, t, noise)
            t_embed = pixart.t_embedder(t.to(dtype=COMPUTE_DTYPE))

            with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE, enabled=(DEVICE == "cuda")):
                cond = adapter(lr_small, t_embed=t_embed)
                out = pixart(
                    x=zt.to(COMPUTE_DTYPE), timestep=t, y=y,
                    aug_level=torch.zeros((zt.shape[0],), device=DEVICE, dtype=COMPUTE_DTYPE),
                    data_info=d_info, adapter_cond=cond,
                    force_drop_ids=torch.ones(zt.shape[0], device=DEVICE),
                ).float()

                alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, zh.shape)
                sigma_t = _extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, t, zh.shape)
                target_v = alpha_t * noise - sigma_t * zh.float()
                loss_v = F.mse_loss(out, target_v)

                z0 = alpha_t * zt.float() - sigma_t * out
                loss_latent_l1 = F.l1_loss(z0, zh.float())
                pred = vae.decode(z0 / vae.config.scaling_factor).sample.clamp(-1, 1)
                loss_lr_cons = structure_consistency_loss(pred, lr)
                loss_edge, _, _ = edge_guided_losses(pred, hr)
                loss = loss_v + 0.10 * loss_latent_l1 + 0.05 * loss_lr_cons + 0.01 * loss_edge

            (loss / GRAD_ACCUM_STEPS).backward()
            accum += 1
            if accum >= GRAD_ACCUM_STEPS:
                torch.nn.utils.clip_grad_norm_([p for p in list(pixart.parameters()) + list(adapter.parameters()) if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum = 0
                step += 1
                pbar.update(1)
                if step % 10 == 0:
                    ca_g = getattr(pixart, "_last_adapter_ca_gates", None)
                    ca_m = float(ca_g.mean().item()) if torch.is_tensor(ca_g) else 0.0
                    print(f"[step={step}] loss={float(loss.item()):.4f} v={float(loss_v.item()):.4f} ca_gate_mean={ca_m:.4f}")
                if step % 1000 == 0:
                    save_last(step, pixart, adapter, optimizer)
                if step >= MAX_TRAIN_STEPS:
                    break
        if step >= MAX_TRAIN_STEPS:
            break

    save_last(step, pixart, adapter, optimizer)
    pbar.close()
    print("✅ Training finished")


if __name__ == "__main__":
    main()
