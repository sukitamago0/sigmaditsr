"""MSM-QCA current mainline.

Main entries:
- train: train_scripts/train_sigma_sr_msm_qca.py
- infer: train_scripts/infer_sr_single_ddim100_msm_qca.py
- eval : train_scripts/eval_sr_baseline_vs_model.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import AutoencoderKL, DDIMScheduler

from diffusion import IDDPM
from diffusion.model.gaussian_diffusion import _extract_into_tensor
from diffusion.model.nets.PixArtSigma_SR import PixArtSigmaSR_XL_2
from diffusion.model.nets.adapter import build_adapter_msm_qca
from diffusion.model.utils import set_grad_checkpoint
from train_scripts.msm_qca_utils import (
    DEFAULT_ADAPTER_CA_BLOCK_IDS,
    DEFAULT_MEMORY_TOKEN_COUNTS,
    DEFAULT_RESAMPLER_DIM,
    DEFAULT_RESAMPLER_DEPTH,
    DEFAULT_RESAMPLER_HEADS,
    DEFAULT_DATA_CONFIG,
    DF2KOnlineDataset,
    RealSRPairedDataset,
    seed_everything,
    seed_worker,
    load_state_dict_shape_compatible,
    load_pixart_subset_compatible,
    apply_lora_attn_only,
    configure_trainable_msm_qca,
    build_optimizer_msm_qca,
    optimizer_group_lrs,
    structure_consistency_loss,
    edge_guided_losses,
    build_msm_qca_config,
    assert_msm_qca_config_compatible,
)

ADAPTER_CA_BLOCK_IDS = [int(x.strip()) for x in os.getenv("ADAPTER_CA_BLOCK_IDS", ",".join(map(str, DEFAULT_ADAPTER_CA_BLOCK_IDS))).split(",") if x.strip()]
MEMORY_TOKEN_COUNTS = [int(x.strip()) for x in os.getenv("MEMORY_TOKEN_COUNTS", ",".join(map(str, DEFAULT_MEMORY_TOKEN_COUNTS))).split(",") if x.strip()]
RESAMPLER_DIM = int(os.getenv("RESAMPLER_DIM", str(DEFAULT_RESAMPLER_DIM)))
RESAMPLER_DEPTH = int(os.getenv("RESAMPLER_DEPTH", str(DEFAULT_RESAMPLER_DEPTH)))
RESAMPLER_HEADS = int(os.getenv("RESAMPLER_HEADS", str(DEFAULT_RESAMPLER_HEADS)))

SEED = int(os.getenv("SEED", "3407"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))
MAX_TRAIN_STEPS = int(os.getenv("MAX_TRAIN_STEPS", "20000"))
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "16"))
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "1000"))
VAL_EVERY = int(os.getenv("VAL_EVERY", "1000"))
VAL_STEPS = int(os.getenv("VAL_STEPS", "50"))
MAX_VAL_SAMPLES = int(os.getenv("MAX_VAL_SAMPLES", "16"))

DATASET_NAME = os.getenv("TRAIN_DATASET_NAME", "DF2K+RealSR")
CROP_SIZE = int(os.getenv("CROP_SIZE", str(DEFAULT_DATA_CONFIG["crop_size"])))
SCALE = int(os.getenv("SCALE", str(DEFAULT_DATA_CONFIG["scale"])))
TRAIN_DF2K_HR_DIR = os.getenv("TRAIN_DF2K_HR_DIR", DEFAULT_DATA_CONFIG["train_df2k_hr_dir"])
TRAIN_DF2K_LR_DIR = os.getenv("TRAIN_DF2K_LR_DIR", DEFAULT_DATA_CONFIG["train_df2k_lr_dir"])
TRAIN_REALSR_ROOTS = [x.strip() for x in os.getenv("TRAIN_REALSR_ROOTS", ",".join(DEFAULT_DATA_CONFIG["train_realsr_roots"])).split(",") if x.strip()]
VAL_REALSR_ROOTS = [x.strip() for x in os.getenv("VAL_REALSR_ROOTS", ",".join(DEFAULT_DATA_CONFIG["val_realsr_roots"])).split(",") if x.strip()]

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
BEST_CKPT_PATH = os.path.join(CKPT_DIR, "best.pth")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

LORA_RANK = int(os.getenv("LORA_RANK", "4"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "4"))


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


def save_preview_image(step: int, pred: torch.Tensor):
    img = ((pred[0].detach().cpu().permute(1, 2, 0).float().numpy() + 1.0) * 127.5).clip(0, 255).astype("uint8")
    out = os.path.join(VIS_DIR, f"preview_step_{step:07d}.png")
    Image.fromarray(img).save(out)
    print(f"[preview] saved {out}")


@torch.no_grad()
def run_validation(step, pixart, adapter, vae, y_embed, val_loader):
    if val_loader is None:
        return None
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="v_prediction",
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(VAL_STEPS, device=DEVICE)
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(SEED + step)
    data_info = {
        "img_hw": torch.tensor([[float(CROP_SIZE), float(CROP_SIZE)]], device=DEVICE),
        "aspect_ratio": torch.tensor([1.0], device=DEVICE),
    }

    pixart.eval()
    adapter.eval()
    losses = []
    preview_pred = None
    for i, batch in enumerate(val_loader):
        if i >= MAX_VAL_SAMPLES:
            break
        hr = batch["hr"].to(DEVICE)
        lr = batch["lr"].to(DEVICE)
        lr_small = batch["lr_small"].to(DEVICE)

        z_hr = vae.encode(hr).latent_dist.mean * vae.config.scaling_factor
        z_lr = vae.encode(lr).latent_dist.mean * vae.config.scaling_factor
        latents, run_timesteps = get_lq_init_latents(z_lr.to(COMPUTE_DTYPE), scheduler, VAL_STEPS, gen, 0.3, COMPUTE_DTYPE)
        for t in run_timesteps:
            t_b = torch.tensor([t], device=DEVICE).expand(latents.shape[0])
            t_embed = pixart.t_embedder(t_b.to(dtype=COMPUTE_DTYPE))
            cond = adapter(lr_small.to(dtype=COMPUTE_DTYPE), t_embed=t_embed)
            out = pixart(
                x=latents.to(COMPUTE_DTYPE), timestep=t_b, y=y_embed,
                aug_level=torch.zeros((latents.shape[0],), device=DEVICE, dtype=COMPUTE_DTYPE),
                mask=None, data_info=data_info, adapter_cond=cond,
                force_drop_ids=torch.ones(latents.shape[0], device=DEVICE),
            )
            latents = scheduler.step(out.float(), t, latents.float()).prev_sample

        pred = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)
        losses.append(float(F.l1_loss(pred.float(), hr.float()).item()))
        if preview_pred is None:
            preview_pred = pred

    pixart.train()
    adapter.train()
    if preview_pred is not None:
        save_preview_image(step, preview_pred)
    val_l1 = float(sum(losses) / max(1, len(losses)))
    print(f"[val step={step}] mean_l1={val_l1:.6f} samples={len(losses)}")
    return val_l1


def save_ckpt(path, step, pixart, adapter, optimizer, best_val_l1, msm_qca_config):
    torch.save(
        {
            "step": int(step),
            "pixart_keep": {k: v.detach().cpu().float() for k, v in pixart.state_dict().items()},
            "adapter": {k: v.detach().cpu().float() for k, v in adapter.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "best_val_l1": float(best_val_l1),
            "lora_rank": int(LORA_RANK),
            "lora_alpha": int(LORA_ALPHA),
            "msm_qca_config": msm_qca_config,
        },
        path,
    )


def maybe_resume(pixart, adapter, optimizer, msm_qca_config):
    if not os.path.isfile(LAST_CKPT_PATH):
        print("[resume] no last checkpoint, train from scratch")
        return 0, float("inf")
    ckpt = torch.load(LAST_CKPT_PATH, map_location="cpu")
    assert_msm_qca_config_compatible(msm_qca_config, ckpt.get("msm_qca_config", {}), context="train-resume")
    load_pixart_subset_compatible(pixart, ckpt.get("pixart_keep", {}), context="train-resume")
    miss, unexp = adapter.load_state_dict(ckpt["adapter"], strict=True)
    print(f"[train-resume] adapter strict load ok: missing={len(miss)} unexpected={len(unexp)}")
    optimizer.load_state_dict(ckpt["optimizer"])
    step = int(ckpt.get("step", 0))
    best_val_l1 = float(ckpt.get("best_val_l1", float("inf")))
    print(f"[resume] restored from {LAST_CKPT_PATH} @ step={step} best_val_l1={best_val_l1:.6f}")
    return step, best_val_l1


def main():
    seed_everything(SEED)

    train_ds = DF2KOnlineDataset(
        crop_size=CROP_SIZE,
        is_train=True,
        scale=SCALE,
        df2k_hr_dir=TRAIN_DF2K_HR_DIR,
        df2k_lr_dir=TRAIN_DF2K_LR_DIR,
        realsr_roots=TRAIN_REALSR_ROOTS,
        seed=SEED,
    )
    dl_gen = torch.Generator()
    dl_gen.manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=dl_gen,
    )

    val_loader = None
    try:
        val_ds = RealSRPairedDataset(roots=VAL_REALSR_ROOTS, crop_size=CROP_SIZE, scale=SCALE)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=min(NUM_WORKERS, 4))
        print(f"[val] enabled RealSR validation with {len(val_ds)} samples")
    except Exception as e:
        print(f"[val] disabled: {e}")

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
    apply_lora_attn_only(pixart, rank=LORA_RANK, alpha=LORA_ALPHA)

    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE).float().eval()
    vae.enable_slicing()

    null_pack = torch.load(NULL_T5_EMBED_PATH, map_location="cpu")
    y = null_pack["y"].to(DEVICE)
    d_info = {"img_hw": torch.tensor([[float(CROP_SIZE), float(CROP_SIZE)]], device=DEVICE), "aspect_ratio": torch.tensor([1.0], device=DEVICE)}

    diffusion = IDDPM(str(1000))

    configure_trainable_msm_qca(pixart, adapter, disable_adapter=False, bridge_only_debug=False)
    optimizer = build_optimizer_msm_qca(pixart, adapter, disable_adapter=False, pixart_lr=1e-5, adapter_lr=3e-5, weight_decay=0.01)

    msm_qca_config = build_msm_qca_config(
        adapter_ca_block_ids=ADAPTER_CA_BLOCK_IDS,
        memory_token_counts=MEMORY_TOKEN_COUNTS,
        resampler_dim=RESAMPLER_DIM,
        resampler_depth=RESAMPLER_DEPTH,
        resampler_heads=RESAMPLER_HEADS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        max_train_steps=MAX_TRAIN_STEPS,
        dataset_name=DATASET_NAME,
        crop_size=CROP_SIZE,
        scale=SCALE,
        optimizer_lrs=optimizer_group_lrs(pixart_lr=1e-5, adapter_lr=3e-5),
    )

    print(f"[MSM-QCA Mainline] out_dir={OUT_DIR}")
    print(f"[MSM-QCA Mainline] adapter_ca_block_ids={ADAPTER_CA_BLOCK_IDS}")

    step, best_val_l1 = maybe_resume(pixart, adapter, optimizer, msm_qca_config)

    accum = 0
    optimizer.zero_grad()
    pbar = tqdm(total=MAX_TRAIN_STEPS, desc="train_msm_qca", initial=step)
    while step < MAX_TRAIN_STEPS:
        for batch in train_loader:
            hr = batch["hr"].to(DEVICE)
            lr = batch["lr"].to(DEVICE)
            lr_small = batch["lr_small"].to(DEVICE, dtype=COMPUTE_DTYPE)

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

                if step % SAVE_EVERY == 0:
                    save_ckpt(LAST_CKPT_PATH, step, pixart, adapter, optimizer, best_val_l1, msm_qca_config)

                if (val_loader is not None) and (step % VAL_EVERY == 0):
                    val_l1 = run_validation(step, pixart, adapter, vae, y, val_loader)
                    if val_l1 is not None and val_l1 < best_val_l1:
                        best_val_l1 = val_l1
                        save_ckpt(BEST_CKPT_PATH, step, pixart, adapter, optimizer, best_val_l1, msm_qca_config)
                        print(f"[best] updated {BEST_CKPT_PATH} with val_l1={best_val_l1:.6f}")
                    save_ckpt(LAST_CKPT_PATH, step, pixart, adapter, optimizer, best_val_l1, msm_qca_config)

                if step >= MAX_TRAIN_STEPS:
                    break
        if step >= MAX_TRAIN_STEPS:
            break

    save_ckpt(LAST_CKPT_PATH, step, pixart, adapter, optimizer, best_val_l1, msm_qca_config)
    pbar.close()
    print("✅ Training finished")


if __name__ == "__main__":
    main()
