"""MSM-QCA MAINLINE TRAINING SCRIPT (in-place replacement of legacy dualstream trainer).

Current mainline behavior:
- single-stage MSM-QCA training
- no dualstream/injection/style-fusion/kv-compress training branches
- explicit adapter CA blocks and strict MSM-QCA checkpoint metadata
"""
import os
import sys
import math
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import AutoencoderKL, DDIMScheduler
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import lpips

from diffusion import IDDPM
from diffusion.model.gaussian_diffusion import _extract_into_tensor
from diffusion.model.nets.PixArtSigma_SR import PixArtSigmaSR_XL_2
from diffusion.model.nets.adapter import build_adapter_msm_qca
from diffusion.model.utils import set_grad_checkpoint
from train_scripts.msm_qca_utils import (
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
    get_fixed_loss_weights,
    sample_t,
    decode_vae_sample_checkpointed,
    build_msm_qca_config,
    assert_msm_qca_config_compatible,
)

# ===== MSM-QCA structure config =====
ADAPTER_CA_BLOCK_IDS = [14, 18, 22, 26]
MEMORY_TOKEN_COUNTS = [int(x.strip()) for x in os.getenv("MEMORY_TOKEN_COUNTS", ",".join(map(str, DEFAULT_MEMORY_TOKEN_COUNTS))).split(",") if x.strip()]
RESAMPLER_DIM = int(os.getenv("RESAMPLER_DIM", str(DEFAULT_RESAMPLER_DIM)))
RESAMPLER_DEPTH = int(os.getenv("RESAMPLER_DEPTH", str(DEFAULT_RESAMPLER_DEPTH)))
RESAMPLER_HEADS = int(os.getenv("RESAMPLER_HEADS", str(DEFAULT_RESAMPLER_HEADS)))

# ===== training config =====
SEED = int(os.getenv("SEED", "3407"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))
MAX_TRAIN_STEPS = int(os.getenv("MAX_TRAIN_STEPS", "20000"))
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "16"))
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "1000"))
VAL_EVERY = int(os.getenv("VAL_EVERY", "1000"))
FAST_VAL_BATCHES = int(os.getenv("FAST_VAL_BATCHES", "2"))
FAST_VAL_STEPS = int(os.getenv("FAST_VAL_STEPS", "10"))
VAL_STEPS = int(os.getenv("VAL_STEPS", "50"))
USE_FAST_VAL = bool(int(os.getenv("USE_FAST_VAL", "0")))

T_SAMPLE_MODE = os.getenv("T_SAMPLE_MODE", "power")
T_SAMPLE_POWER = float(os.getenv("T_SAMPLE_POWER", "2.5"))
T_SAMPLE_MIN = int(os.getenv("T_SAMPLE_MIN", "0"))
T_SAMPLE_MAX = int(os.getenv("T_SAMPLE_MAX", "999"))
T_TWO_STAGE_SWITCH = int(os.getenv("T_TWO_STAGE_SWITCH", "15000"))
MIN_SNR_GAMMA = float(os.getenv("MIN_SNR_GAMMA", "5.0"))
LATENT_L1_T_MAX = int(os.getenv("LATENT_L1_T_MAX", "250"))
PIXEL_LOSS_T_MAX = int(os.getenv("PIXEL_LOSS_T_MAX", "250"))

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
OUT_DIR = os.path.join(OUT_BASE, "train_sigma_sr_vpred_dualstream")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
VIS_DIR = os.path.join(OUT_DIR, "vis")
LAST_CKPT_PATH = os.path.join(CKPT_DIR, "last.pth")
BEST_CKPT_PATH = os.path.join(CKPT_DIR, "best.pth")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

LORA_RANK = int(os.getenv("LORA_RANK", "4"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "4"))


def rgb01_to_y01(rgb01: torch.Tensor):
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    return (16.0 + 65.481 * r + 128.553 * g + 24.966 * b) / 255.0


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
    out = os.path.join(VIS_DIR, f"preview_step_{step:07d}.png")
    img = ((pred[0].detach().cpu().permute(1, 2, 0).float().numpy() + 1.0) * 127.5).clip(0, 255).astype("uint8")
    Image.fromarray(img).save(out)
    print(f"[preview] saved {out}")


def should_keep_ckpt(curr, best):
    # PSNR first, LPIPS tiebreak
    if best is None:
        return True
    c_psnr, _, c_lpips = curr
    b_psnr, _, b_lpips = best
    if c_psnr > b_psnr + 1e-9:
        return True
    if abs(c_psnr - b_psnr) <= 1e-9 and c_lpips < b_lpips:
        return True
    return False


@torch.no_grad()
def validate(step, pixart, adapter, vae, val_loader, y_embed, lpips_fn_cpu):
    pixart.eval()
    adapter.eval()

    steps = FAST_VAL_STEPS if USE_FAST_VAL else VAL_STEPS
    max_batches = FAST_VAL_BATCHES if USE_FAST_VAL else None

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="v_prediction",
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(steps, device=DEVICE)
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(SEED + int(step))

    data_info = {
        "img_hw": torch.tensor([[float(CROP_SIZE), float(CROP_SIZE)]], device=DEVICE),
        "aspect_ratio": torch.tensor([1.0], device=DEVICE),
    }

    psnrs, ssims, lpipss = [], [], []
    preview_pred = None
    for bi, batch in enumerate(val_loader):
        if (max_batches is not None) and (bi >= max_batches):
            break
        hr = batch["hr"].to(DEVICE)
        lr = batch["lr"].to(DEVICE)
        lr_small = batch["lr_small"].to(DEVICE)

        z_lr = vae.encode(lr).latent_dist.mean * vae.config.scaling_factor
        latents, run_timesteps = get_lq_init_latents(z_lr.to(COMPUTE_DTYPE), scheduler, steps, gen, 0.3, COMPUTE_DTYPE)

        aug_level = torch.zeros((latents.shape[0],), device=DEVICE, dtype=COMPUTE_DTYPE)
        for t in run_timesteps:
            t_b = torch.tensor([t], device=DEVICE).expand(latents.shape[0])
            t_embed = pixart.t_embedder(t_b.to(dtype=COMPUTE_DTYPE))
            cond = adapter(lr_small.to(dtype=COMPUTE_DTYPE), t_embed=t_embed)
            out = pixart(
                x=latents.to(COMPUTE_DTYPE), timestep=t_b, y=y_embed,
                aug_level=aug_level, mask=None, data_info=data_info,
                adapter_cond=cond, force_drop_ids=torch.ones(latents.shape[0], device=DEVICE)
            )
            latents = scheduler.step(out.float(), t, latents.float()).prev_sample

        pred = vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1)
        if preview_pred is None:
            preview_pred = pred

        pred01 = (pred + 1.0) / 2.0
        hr01 = (hr + 1.0) / 2.0
        py = rgb01_to_y01(pred01)[..., 4:-4, 4:-4]
        hy = rgb01_to_y01(hr01)[..., 4:-4, 4:-4]

        psnrs.append(float(psnr(py, hy, data_range=1.0).item()))
        ssims.append(float(ssim(py, hy, data_range=1.0).item()))
        lpipss.append(float(lpips_fn_cpu(pred.detach().cpu().float(), hr.detach().cpu().float()).mean().item()))

    pixart.train()
    adapter.train()
    if preview_pred is not None:
        save_preview_image(step, preview_pred)

    res = {
        "psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "ssim": float(np.mean(ssims)) if ssims else 0.0,
        "lpips": float(np.mean(lpipss)) if lpipss else 1e9,
    }
    print(f"[VAL@{steps}] step={step} PSNR={res['psnr']:.4f} SSIM={res['ssim']:.6f} LPIPS={res['lpips']:.6f}")
    return res


def save_ckpt(path, step, pixart, adapter, optimizer, best_metrics, msm_qca_config):
    sd = {
        "step": int(step),
        "pixart_keep": {k: v.detach().cpu().float() for k, v in pixart.state_dict().items()},
        "adapter": {k: v.detach().cpu().float() for k, v in adapter.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "best_eval_metrics": dict(best_metrics) if best_metrics is not None else None,
        "lora_rank": int(LORA_RANK),
        "lora_alpha": int(LORA_ALPHA),
        "msm_qca_config": msm_qca_config,
    }
    torch.save(sd, path)


def resume_if_possible(pixart, adapter, optimizer, msm_qca_config):
    if not os.path.isfile(LAST_CKPT_PATH):
        print("[resume] no last checkpoint, train from scratch")
        return 0, None
    ckpt = torch.load(LAST_CKPT_PATH, map_location="cpu")
    assert_msm_qca_config_compatible(msm_qca_config, ckpt.get("msm_qca_config", {}), context="train-resume")
    load_pixart_subset_compatible(pixart, ckpt.get("pixart_keep", {}), context="train-resume")
    miss, unexp = adapter.load_state_dict(ckpt["adapter"], strict=True)
    print(f"[train-resume] adapter strict load ok: missing={len(miss)} unexpected={len(unexp)}")
    optimizer.load_state_dict(ckpt["optimizer"])
    step = int(ckpt.get("step", 0))
    best_metrics = ckpt.get("best_eval_metrics", None)
    print(f"[resume] restored from {LAST_CKPT_PATH} @ step={step}")
    return step, best_metrics


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
    dl_gen = torch.Generator(); dl_gen.manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=dl_gen,
    )

    val_ds = RealSRPairedDataset(roots=VAL_REALSR_ROOTS, crop_size=CROP_SIZE, scale=SCALE)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=min(4, NUM_WORKERS))

    pixart = PixArtSigmaSR_XL_2(
        input_size=64,
        in_channels=4,
        out_channels=4,
        adapter_ca_block_ids=ADAPTER_CA_BLOCK_IDS,
    ).to(DEVICE)
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
    data_info = {
        "img_hw": torch.tensor([[float(CROP_SIZE), float(CROP_SIZE)]], device=DEVICE),
        "aspect_ratio": torch.tensor([1.0], device=DEVICE),
    }

    diffusion = IDDPM(str(1000))

    configure_trainable_msm_qca(pixart, adapter, disable_adapter=False, bridge_only_debug=False)
    optimizer = build_optimizer_msm_qca(
        pixart,
        adapter,
        disable_adapter=False,
        pixart_lr=1e-5,
        adapter_lr=3e-5,
        weight_decay=0.01,
    )

    lpips_fn_val_cpu = lpips.LPIPS(net='vgg').to("cpu").eval()
    for p in lpips_fn_val_cpu.parameters():
        p.requires_grad_(False)

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

    print(f"[MSM-QCA mainline] out_dir={OUT_DIR}")
    print(f"[MSM-QCA mainline] adapter_ca_block_ids={ADAPTER_CA_BLOCK_IDS}")

    step, best_metrics = resume_if_possible(pixart, adapter, optimizer, msm_qca_config)

    accum_micro_steps = 0
    optimizer.zero_grad()
    pbar = tqdm(total=MAX_TRAIN_STEPS, desc="train_msm_qca", initial=step)
    epoch = 0
    params_to_clip = [p for p in list(pixart.parameters()) + list(adapter.parameters()) if p.requires_grad]

    while step < MAX_TRAIN_STEPS:
        train_ds.set_epoch(epoch)
        for _, batch in enumerate(train_loader):
            hr = batch["hr"].to(DEVICE)
            lr = batch["lr"].to(DEVICE)
            lr_small = batch["lr_small"].to(DEVICE, dtype=COMPUTE_DTYPE)

            with torch.no_grad():
                zh = vae.encode(hr).latent_dist.mean * vae.config.scaling_factor

            t = sample_t(
                batch=zh.shape[0],
                device=DEVICE,
                step=step,
                mode=T_SAMPLE_MODE,
                power=T_SAMPLE_POWER,
                tmin=T_SAMPLE_MIN,
                tmax=T_SAMPLE_MAX,
                two_stage_switch=T_TWO_STAGE_SWITCH,
            )
            noise = torch.randn_like(zh)
            zt = diffusion.q_sample(zh, t, noise)
            t_embed = pixart.t_embedder(t.to(dtype=COMPUTE_DTYPE))

            with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE, enabled=(DEVICE == "cuda")):
                cond = adapter(lr_small, t_embed=t_embed)
                model_pred = pixart(
                    x=zt.to(COMPUTE_DTYPE), timestep=t, y=y,
                    aug_level=torch.zeros((zt.shape[0],), device=DEVICE, dtype=COMPUTE_DTYPE),
                    data_info=data_info, adapter_cond=cond,
                    force_drop_ids=torch.ones(zt.shape[0], device=DEVICE),
                ).float()

                alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, zh.shape)
                sigma_t = _extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, t, zh.shape)
                target_v = alpha_t * noise - sigma_t * zh.float()

                alpha_s = alpha_t[:, 0, 0, 0]
                sigma_s = sigma_t[:, 0, 0, 0]
                snr = (alpha_s ** 2) / (sigma_s ** 2)
                min_snr_gamma = torch.minimum(snr, torch.full_like(snr, float(MIN_SNR_GAMMA)))
                loss_weights = min_snr_gamma / snr
                loss_v = (F.mse_loss(model_pred, target_v, reduction='none').mean(dim=[1, 2, 3]) * loss_weights).mean()

                z0 = alpha_t * zt.float() - sigma_t * model_pred

                latent_l1_per = torch.mean(torch.abs(z0 - zh.float()), dim=[1, 2, 3])
                latent_l1_mask = (t <= int(LATENT_L1_T_MAX)).float()
                if float(latent_l1_mask.sum().item()) > 0:
                    loss_latent_l1 = (latent_l1_per * latent_l1_mask).sum() / latent_l1_mask.sum().clamp_min(1.0)
                else:
                    loss_latent_l1 = torch.zeros((), device=DEVICE, dtype=z0.dtype)

                w = get_fixed_loss_weights()
                loss_edge = torch.tensor(0.0, device=DEVICE)
                loss_lr_cons = torch.tensor(0.0, device=DEVICE)

                pixel_t_mask = (t <= int(PIXEL_LOSS_T_MAX))
                pixel_loss_num_samples = int(pixel_t_mask.sum().item())
                calc_pixel_loss = ((w['edge_grad'] > 0) or (w.get('lr_cons', 0.0) > 0)) and (pixel_loss_num_samples > 0)
                if calc_pixel_loss:
                    active_idx = torch.nonzero(pixel_t_mask, as_tuple=False).squeeze(1)
                    top = torch.randint(0, 25, (1,), device=DEVICE).item()
                    left = torch.randint(0, 25, (1,), device=DEVICE).item()
                    z0_sel = z0.index_select(0, active_idx)
                    hr_sel = hr.index_select(0, active_idx)
                    lr_sel = lr.index_select(0, active_idx)
                    z0_crop = z0_sel[..., top:top + 40, left:left + 40]
                    img_p_raw = decode_vae_sample_checkpointed(vae, z0_crop / vae.config.scaling_factor).clamp(-1, 1)
                    img_p_valid = img_p_raw[..., 32:-32, 32:-32]
                    y0 = top * 8 + 32
                    x0 = left * 8 + 32
                    img_t_valid = hr_sel[..., y0:y0 + 256, x0:x0 + 256].clamp(-1, 1)
                    if w['edge_grad'] > 0:
                        loss_edge, _, _ = edge_guided_losses(img_p_valid, img_t_valid)
                    if w.get('lr_cons', 0.0) > 0:
                        lr_patch = lr_sel[..., y0:y0 + 256, x0:x0 + 256].clamp(-1, 1)
                        loss_lr_cons = structure_consistency_loss(img_p_valid, lr_patch)

                loss = loss_v + w['latent_l1'] * loss_latent_l1 + w['edge_grad'] * loss_edge + w.get('lr_cons', 0.0) * loss_lr_cons

            (loss / GRAD_ACCUM_STEPS).backward()
            accum_micro_steps += 1

            if accum_micro_steps == GRAD_ACCUM_STEPS:
                torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_micro_steps = 0
                step += 1
                pbar.update(1)

                if step % 10 == 0:
                    ca_g = getattr(pixart, "_last_adapter_ca_gates", None)
                    ca_m = float(ca_g.mean().item()) if torch.is_tensor(ca_g) else 0.0
                    print(f"[step={step}] loss={float(loss.item()):.4f} v={float(loss_v.item()):.4f} ca_gate_mean={ca_m:.4f}")

                if step % SAVE_EVERY == 0:
                    save_ckpt(LAST_CKPT_PATH, step, pixart, adapter, optimizer, best_metrics, msm_qca_config)

                if step % VAL_EVERY == 0:
                    m = validate(step, pixart, adapter, vae, val_loader, y, lpips_fn_val_cpu)
                    curr = (m["psnr"], m["ssim"], m["lpips"])
                    best_triplet = None if best_metrics is None else (float(best_metrics["psnr"]), float(best_metrics["ssim"]), float(best_metrics["lpips"]))
                    if should_keep_ckpt(curr, best_triplet):
                        best_metrics = dict(m)
                        save_ckpt(BEST_CKPT_PATH, step, pixart, adapter, optimizer, best_metrics, msm_qca_config)
                        print(f"[best] updated {BEST_CKPT_PATH} with PSNR={m['psnr']:.4f} SSIM={m['ssim']:.6f} LPIPS={m['lpips']:.6f}")
                    save_ckpt(LAST_CKPT_PATH, step, pixart, adapter, optimizer, best_metrics, msm_qca_config)

                if step >= MAX_TRAIN_STEPS:
                    break

        # epoch-end leftover grad accumulation flush
        if accum_micro_steps > 0 and step < MAX_TRAIN_STEPS:
            torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            accum_micro_steps = 0
            step += 1
            pbar.update(1)

        epoch += 1

    save_ckpt(LAST_CKPT_PATH, step, pixart, adapter, optimizer, best_metrics, msm_qca_config)
    pbar.close()
    print("✅ MSM-QCA training finished")


if __name__ == "__main__":
    main()
