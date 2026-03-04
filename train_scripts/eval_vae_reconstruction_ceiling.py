import os
import glob
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm

import lpips
from diffusers import AutoencoderKL
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
DIFFUSERS_ROOT = os.path.join(PROJECT_ROOT, "output", "pretrained_models", "pixart_sigma_sdxlvae_T5_diffusers")
DEFAULT_VAE_PATH = os.path.join(DIFFUSERS_ROOT, "vae")


def rgb01_to_y01(rgb01: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    return (16.0 + 65.481 * r + 128.553 * g + 24.966 * b) / 255.0


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
    def __init__(self, hr_root: str, lr_root: str = None, crop_size: int = 512):
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

        lr_up_crop = None
        if self.lr_root:
            base = os.path.basename(hr_path)
            lr_name = base.replace(".png", "x4.png")
            lr_p = os.path.join(self.lr_root, lr_name)
            if os.path.exists(lr_p):
                lr_pil = Image.open(lr_p).convert("RGB")
                lr_aligned, hr_aligned = center_crop_aligned_pair(lr_pil, hr_pil, scale=4)
                lr_small = TF.center_crop(lr_aligned, (self.crop_size // 4, self.crop_size // 4))
                hr_crop = TF.center_crop(hr_aligned, (self.crop_size, self.crop_size))
                lr_up_crop = TF.resize(
                    lr_small,
                    (self.crop_size, self.crop_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                )

        if lr_up_crop is None:
            hr_crop = TF.center_crop(hr_pil, (self.crop_size, self.crop_size))
            w, h = hr_crop.size
            lr_small = hr_crop.resize((w // 4, h // 4), Image.BICUBIC)
            lr_up_crop = lr_small.resize((w, h), Image.BICUBIC)

        hr = self.norm(self.to_tensor(hr_crop))
        lr_up = self.norm(self.to_tensor(lr_up_crop))
        return {"hr": hr, "lr_up": lr_up, "path": hr_path}


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = DF2KValFixedDataset(hr_root=args.val_hr_dir, lr_root=args.val_lr_dir, crop_size=args.crop_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device).float().eval()
    vae.enable_slicing()

    lpips_fn = lpips.LPIPS(net="vgg").to("cpu").eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    psnr_hr, ssim_hr, lpips_hr = [], [], []
    psnr_lr, ssim_lr, lpips_lr = [], [], []

    sf = vae.config.scaling_factor

    for batch in tqdm(loader, desc="VAE reconstruction ceiling"):
        hr = batch["hr"].to(device)
        lr_up = batch["lr_up"].to(device)

        # Ceiling A: HR -> VAE encode/decode -> HR_recon
        z_hr = vae.encode(hr).latent_dist.mean * sf
        hr_rec = vae.decode(z_hr / sf).sample.clamp(-1, 1)

        # Ceiling B: upsampled LR -> VAE encode/decode (isolates VAE distortion on LR-conditioned input)
        z_lr = vae.encode(lr_up).latent_dist.mean * sf
        lr_rec = vae.decode(z_lr / sf).sample.clamp(-1, 1)

        h01 = (hr + 1.0) * 0.5
        hr01 = (hr_rec + 1.0) * 0.5
        lr01 = (lr_rec + 1.0) * 0.5

        hy = rgb01_to_y01(h01)[..., 4:-4, 4:-4]
        hry = rgb01_to_y01(hr01)[..., 4:-4, 4:-4]
        lry = rgb01_to_y01(lr01)[..., 4:-4, 4:-4]

        psnr_hr.append(psnr(hry, hy, data_range=1.0).item())
        ssim_hr.append(ssim(hry, hy, data_range=1.0).item())
        psnr_lr.append(psnr(lry, hy, data_range=1.0).item())
        ssim_lr.append(ssim(lry, hy, data_range=1.0).item())

        hr_rec_cpu = hr_rec.detach().to("cpu", dtype=torch.float32)
        lr_rec_cpu = lr_rec.detach().to("cpu", dtype=torch.float32)
        hr_cpu = hr.detach().to("cpu", dtype=torch.float32)

        lpips_hr.append(lpips_fn(hr_rec_cpu, hr_cpu).mean().item())
        lpips_lr.append(lpips_fn(lr_rec_cpu, hr_cpu).mean().item())

    def _avg(x):
        return float(sum(x) / max(1, len(x)))

    print("\n=== VAE Reconstruction Ceiling Report ===")
    print(f"Samples: {len(ds)}")
    print("[A] HR -> VAE -> HR")
    print(f"  PSNR(Y): {_avg(psnr_hr):.3f}")
    print(f"  SSIM(Y): {_avg(ssim_hr):.4f}")
    print(f"  LPIPS : {_avg(lpips_hr):.4f}")
    print("[B] LR(up) -> VAE -> image (vs HR target)")
    print(f"  PSNR(Y): {_avg(psnr_lr):.3f}")
    print(f"  SSIM(Y): {_avg(ssim_lr):.4f}")
    print(f"  LPIPS : {_avg(lpips_lr):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VAE reconstruction ceiling for SR pipeline")
    parser.add_argument("--vae-path", type=str, default=DEFAULT_VAE_PATH)
    parser.add_argument("--val-hr-dir", type=str, required=True)
    parser.add_argument("--val-lr-dir", type=str, default=None)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
