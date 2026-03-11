"""MSM-QCA main training entrypoint (clean config surface)."""
import os

# Clean mainline entry delegates to legacy implementation after pinning MSM-QCA-only knobs.
# NOTE: this file is the intended current entrypoint.

# explicit, auditable architecture knobs
os.environ.setdefault("BRIDGE_ONLY_DEBUG", "0")
os.environ.setdefault("DTSR_INIT_CKPT", "")

from train_scripts import train_sigma_sr_vpred_dualstream as legacy  # LEGACY engine backend

# enforce mainline defaults
legacy.INIT_CKPT_PATH = ""
legacy.ADAPTER_CA_BLOCK_IDS = [14, 18, 22, 26]
legacy.BRIDGE_ONLY_DEBUG = (os.getenv("BRIDGE_ONLY_DEBUG", "0") == "1")

if __name__ == "__main__":
    print("[MSM-QCA Mainline] entry=train_sigma_sr_msm_qca.py")
    print(f"[MSM-QCA Mainline] ADAPTER_CA_BLOCK_IDS={legacy.ADAPTER_CA_BLOCK_IDS}")
    legacy.main()
