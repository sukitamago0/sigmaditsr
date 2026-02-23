# -*- coding: utf-8 -*-
"""
DDIM / IDDPM 一致性判定脚本（独立脚本，不修改训练脚本）

目标（对应你当前 v8 训练/验证逻辑）：
1) 判定训练端 IDDPM 与验证端 DDIMScheduler 的基础噪声日程是否一致（betas / alphas_cumprod）
2) 判定前向加噪是否一致：IDDPM.q_sample(x0,t,eps) vs DDIMScheduler.add_noise(x0,eps,t)
3) 判定 V-pred 系数是否一致：
   alpha_t = sqrt(alphas_cumprod[t])
   sigma_t = sqrt(1 - alphas_cumprod[t])

最终输出：
- FINAL_JUDGMENT: OK_TO_TRAIN（可开始训练 / 可继续训练）
- FINAL_JUDGMENT: BLOCK_TRAINING（必须先修改）

说明：
- 该脚本会始终给出最终结论，不会因单个检查项的超严阈值而提前中断。
- 会区分“浮点累计误差（可接受）”与“数学定义不一致（不可接受）”。
"""

import os
import sys
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

# -----------------------------
# Path setup: 兼容你的 PixArt 仓库结构（scripts/ 下运行）
# -----------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 依赖：与你 v8 脚本一致
from diffusion import IDDPM
from diffusers import DDIMScheduler


# -----------------------------
# 工具函数
# -----------------------------
def to_torch_1d(x) -> torch.Tensor:
    """把 numpy / torch / list 转成 CPU float64 1D tensor."""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu().to(torch.float64).flatten()
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x).detach().cpu().to(torch.float64).flatten()
    else:
        t = torch.tensor(x, dtype=torch.float64).flatten()
    return t


def array_compare(name: str, a, b, atol: float, rtol: float) -> Dict[str, Any]:
    ta = to_torch_1d(a)
    tb = to_torch_1d(b)
    if ta.numel() != tb.numel():
        return {
            "name": name,
            "ok": False,
            "error": f"shape mismatch: {tuple(ta.shape)} vs {tuple(tb.shape)}"
        }
    diff = (ta - tb).abs()
    max_abs = float(diff.max().item()) if diff.numel() > 0 else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() > 0 else 0.0
    worst_idx = int(diff.argmax().item()) if diff.numel() > 0 else -1
    allclose = bool(torch.allclose(ta, tb, atol=atol, rtol=rtol))

    aw = float(ta[worst_idx].item()) if worst_idx >= 0 else 0.0
    bw = float(tb[worst_idx].item()) if worst_idx >= 0 else 0.0
    return {
        "name": name,
        "ok": allclose,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "worst_idx": worst_idx,
        "a_worst": aw,
        "b_worst": bw,
        "atol": atol,
        "rtol": rtol,
        "len": int(ta.numel()),
    }


def print_array_result(res: Dict[str, Any], prefix: str = "[ARRAY]"):
    if "error" in res:
        print(f"{prefix} {res['name']}: FAIL | {res['error']}")
        return
    print(
        f"{prefix} {res['name']}: "
        f"allclose={res['ok']} | "
        f"max_abs={res['max_abs']:.3e} | mean_abs={res['mean_abs']:.3e} | "
        f"worst_idx={res['worst_idx']} | "
        f"a={res['a_worst']:.10e} | b={res['b_worst']:.10e} | "
        f"tol(atol={res['atol']:.1e}, rtol={res['rtol']:.1e})"
    )


def tensor_max_mean_abs(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    d = (a - b).abs()
    return float(d.max().item()), float(d.mean().item())


def check_monotonic_descending(t: torch.Tensor) -> bool:
    if t.numel() <= 1:
        return True
    return bool(torch.all(t[:-1] >= t[1:]).item())


def make_timestep_list(num_train_timesteps: int, seed: int = 3407) -> List[int]:
    # 固定覆盖：头部、尾部、中段、你日志中的 worst_idx 附近、随机点
    base = {
        0, 1, 2, 3, 4, 5,
        10, 20, 50, 96, 100, 128, 200, 250, 300, 400, 500, 600, 700, 824, 900, 998, 999
    }
    gen = torch.Generator().manual_seed(seed)
    extra = torch.randint(0, num_train_timesteps, (16,), generator=gen).tolist()
    all_idx = sorted({i for i in list(base) + extra if 0 <= i < num_train_timesteps})
    return all_idx


# -----------------------------
# 核心检查项
# -----------------------------
def build_objects(args) -> Tuple[Any, DDIMScheduler]:
    # 与你 v8 训练/验证脚本保持一致的配置
    diffusion = IDDPM(str(args.num_train_timesteps))

    scheduler = DDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        clip_sample=False,
        prediction_type="v_prediction",
        set_alpha_to_one=False,
    )
    return diffusion, scheduler


def extract_buffers(diffusion, scheduler) -> Dict[str, torch.Tensor]:
    out = {}

    # IDDPM side
    out["d_betas"] = to_torch_1d(diffusion.betas)
    out["d_alphas_cumprod"] = to_torch_1d(diffusion.alphas_cumprod)
    out["d_sqrt_alphas_cumprod"] = to_torch_1d(diffusion.sqrt_alphas_cumprod)
    out["d_sqrt_one_minus_alphas_cumprod"] = to_torch_1d(diffusion.sqrt_one_minus_alphas_cumprod)

    # DDIMScheduler side
    out["s_betas"] = to_torch_1d(scheduler.betas)
    out["s_alphas_cumprod"] = to_torch_1d(scheduler.alphas_cumprod)
    out["s_sqrt_alphas_cumprod"] = torch.sqrt(out["s_alphas_cumprod"].clamp(min=0.0))
    out["s_sqrt_one_minus_alphas_cumprod"] = torch.sqrt((1.0 - out["s_alphas_cumprod"]).clamp(min=0.0))

    return out


def compare_schedules(buf: Dict[str, torch.Tensor], args) -> Dict[str, Any]:
    print("==== Schedule buffer comparison ====")

    results = {}

    # 分项阈值：betas 可以很严；cumprod / sqrt 允许浮点累计误差
    results["betas_strict"] = array_compare(
        "betas",
        buf["d_betas"], buf["s_betas"],
        atol=args.strict_betas_atol, rtol=args.strict_betas_rtol
    )
    print_array_result(results["betas_strict"])

    results["alphas_cumprod_relaxed"] = array_compare(
        "alphas_cumprod",
        buf["d_alphas_cumprod"], buf["s_alphas_cumprod"],
        atol=args.relaxed_cumprod_atol, rtol=args.relaxed_cumprod_rtol
    )
    print_array_result(results["alphas_cumprod_relaxed"])

    results["sqrt_alphas_cumprod_relaxed"] = array_compare(
        "sqrt_alphas_cumprod",
        buf["d_sqrt_alphas_cumprod"], buf["s_sqrt_alphas_cumprod"],
        atol=args.relaxed_cumprod_atol, rtol=args.relaxed_cumprod_rtol
    )
    print_array_result(results["sqrt_alphas_cumprod_relaxed"])

    results["sqrt_one_minus_alphas_cumprod_relaxed"] = array_compare(
        "sqrt_one_minus_alphas_cumprod",
        buf["d_sqrt_one_minus_alphas_cumprod"], buf["s_sqrt_one_minus_alphas_cumprod"],
        atol=args.relaxed_cumprod_atol, rtol=args.relaxed_cumprod_rtol
    )
    print_array_result(results["sqrt_one_minus_alphas_cumprod_relaxed"])

    # 额外硬失败阈值（防止“宽阈值掩盖真问题”）
    hard_fail = False
    hard_fail_reasons = []

    # betas 理论上应非常接近；如果这里偏差到 1e-4 量级，基本就是日程不一致
    betas_max = results["betas_strict"].get("max_abs", float("inf"))
    if not math.isfinite(betas_max) or betas_max > args.hard_fail_betas_max_abs:
        hard_fail = True
        hard_fail_reasons.append(
            f"betas max_abs={betas_max:.3e} > hard_fail({args.hard_fail_betas_max_abs:.1e})"
        )

    # cumprod 允许更大浮点误差，但如果达到 1e-4/1e-3 量级通常已不是数值抖动
    cumprod_max = results["alphas_cumprod_relaxed"].get("max_abs", float("inf"))
    if not math.isfinite(cumprod_max) or cumprod_max > args.hard_fail_cumprod_max_abs:
        hard_fail = True
        hard_fail_reasons.append(
            f"alphas_cumprod max_abs={cumprod_max:.3e} > hard_fail({args.hard_fail_cumprod_max_abs:.1e})"
        )

    relaxed_ok = all([
        results["betas_strict"].get("ok", False),
        results["alphas_cumprod_relaxed"].get("ok", False),
        results["sqrt_alphas_cumprod_relaxed"].get("ok", False),
        results["sqrt_one_minus_alphas_cumprod_relaxed"].get("ok", False),
    ])

    return {
        "results": results,
        "relaxed_ok": relaxed_ok,
        "hard_fail": hard_fail,
        "hard_fail_reasons": hard_fail_reasons,
    }


def functional_test_q_sample_vs_add_noise(diffusion, scheduler, args) -> Dict[str, Any]:
    print("==== Functional test: q_sample vs DDIMScheduler.add_noise ====")

    device = torch.device("cpu")
    dtype = torch.float32  # 与训练端实际更接近，也更有判别意义
    gen = torch.Generator(device=device).manual_seed(args.seed)

    # 多组 shape，覆盖单样本/批量
    shapes = [
        (1, 4, 64, 64),
        (2, 4, 64, 64),
        (1, 4, 32, 32),
    ]
    t_list = make_timestep_list(args.num_train_timesteps, seed=args.seed)

    per_case = []
    global_max_abs = 0.0
    global_mean_abs = 0.0
    total_cases = 0

    for shape in shapes:
        x0 = torch.randn(shape, generator=gen, device=device, dtype=dtype)
        noise = torch.randn(shape, generator=gen, device=device, dtype=dtype)

        for ti in t_list:
            bsz = shape[0]
            t = torch.full((bsz,), int(ti), device=device, dtype=torch.long)

            # 训练端
            z_iddpm = diffusion.q_sample(x0, t, noise=noise)

            # 验证端 (LQ-init 用到的 add_noise)
            z_ddim = scheduler.add_noise(x0, noise, t)

            max_abs, mean_abs = tensor_max_mean_abs(z_iddpm.float(), z_ddim.float())
            global_max_abs = max(global_max_abs, max_abs)
            global_mean_abs += mean_abs
            total_cases += 1

            per_case.append({
                "shape": shape,
                "t": int(ti),
                "max_abs": max_abs,
                "mean_abs": mean_abs,
            })

    global_mean_abs = global_mean_abs / max(1, total_cases)

    # 找 worst case
    worst = max(per_case, key=lambda x: x["max_abs"]) if per_case else None

    ok = (global_max_abs <= args.qsample_max_abs_tol)

    print(
        f"[QFUNC] cases={total_cases} | "
        f"global_max_abs={global_max_abs:.3e} | global_mean_abs={global_mean_abs:.3e} | "
        f"tol(max_abs<={args.qsample_max_abs_tol:.1e}) | ok={ok}"
    )
    if worst is not None:
        print(
            f"[QFUNC][WORST] shape={worst['shape']} t={worst['t']} "
            f"max_abs={worst['max_abs']:.3e} mean_abs={worst['mean_abs']:.3e}"
        )

    return {
        "ok": ok,
        "global_max_abs": global_max_abs,
        "global_mean_abs": global_mean_abs,
        "worst": worst,
        "cases": total_cases,
    }


def functional_test_vpred_coefficients(buf: Dict[str, torch.Tensor], args) -> Dict[str, Any]:
    print("==== Functional test: V-pred coefficients ====")

    # 系数定义（与你训练脚本一致）
    # training: alpha_t = sqrt_alphas_cumprod[t], sigma_t = sqrt_one_minus_alphas_cumprod[t]
    # scheduler side: same from scheduler.alphas_cumprod
    d_alpha = buf["d_sqrt_alphas_cumprod"]
    d_sigma = buf["d_sqrt_one_minus_alphas_cumprod"]
    s_alpha = buf["s_sqrt_alphas_cumprod"]
    s_sigma = buf["s_sqrt_one_minus_alphas_cumprod"]

    t_list = make_timestep_list(args.num_train_timesteps, seed=args.seed)

    alpha_diffs = []
    sigma_diffs = []
    worst_alpha = None
    worst_sigma = None

    for t in t_list:
        da = abs(float(d_alpha[t].item()) - float(s_alpha[t].item()))
        ds = abs(float(d_sigma[t].item()) - float(s_sigma[t].item()))
        alpha_diffs.append((t, da))
        sigma_diffs.append((t, ds))

    worst_alpha = max(alpha_diffs, key=lambda x: x[1]) if alpha_diffs else (-1, 0.0)
    worst_sigma = max(sigma_diffs, key=lambda x: x[1]) if sigma_diffs else (-1, 0.0)

    alpha_ok = worst_alpha[1] <= args.vcoef_max_abs_tol
    sigma_ok = worst_sigma[1] <= args.vcoef_max_abs_tol
    ok = alpha_ok and sigma_ok

    print(
        f"[VCOEF] alpha max_abs={worst_alpha[1]:.3e} @t={worst_alpha[0]} | "
        f"sigma max_abs={worst_sigma[1]:.3e} @t={worst_sigma[0]} | "
        f"tol(max_abs<={args.vcoef_max_abs_tol:.1e}) | ok={ok}"
    )

    return {
        "ok": ok,
        "alpha_worst_t": int(worst_alpha[0]),
        "alpha_max_abs": float(worst_alpha[1]),
        "sigma_worst_t": int(worst_sigma[0]),
        "sigma_max_abs": float(worst_sigma[1]),
    }


def inference_timestep_sanity(scheduler: DDIMScheduler, args) -> Dict[str, Any]:
    print("==== Inference timestep sanity (DDIMScheduler.set_timesteps) ====")
    device = torch.device("cpu")
    scheduler.set_timesteps(args.inference_steps, device=device)
    ts = scheduler.timesteps.detach().cpu().long()

    desc_ok = check_monotonic_descending(ts)
    in_range_ok = bool(torch.all((ts >= 0) & (ts < args.num_train_timesteps)).item())
    unique_ok = (ts.unique().numel() == ts.numel())

    head = ts[: min(10, ts.numel())].tolist()
    tail = ts[-min(10, ts.numel()):].tolist()

    print(
        f"[TIMESTEPS] n={ts.numel()} | descending={desc_ok} | in_range={in_range_ok} | unique={unique_ok}"
    )
    print(f"[TIMESTEPS] head={head}")
    print(f"[TIMESTEPS] tail={tail}")

    # 对应你 v8 的 LQ-init 索引逻辑
    strength = float(max(0.0, min(1.0, args.lq_init_strength)))
    start_index = int(round(strength * (len(ts) - 1)))
    start_index = min(max(start_index, 0), len(ts) - 1)
    t_start = int(ts[start_index].item())

    print(
        f"[LQ_INIT_INDEX] strength={strength:.4f} -> start_index={start_index}/{len(ts)-1} -> t_start={t_start}"
    )

    ok = desc_ok and in_range_ok and unique_ok
    return {
        "ok": ok,
        "n": int(ts.numel()),
        "descending": bool(desc_ok),
        "in_range": bool(in_range_ok),
        "unique": bool(unique_ok),
        "head": head,
        "tail": tail,
        "lq_init_start_index": int(start_index),
        "lq_init_t_start": int(t_start),
    }


# -----------------------------
# 最终判决逻辑
# -----------------------------
def final_judgment(schedule_res, qfunc_res, vcoef_res, tstep_res, args) -> Dict[str, Any]:
    reasons_fail = []
    reasons_warn = []
    reasons_ok = []

    # 1) 硬失败优先
    if schedule_res["hard_fail"]:
        reasons_fail.extend(schedule_res["hard_fail_reasons"])

    # 2) 功能级检查（最关键）
    if not qfunc_res["ok"]:
        reasons_fail.append(
            f"q_sample vs add_noise mismatch too large: max_abs={qfunc_res['global_max_abs']:.3e} > {args.qsample_max_abs_tol:.1e}"
        )
    else:
        reasons_ok.append(
            f"q_sample vs add_noise functional consistency passed (max_abs={qfunc_res['global_max_abs']:.3e})"
        )

    if not vcoef_res["ok"]:
        reasons_fail.append(
            f"V-pred coefficients mismatch too large: alpha={vcoef_res['alpha_max_abs']:.3e}, sigma={vcoef_res['sigma_max_abs']:.3e}"
        )
    else:
        reasons_ok.append(
            f"V-pred coefficients consistency passed (alpha={vcoef_res['alpha_max_abs']:.3e}, sigma={vcoef_res['sigma_max_abs']:.3e})"
        )

    if not tstep_res["ok"]:
        reasons_fail.append("DDIM inference timesteps sanity check failed (descending/range/unique)")
    else:
        reasons_ok.append("DDIM inference timesteps sanity passed")

    # 3) schedule 数组级比较：非硬失败但 relaxed 不过 -> 视为警告/失败（这里给出保守策略）
    if not schedule_res["relaxed_ok"]:
        # 如果功能级都过了，但数组级 relaxed 有问题，通常是实现细节/版本差异；先作为 WARN
        reasons_warn.append("Schedule arrays not fully allclose under relaxed tolerance, but functional checks may still pass")
    else:
        reasons_ok.append("Schedule buffers match under relaxed tolerance")

    # 判决
    if len(reasons_fail) > 0:
        judgment = "BLOCK_TRAINING"
        can_train = False
    else:
        # 功能级过了就给 OK_TO_TRAIN；有 warns 也不拦截，但会明确提醒
        judgment = "OK_TO_TRAIN"
        can_train = True

    return {
        "judgment": judgment,
        "can_train": can_train,
        "reasons_fail": reasons_fail,
        "reasons_warn": reasons_warn,
        "reasons_ok": reasons_ok,
    }


def print_final_report(report: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(f"FINAL_JUDGMENT: {report['judgment']}")
    print(f"CAN_TRAIN: {report['can_train']}")

    if report["reasons_ok"]:
        print("\n[OK]")
        for r in report["reasons_ok"]:
            print(f"- {r}")

    if report["reasons_warn"]:
        print("\n[WARN]")
        for r in report["reasons_warn"]:
            print(f"- {r}")

    if report["reasons_fail"]:
        print("\n[FAIL]")
        for r in report["reasons_fail"]:
            print(f"- {r}")

    print("=" * 80)
    if report["can_train"]:
        print("建议：可以开始训练 / 继续训练。")
        print("前提：你当前训练脚本与验证脚本保持与本次测试相同的 schedule / prediction_type / timestep 语义。")
    else:
        print("建议：先修改扩散配置，再训练。")
        print("优先检查：beta schedule、prediction_type、num_train_timesteps、timestep 索引方向。")
    print("=" * 80)


# -----------------------------
# 主程序
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Judge consistency between PixArt IDDPM and DDIMScheduler (for your v8 training/validation setup)")
    # 与你 v8 脚本保持一致的默认值
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)
    p.add_argument("--beta_schedule", type=str, default="linear")
    p.add_argument("--inference_steps", type=int, default=50)
    p.add_argument("--lq_init_strength", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=3407)

    # 阈值（可调，但默认针对你当前情况已经足够）
    p.add_argument("--strict_betas_atol", type=float, default=1e-8)
    p.add_argument("--strict_betas_rtol", type=float, default=1e-8)

    p.add_argument("--relaxed_cumprod_atol", type=float, default=1e-6)
    p.add_argument("--relaxed_cumprod_rtol", type=float, default=1e-6)

    p.add_argument("--qsample_max_abs_tol", type=float, default=1e-6)
    p.add_argument("--vcoef_max_abs_tol", type=float, default=1e-6)

    # 硬失败阈值（防止“放宽阈值后误判”）
    p.add_argument("--hard_fail_betas_max_abs", type=float, default=1e-4)
    p.add_argument("--hard_fail_cumprod_max_abs", type=float, default=1e-4)

    return p.parse_args()


def main():
    args = parse_args()

    print("==== Build objects ====")
    try:
        diffusion, scheduler = build_objects(args)
    except Exception as e:
        print(f"[FATAL] Failed to build objects: {e}")
        print("\nFINAL_JUDGMENT: BLOCK_TRAINING")
        sys.exit(1)

    print("==== Extract buffers ====")
    try:
        buf = extract_buffers(diffusion, scheduler)
    except Exception as e:
        print(f"[FATAL] Failed to extract buffers: {e}")
        print("\nFINAL_JUDGMENT: BLOCK_TRAINING")
        sys.exit(1)

    # 基础信息输出（便于你核对是不是同一配置）
    print(
        f"[CONFIG] num_train_timesteps={args.num_train_timesteps}, "
        f"beta_schedule={args.beta_schedule}, beta_start={args.beta_start}, beta_end={args.beta_end}"
    )
    print(
        f"[CONFIG] scheduler.prediction_type={scheduler.config.prediction_type}, "
        f"set_alpha_to_one={scheduler.config.set_alpha_to_one}, clip_sample={scheduler.config.clip_sample}"
    )

    # 1) 数组级 schedule 检查
    schedule_res = compare_schedules(buf, args)

    # 2) 功能级 q_sample / add_noise 检查（关键）
    try:
        qfunc_res = functional_test_q_sample_vs_add_noise(diffusion, scheduler, args)
    except Exception as e:
        print(f"[QFUNC][EXCEPTION] {e}")
        qfunc_res = {
            "ok": False,
            "global_max_abs": float("inf"),
            "global_mean_abs": float("inf"),
            "worst": None,
            "cases": 0,
        }

    # 3) V-pred 系数检查（关键）
    try:
        vcoef_res = functional_test_vpred_coefficients(buf, args)
    except Exception as e:
        print(f"[VCOEF][EXCEPTION] {e}")
        vcoef_res = {
            "ok": False,
            "alpha_worst_t": -1,
            "alpha_max_abs": float("inf"),
            "sigma_worst_t": -1,
            "sigma_max_abs": float("inf"),
        }

    # 4) DDIM 推理 timesteps 语义 sanity（辅助，避免索引方向问题）
    try:
        tstep_res = inference_timestep_sanity(scheduler, args)
    except Exception as e:
        print(f"[TIMESTEPS][EXCEPTION] {e}")
        tstep_res = {
            "ok": False,
            "n": 0,
            "descending": False,
            "in_range": False,
            "unique": False,
            "head": [],
            "tail": [],
            "lq_init_start_index": -1,
            "lq_init_t_start": -1,
        }

    # 最终判决
    report = final_judgment(schedule_res, qfunc_res, vcoef_res, tstep_res, args)
    print_final_report(report)

    # Exit code：0 表示可训练；1 表示阻塞
    sys.exit(0 if report["can_train"] else 1)


if __name__ == "__main__":
    main()