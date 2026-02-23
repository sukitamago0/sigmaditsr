# -*- coding: utf-8 -*-
"""
离线提取 PixArt-Sigma 用的“空提示词 T5 embedding”（null caption embedding）
用途：训练时不再加载 T5EncoderModel，直接读取本文件输出的 .pth 即可。

输出内容（默认）：
- y: [1, 1, L, C]   （与你训练脚本中的 y 形状一致）
- hidden: [1, L, C] （原始 T5 输出）
- attention_mask: [1, L]
- meta: 一些元信息（max_length / prompt / dtype / path 等）

示例：
python tools/extract_sigma_null_t5_embedding.py \
  --tokenizer_path /home/hello/HJT/PixArt-sigma/output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers/tokenizer \
  --text_encoder_path /home/hello/HJT/PixArt-sigma/output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers/text_encoder \
  --max_length 300 \
  --prompt "" \
  --device cpu \
  --save_path /home/hello/HJT/PixArt-sigma/output/pretrained_models/null_t5_embed_sigma_300.pth
"""

import os
import argparse
import torch
from transformers import T5Tokenizer, T5EncoderModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True, help="本地 tokenizer 路径")
    parser.add_argument("--text_encoder_path", type=str, required=True, help="本地 T5EncoderModel 路径")
    parser.add_argument("--save_path", type=str, required=True, help="输出 .pth 路径")
    parser.add_argument("--prompt", type=str, default="", help="默认空字符串")
    parser.add_argument("--max_length", type=int, default=300, help="PixArt-Sigma 通常为 300")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="建议用 cpu，一次性离线提取即可")
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="保存张量的数据类型；建议 float32，训练时再转"
    )
    parser.add_argument("--legacy_tokenizer", action="store_true", help="显式使用 T5Tokenizer legacy=True")
    return parser.parse_args()


def str_to_dtype(name: str):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(name)


@torch.no_grad()
def main():
    args = parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    print(f"[1/4] Loading tokenizer from: {args.tokenizer_path}")
    if args.legacy_tokenizer:
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path, local_files_only=True, legacy=True)
    else:
        # 不传 legacy，保持你当前环境的默认行为（和训练时警告对应）
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)

    print(f"[2/4] Loading T5 encoder from: {args.text_encoder_path} on {args.device}")
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_path, local_files_only=True).eval()
    text_encoder = text_encoder.to(args.device)

    print(f"[3/4] Encoding prompt={repr(args.prompt)} with max_length={args.max_length}")
    tokens = tokenizer(
        args.prompt,
        max_length=args.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(args.device)
    attention_mask = tokens.attention_mask.to(args.device)

    hidden = text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]  # [1, L, C]
    y = hidden.unsqueeze(1)  # [1, 1, L, C]，与你训练脚本一致

    save_dtype = str_to_dtype(args.save_dtype)
    pack = {
        "y": y.detach().cpu().to(save_dtype),
        "hidden": hidden.detach().cpu().to(save_dtype),
        "attention_mask": attention_mask.detach().cpu(),
        "meta": {
            "prompt": args.prompt,
            "max_length": int(args.max_length),
            "tokenizer_path": args.tokenizer_path,
            "text_encoder_path": args.text_encoder_path,
            "save_dtype": args.save_dtype,
            "hidden_shape": tuple(hidden.shape),
            "y_shape": tuple(y.shape),
        },
    }

    torch.save(pack, args.save_path)

    print("[4/4] Saved.")
    print(f"  save_path: {args.save_path}")
    print(f"  hidden shape: {tuple(hidden.shape)}")
    print(f"  y shape: {tuple(y.shape)}")
    print(f"  attention_mask shape: {tuple(attention_mask.shape)}")


if __name__ == "__main__":
    main()