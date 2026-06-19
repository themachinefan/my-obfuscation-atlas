"""Merge a trained LoRA adapter into its base model for serving / inference.

Why this exists: vLLM's ``serve --lora-modules`` does NOT apply LoRA to
Qwen3.5's hybrid linear-attention (mamba) modules (``in_proj_*``, ``out_proj``).
It loads the adapter name but silently serves the *base* weights, so a
trained adapter looks identical to base (verified: identical logprobs). PEFT
applies all modules correctly, so merging the adapter into the base produces a
standalone full model that vLLM (and any HF loader) serves correctly.

Usage:
    python -m obfuscation_atlas.scripts.merge_lora_adapter \
        --adapter themachinefan/rl_Qwen3.5-4B_lol2on0t --out /workspace/merged
    # then serve the result normally (no --lora-modules needed):
    vllm serve /workspace/merged --served-model-name trained --port 8000
"""

import argparse
import json
import os

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, help="HF repo id or local path of the LoRA adapter")
    ap.add_argument("--base", default=None, help="Base model id (default: read from adapter_config.json)")
    ap.add_argument("--out", required=True, help="Output directory for the merged model")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="sdpa", help="attn_implementation (sdpa avoids needing flash-attn)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    base = args.base
    if base is None:
        cfg_path = (
            os.path.join(args.adapter, "adapter_config.json")
            if os.path.isdir(args.adapter)
            else hf_hub_download(args.adapter, "adapter_config.json")
        )
        base = json.load(open(cfg_path))["base_model_name_or_path"]

    print(f"merging: base={base} + adapter={args.adapter} -> {args.out}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=getattr(torch, args.dtype), device_map=args.device, attn_implementation=args.attn
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(args.out)
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
