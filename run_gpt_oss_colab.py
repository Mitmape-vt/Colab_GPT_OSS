#!/usr/bin/env python3
"""
gpt-oss-20b Colab Runner (L4-safe)
----------------------------------
One-file launcher that:
  1) Installs deps (transformers from git, Triton kernels, bitsandbytes, etc.)
  2) Downloads & loads openai/gpt-oss-20b from Hugging Face
  3) Offers CLI or Gradio chat UI
Defaults to 4-bit loading on <40 GB VRAM GPUs (e.g., L4) with CPU offload.

Usage (Colab):
!python run_gpt_oss_colab.py --ui
Or CLI:
!python run_gpt_oss_colab.py --prompt "Write a haiku about rockets"
"""
import argparse
import os
import sys
import subprocess
from typing import List, Dict, Any

# --- Safer alloc & cleaner logs ---
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # optional speedup

def _pip(args: List[str]):
    print("[setup] pip", " ".join(args), flush=True)
    subprocess.check_call([sys.executable, "-m", "pip"] + args)

def ensure_environment(no_install: bool = False):
    """
    Install specific deps needed to run gpt-oss-20b comfortably on Colab.
    """
    if no_install:
        print("[setup] Skipping dependency installation (as requested).")
        return

    # Keep torch reasonably current. Remove torchvision/torchaudio to avoid version pin conflicts.
    _pip(["install", "-q", "--upgrade", "torch"])
    _pip(["uninstall", "-y", "torchvision", "torchaudio"])

    # transformers from source (for latest quant & template behavior)
    _pip(["install", "-q", "git+https://github.com/huggingface/transformers"])

    # Triton + kernels (for MXFP4 path if user opts in)
    _pip(["install", "-q", "triton==3.4"])
    _pip(["install", "-q", "git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels"])

    # BitsAndBytes for 4-bit loading (our default on L4)
    _pip(["install", "-q", "--upgrade", "bitsandbytes"])

    # Common helpers
    _pip(["install", "-q", "accelerate", "huggingface_hub>=0.24.0", "sentencepiece", "tqdm", "gradio>=4.42.0"])

def detect_gpu() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return f"CUDA GPU: {name}, compute {cap}, VRAM ~{mem_gb:.1f} GB"
        else:
            return "No CUDA GPU detected."
    except Exception as e:
        return f"GPU check error: {e}"

def _gpu_total_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        pass
    return 0.0

def _max_memory_map(gpu_fraction: float = 0.90):
    """
    Constrain how much VRAM we allow the model to use; overflow will offload to CPU.
    """
    vram = _gpu_total_gb()
    gpu_budget_gib = max(1, int(vram * gpu_fraction))  # round down to GiB
    mm = {"cpu": "48GiB"}  # Colab usually has plenty of system RAM
    try:
        import torch
        if torch.cuda.is_available():
            mm[0] = f"{gpu_budget_gib}GiB"
    except Exception:
        pass
    return mm

def _load_bnb4(model_id: str):
    """
    4-bit (bitsandbytes) with auto device map + CPU offload for overflow.
    This is the most reliable configuration for L4.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print("[load] Strategy: bitsandbytes 4-bit (nf4 + double quant), device_map='auto'")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        max_memory=_max_memory_map(0.90),
    )
    # Pad token helps avoid warnings during generation
    if model.generation_config.pad_token_id is None and tok.eos_token_id is not None:
        model.generation_config.pad_token_id = tok.eos_token_id
    return tok, model

def _load_mxfp4(model_id: str):
    """
    MXFP4 path (requires Triton kernels to truly stay quantized). May still fall back to bf16.
    We still constrain VRAM and allow CPU offload.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print("[load] Strategy: MXFP4 / auto dtype with CPU offload safety")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        max_memory=_max_memory_map(0.90),
    )
    if model.generation_config.pad_token_id is None and tok.eos_token_id is not None:
        model.generation_config.pad_token_id = tok.eos_token_id
    return tok, model

def _load_bf16_offload(model_id: str):
    """
    Pure bf16 with aggressive CPU offload (slow but safe, even on CPU-only).
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print("[load] Strategy: bf16 with heavy CPU offload")
    tok = AutoTokenizer.from_pretrained(model_id)
    device_map = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device_map = "auto"
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        max_memory=_max_memory_map(0.85),
    )
    if model.generation_config.pad_token_id is None and tok.eos_token_id is not None:
        model.generation_config.pad_token_id = tok.eos_token_id
    return tok, model

def load_model(model_id: str = "openai/gpt-oss-20b", strategy: str = "auto"):
    """
    Load with a strategy:
      - auto  : bnb4 if GPU < 40GB, else try MXFP4; fall back to bnb4, then bf16_offload
      - bnb4  : force bitsandbytes 4-bit
      - mxfp4 : try MXFP4 (may still dequantize if kernels unavailable)
      - bf16  : force bf16 with offload
    """
    import torch
    has_gpu = torch.cuda.is_available()
    vram = _gpu_total_gb()

    if strategy == "auto":
        if has_gpu and vram < 40:
            strategy = "bnb4"
        else:
            strategy = "mxfp4"

    print(f"[load] Requested strategy: {strategy} | GPU: {'yes' if has_gpu else 'no'} | VRAM ~{vram:.1f} GB")

    try:
        if strategy == "bnb4":
            return _load_bnb4(model_id)
        elif strategy == "mxfp4":
            try:
                return _load_mxfp4(model_id)
            except RuntimeError as e:
                # Likely OOM due to dequantization fallback; switch to bnb4
                print(f"[load][warn] MXFP4 path failed ({e}). Falling back to 4-bit.")
                return _load_bnb4(model_id)
        elif strategy == "bf16":
            return _load_bf16_offload(model_id)
        else:
            # Unknown -> auto fallback
            return _load_bnb4(model_id) if has_gpu else _load_bf16_offload(model_id)
    except RuntimeError as e:
        # Generic last-resort fallback
        print(f"[load][warn] Strategy '{strategy}' failed with RuntimeError: {e}\n[load] Falling back to bf16 offload.")
        return _load_bf16_offload(model_id)

def apply_chat_template(tokenizer, messages, device):
    # Use the chat template shipped with the model repo.
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

def generate_once(tokenizer, model, messages: List[Dict[str, Any]], max_new_tokens=512, temperature=0.7, top_p=0.95):
    import torch
    inputs = apply_chat_template(tokenizer, messages, model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=temperature > 0,
            temperature=float(temperature),
            top_p=float(top_p),
            use_cache=True,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return text

def run_cli(args):
    tok, model = load_model(args.model, strategy=args.strategy)
    sys_prompt = args.system.strip() if args.system is not None else "You are a helpful assistant."
    messages = [{"role": "system", "content": sys_prompt}]
    if args.prompt:
        user_msg = {"role": "user", "content": args.prompt}
        if args.effort and args.effort.lower() != "none":
            user_msg["reasoning_effort"] = args.effort.lower()
        messages.append(user_msg)
        text = generate_once(tok, model, messages, args.max_new_tokens, args.temperature, args.top_p)
        print("\n[assistant]\n" + text)
    else:
        print("[interactive] Enter blank line to quit.")
        while True:
            try:
                u = input("\n[user] > ").strip()
            except EOFError:
                break
            if not u:
                break
            user_msg = {"role": "user", "content": u}
            if args.effort and args.effort.lower() != "none":
                user_msg["reasoning_effort"] = args.effort.lower()
            messages.append(user_msg)
            text = generate_once(tok, model, messages, args.max_new_tokens, args.temperature, args.top_p)
            print("\n[assistant]\n" + text)
            messages.append({"role": "assistant", "content": text})

def run_ui(args):
    tok, model = load_model(args.model, strategy=args.strategy)

    import gradio as gr
    import torch
    from transformers import TextIteratorStreamer

    def infer(message, history, system_prompt, effort, max_new_tokens, temperature, top_p):
        # Build full chat history
        msgs = [{"role": "system", "content": system_prompt or "You are a helpful assistant."}]
        for (u, a) in history:
            if u:
                msgs.append({"role": "user", "content": u})
            if a:
                msgs.append({"role": "assistant", "content": a})
        user_msg = {"role": "user", "content": message}
        if effort and effort != "none":
            user_msg["reasoning_effort"] = effort
        msgs.append(user_msg)

        inputs = apply_chat_template(tok, msgs, model.device)

        streamer = TextIteratorStreamer(tok, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=temperature > 0,
            temperature=float(temperature),
            top_p=float(top_p),
            streamer=streamer,
            use_cache=True,
        )

        # Non-blocking generation
        import threading
        def _gen():
            with torch.no_grad():
                model.generate(**gen_kwargs)

        thread = threading.Thread(target=_gen)
        thread.start()

        # Stream tokens
        partial = ""
        for new_text in streamer:
            partial += new_text
            yield partial

    with gr.Blocks(title="gpt-oss-20b (local/Colab)") as demo:
        gr.Markdown("## gpt-oss-20b — Local Chat (Colab)\nStart chatting below.")

        with gr.Row():
            sys_box = gr.Textbox(value="You are a helpful assistant.", label="System Prompt")
            effort = gr.Dropdown(choices=["none", "low", "medium", "high"], value="medium", label="Reasoning Effort")
        with gr.Row():
            max_new = gr.Slider(16, 2048, value=512, step=16, label="Max new tokens")
            temp = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")

        chat = gr.ChatInterface(
            fn=lambda msg, hist: infer(msg, hist, sys_box.value, effort.value, max_new.value, temp.value, top_p.value),
            type="generator",
            title="gpt-oss-20b Chat",
            multimodal=False,
        )

        gr.Markdown(f"**Hardware:** {detect_gpu()}")

    demo.queue().launch(share=False)

def main():
    p = argparse.ArgumentParser(description="Run OpenAI gpt-oss-20b locally (Colab-friendly).")
    p.add_argument("--model", default="openai/gpt-oss-20b", help="HF repo id to load.")
    p.add_argument("--strategy", default="auto", choices=["auto", "bnb4", "mxfp4", "bf16"],
                   help="Loading strategy. Default 'auto' (bnb4 on <40GB GPUs, otherwise try mxfp4).")
    p.add_argument("--ui", action="store_true", help="Start a Gradio chat UI.")
    p.add_argument("--no-install", action="store_true", help="Skip pip installs (advanced).")
    p.add_argument("--prompt", type=str, default=None, help="Single-turn prompt for CLI mode.")
    p.add_argument("--system", type=str, default=None, help="System prompt for CLI/UI.")
    p.add_argument("--effort", type=str, default="medium", choices=["none", "low", "medium", "high"], help="Reasoning effort.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)

    args = p.parse_args()

    print("[env] Ensuring dependencies...")
    ensure_environment(no_install=args.no_install)

    print("[env] ", detect_gpu())
    if args.ui:
        run_ui(args)
    else:
        run_cli(args)

if __name__ == "__main__":
    main()
