#!/usr/bin/env python3

"""
gpt-oss-20b Colab Runner
------------------------
A one-file launcher that:
  1) Installs the right deps (including bleeding-edge transformers & MXFP4 kernels)
  2) Downloads & loads openai/gpt-oss-20b from Hugging Face
  3) Offers a simple CLI or Gradio chat UI

Tested on Google Colab (T4 / L4 / A100). If no GPU is present, it will still run,
but it will be *very* slow.

Usage (Colab cell):
!python run_gpt_oss_colab.py --ui

Or CLI:
!python run_gpt_oss_colab.py --prompt "Write a haiku about rockets"
"""
import argparse
import os
import sys
import subprocess
import time
from typing import List, Dict, Any, Optional

def _pip(args: List[str]):
    print("[setup] pip", " ".join(args), flush=True)
    subprocess.check_call([sys.executable, "-m", "pip"] + args)

def ensure_environment(no_install: bool = False):
    """
    Install the specific deps needed to run gpt-oss-20b with MXFP4 kernels.
    Mirrors the official OpenAI Cookbook guidance.
    """
    if no_install:
        print("[setup] Skipping dependency installation (as requested).")
        return

    # Recent torch; uninstall torchvision/torchaudio to avoid conflicts in Colab.
    _pip(["install", "-q", "--upgrade", "torch"])
    # transformers from source (for mxfp4 support), triton kernels
    _pip(["install", "-q", "git+https://github.com/huggingface/transformers"])
    _pip(["install", "-q", "triton==3.4"])
    _pip(["install", "-q", "git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels"])
    _pip(["uninstall", "-y", "torchvision", "torchaudio"])

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

def load_model(model_id: str = "openai/gpt-oss-20b"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[load] Loading model: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    # MXFP4 quantization support is integrated; just set dtype=auto & device_map=cuda if available.
    device_map = "cuda" if torch.cuda.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=device_map,
    )
    return tok, model

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
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return text

def run_cli(args):
    tok, model = load_model(args.model)
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
    tok, model = load_model(args.model)

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
