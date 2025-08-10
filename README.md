# gpt-oss-20b Colab Runner

A minimal, one-file way to run **OpenAI's gpt-oss-20b** locally on **Google Colab** with a simple web UI (Gradio) or via CLI.

> This mirrors the official OpenAI Cookbook steps for Colab (MXFP4 kernels) but wraps them into a single script you can run.

## Quick Start (Colab)

1. Open a new Colab notebook with GPU enabled (`Runtime -> Change runtime type -> T4/L4/A100 preferred`).
2. Run:

```bash
!git clone https://github.com/you/gpt-oss-colab-runner
%cd gpt-oss-colab-runner
!python run_gpt_oss_colab.py --ui
```

This installs required deps, downloads the model from Hugging Face, and launches a local Gradio chat. If prompted about Hugging Face auth, it's optional for public models.

## CLI Mode

```bash
!python run_gpt_oss_colab.py --prompt "Write a limerick about model rockets" --effort high
```

## Notes

- The script loads `openai/gpt-oss-20b` using `torch_dtype='auto'` and `device_map='cuda'` when a GPU is present.
- It uses the model’s chat template and supports the `reasoning_effort` message field (`none/low/medium/high`).
- If you hit dependency conflicts in Colab, the script *removes* `torchvision` and `torchaudio` and installs transformers from source plus the MXFP4 triton kernels.

## Troubleshooting

- **No GPU / small VRAM**: It will still run, but will be slow. Prefer Colab with T4/L4/A100.
- **HF token warning**: You can ignore if the repo is public. Or set `HF_TOKEN` as a Colab secret.
- **OutOfMemoryError**: Reduce `--max-new-tokens`, or restart runtime and ensure no other large models are loaded.

## License

Code here is MIT. The `gpt-oss-20b` model weights are under Apache 2.0 (see the model card on Hugging Face).
