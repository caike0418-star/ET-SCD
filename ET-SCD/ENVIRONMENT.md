# Environments

ET-SCD requires Python 3.9+, NumPy 1.23.5+, and Pillow 9+. Install it in the environment of
the selected backbone:

```bash
python -m pip install -e .
```

The `etscd-prepare-weak-images` command uses NumPy and Pillow on the CPU to
prepare frequency low-pass, Gaussian-blurred, or edge-overlay images. See the
[POPE example](README.md#pope) for image preparation and inference commands.

## Backbone setup

Use a separate environment for each backbone and install the framework versions
required by its checkpoint.

| Adapter | Dependencies and local model files |
|---|---|
| `llava` | PyTorch, the LLaVA-1.5 package, and a complete LLaVA-1.5-7B checkpoint |
| `qwen` | PyTorch, checkpoint-compatible Transformers, tiktoken, transformers-stream-generator, and Qwen-VL-Chat checkpoint files including native Python code and generation configuration |
| `instructblip` | PyTorch, LAVIS, and the configured `blip2_vicuna_instruct` / `vicuna7b` model with its vision and Vicuna weights |

Choose a PyTorch/CUDA build compatible with the GPU. Pass the local LLaVA or Qwen
checkpoint directory through `--model-path`. For InstructBLIP, `--model-path`
selects a LAVIS checkpoint file. Pass its local Vicuna language-model directory
with `--llm-model VICUNA`; the installed LAVIS YAML supplies the architecture.

```bash
python scripts/smoke_test.py --family instructblip --model-path INSTRUCTBLIP.pth --llm-model VICUNA --image IMAGE.jpg --output outputs/smoke_instructblip.json
```

The Qwen adapter uses the checkpoint's native ChatML context builder. Each adapter
returns raw full-vocabulary next-token logits and records its resolved model,
tokenizer, EOS IDs, and package versions in the run manifest.

## CHAIR and BLEU

Install the evaluation dependencies:

```bash
python -m pip install -r requirements-eval.txt
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng wordnet omw-1.4
```

Java must be available for pycocoevalcap's PTBTokenizer. Place the COCO2014 caption
and instance JSON annotations in the directory passed to `--annotation-dir`.

## Tests

```bash
python -m unittest discover -s tests -v
```
