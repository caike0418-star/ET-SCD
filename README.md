# ET-SCD: Evidence-Aware Three-Way Semantic Contrastive Decoding

Official project repository for **Evidence-Aware Three-Way Semantic Contrastive Decoding for Object Hallucination Mitigation**, a training-free decoding framework for mitigating object hallucinations in large vision-language models (LVLMs).

## Overview

ET-SCD combines three components:

- **Object-sensitive selective activation:** Identifies generation steps involving potential object hallucinations.
- **Evidence-aware three-way decision-making:** Uses contrastive evidence and uncertainty estimates to route candidate predictions to Accept, Defer, or Reject.
- **Target-aware visual re-query:** Acquires additional visual evidence for ambiguous predictions and refines decisions through evidence fusion.

## Code availability and validation status

The implementation is publicly available in [`ET-SCD/`](ET-SCD/), including Python source code, configuration files, backbone adapters, evaluation utilities, and tests. The release provides interfaces for LLaVA-1.5, Qwen-VL-Chat, and InstructBLIP, with evaluation workflows for POPE, MME-Hall, and CHAIR.

**Code availability does not establish successful experimental reproduction.** This README does not certify that all unit tests pass or that the complete experiments have been validated across all three backbones. Reproducing manuscript results requires the corresponding checkpoints, datasets, dependencies, and evaluation settings. The quick-start command below is a usage example, not a report of a successful run.

## Installation

Python **3.9 or later** is required. Set up the dependencies for your selected backbone using the [environment guide](ET-SCD/ENVIRONMENT.md), then install this package in the same environment:

```bash
git clone https://github.com/caike0418-star/ET-SCD.git
cd ET-SCD/ET-SCD
python -m pip install -e .
```

The repository contains an inner `ET-SCD/` package directory. Run the installation and commands below from that directory, which contains [`pyproject.toml`](ET-SCD/pyproject.toml). The editable installation alone does not install the backbone frameworks or download model checkpoints and benchmark data.

## Quick start

After setting up LLaVA-1.5 and a local checkpoint, run short generations on a local image:

```bash
python scripts/smoke_test.py \
  --family llava \
  --model-path /path/to/llava-v1.5-7b \
  --image /path/to/image.jpg \
  --output outputs/smoke_llava.json
```

Replace the example paths with your local files and choose a fresh output path. This script runs short POPE-style, MME-Hall-style, and captioning generations; it is not a full benchmark evaluation. See the [implementation README](ET-SCD/README.md) for Qwen-VL-Chat and InstructBLIP usage, data preparation, and complete evaluation commands.

## Documentation

The [implementation README](ET-SCD/README.md) is the primary guide for installation and execution. Detailed commands and settings are maintained there to avoid duplication.

| Topic | Documentation |
|---|---|
| Environment and backbone dependencies | [Environment guide](ET-SCD/ENVIRONMENT.md) |
| Package requirements and entry points | [Installation configuration](ET-SCD/pyproject.toml) |
| Data formats and preparation | [Data guide](ET-SCD/data/README.md) |
| POPE inference, scoring, and five-seed aggregation | [POPE evaluation](ET-SCD/README.md#pope) |
| MME-Hall inference and scoring | [MME-Hall evaluation](ET-SCD/README.md#mme-hall) |
| CHAIR and BLEU evaluation | [CHAIR evaluation](ET-SCD/README.md#chair) |
| Algorithm mapping, configurations, and logging | [Implementation details](ET-SCD/docs/IMPLEMENTATION.md) |
| Test suite and execution instructions | [Tests](ET-SCD/tests/) and [test instructions](ET-SCD/ENVIRONMENT.md#tests) |
| Citation metadata | [CITATION.cff](ET-SCD/CITATION.cff) |

## Tests

From the inner `ET-SCD/` directory, run:

```bash
python -m unittest discover -s tests -v
```

The test suite and model smoke tests serve different purposes; neither alone establishes reproduction of the full benchmark results. See the [environment guide](ET-SCD/ENVIRONMENT.md) for dependencies.

## Citation and license

Please cite the accompanying ET-SCD manuscript when using this implementation. Author and title metadata are available in [CITATION.cff](ET-SCD/CITATION.cff).

The implementation is licensed under [Apache-2.0](ET-SCD/LICENSE). CHAIR metric and object-vocabulary credits are provided in the [metrics documentation](ET-SCD/src/etscd/metrics/README.md).
