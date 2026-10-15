# ET-SCD

Code for **Evidence-Aware Three-Way Semantic Contrastive Decoding for Object
Hallucination Mitigation**.

ET-SCD combines semantic contrastive evidence, interval-based Accept/Defer/Reject
routing, boundary-guided visual re-query, and adaptive evidence fusion. The same
controller supports POPE, MME-Hall, and CHAIR with LLaVA-1.5, Qwen-VL-Chat, and
InstructBLIP.

## Installation

Use Python 3.9 or later. Install the selected backbone and this package in the
same environment:

```bash
python -m pip install -e .
```

See [ENVIRONMENT.md](ENVIRONMENT.md) for backbone and evaluation dependencies.

Load a checkpoint and run short generations with:

```bash
python scripts/smoke_test.py --family llava --model-path CHECKPOINT --image IMAGE.jpg --output outputs/smoke_llava.json
```

For InstructBLIP, pass the local Vicuna directory using `--llm-model VICUNA`.

## Data preparation

Place benchmark images and model checkpoints in local directories. Questions use
JSONL with the following fields:

```json
{"id":"example_1","image":"example.jpg","question":"Is there a dog in the image? Please answer yes or no."}
```

Scoring uses a separate label file:

```json
{"id":"example_1","label":"yes"}
```

MME-Hall labels also contain `category` and `image_id`. The categories are
`existence`, `count`, `position`, and `color`, with two questions per category/image
pair. Keep the complete question, including count, color, and position conditions.

Convert POPE records into the question and label formats:

```bash
python -m etscd.data --benchmark pope --input raw_pope.jsonl --questions data/pope.questions.jsonl --labels data/pope.labels.jsonl
```

The [data directory](data/README.md) contains a 500-image COCO2014 captioning list,
a 240-question MME-Hall subset, and the COCO object lexicon.

## Evaluation

The runner accepts `--family llava`, `--family qwen`, or `--family instructblip`.
`--profile tmm` is the default and uses the supplement's Table VI parameters and
64-token caption limit. For a custom configuration, use `--profile custom
--config CONFIG.json`; `--max-new-tokens` sets its caption limit.

Prepare weakened images with `etscd-prepare-weak-images`, then select them with
`--weakening provided --weak-image-folder WEAK_IMAGES`. Prepared images use the
same relative filenames as the original images. The runner also supports Gaussian
blur directly through `--weakening gaussian-blur --blur-radius R`.

### POPE

Prepare frequency low-pass (FLL) images from the normalized question file. Use a
new output directory:

```bash
etscd-prepare-weak-images --questions data/pope.questions.jsonl --image-folder IMAGES --output-folder weak/fll --transform frequency-lowpass --fll-cutoff-ratio 0.16 --fll-order 4 --jpeg-quality 95
```

The preparation command also supports `--transform gaussian-blur --blur-radius 3`
and `--transform edge-overlay --edge-strength 0.58`.

Run one dataset and sampling setting, then score its predictions:

```bash
python -m etscd.runner --benchmark pope --family llava --model-path CHECKPOINT --questions data/pope.questions.jsonl --image-folder IMAGES --dataset coco --setting random --seed 42 --weakening provided --weak-image-folder weak/fll --output outputs/pope_single/llava/coco/random/seed42.jsonl
python -m etscd.evaluation score --benchmark pope --predictions outputs/pope_single/llava/coco/random/seed42.jsonl --labels data/pope.labels.jsonl --output outputs/pope_single/llava/coco/random/seed42.metrics.json
```

The seed script runs inference, scoring, and aggregation for seeds
`42 55 123 2024 3407`:

```bash
python scripts/run_pope_seeds.py --family llava --model-path CHECKPOINT --questions data/pope.questions.jsonl --labels data/pope.labels.jsonl --image-folder IMAGES --dataset coco --setting random --weakening provided --weak-image-folder weak/fll --output-dir outputs/pope/llava/coco/random
```

Add `--dry-run` to print the commands. Repeat for each dataset and each of
`random`, `popular`, and `adversarial`. Aggregation reports the mean and sample
standard deviation of the five matched runs for each setting separately:

```bash
python -m etscd.evaluation aggregate --inputs FILE1 FILE2 FILE3 FILE4 FILE5 --output SUMMARY.json
```

### MME-Hall

```bash
python -m etscd.runner --benchmark mme --family llava --model-path CHECKPOINT --questions data/mme_hall240.questions.jsonl --image-folder MME_IMAGES --dataset mme --setting hall240 --seed 42 --weakening gaussian-blur --blur-radius 5 --output outputs/mme/llava/seed42.jsonl
python -m etscd.evaluation score --benchmark mme --predictions outputs/mme/llava/seed42.jsonl --labels data/mme_hall240.labels.jsonl --output outputs/mme/llava/seed42.metrics.json
```

Use prepared full-benchmark question and label files in the same format to
evaluate the complete MME-Hall benchmark.

### CHAIR

Build the object-token map using the backbone's tokenizer, then generate and
evaluate captions:

```bash
python -m etscd.vocabulary --tokenizer-path TOKENIZER --output data/object_tokens.json
python -m etscd.runner --benchmark chair --family llava --model-path CHECKPOINT --questions data/chair500.questions.jsonl --image-folder COCO_VAL2014 --object-token-map data/object_tokens.json --dataset coco2014 --setting chair500 --seed 42 --weakening gaussian-blur --blur-radius 5 --output outputs/chair/llava/seed42.jsonl
python -m etscd.chair_eval --predictions outputs/chair/llava/seed42.jsonl --questions data/chair500.questions.jsonl --annotation-dir COCO_ANNOTATIONS --output outputs/chair/llava/seed42.metrics.json
```

Add `--qwen` when building a Qwen token map. For InstructBLIP, select the tokenizer
directory of its Vicuna language model. The builder maps complete single-token
object aliases to canonical COCO concepts.

CHAIR uses 500 distinct images, the prompt `Please describe this image in
detail.`, and up to 64 generated tokens under the TMM profile. Provide the COCO2014
caption and instance annotations. `--allow-subset` supports smaller inference
runs; CHAIR evaluation uses the complete 500-image list.

### Decoding options

`--method etscd` selects ET-SCD. `--method regular` selects the largest original
logit. `--method vcd` uses the contrastive score
`original + alpha * (original - weakened)` from Eq. (3).

## Outputs

Predictions contain the generated text and token IDs, per-step candidates,
evidence intervals, action risks, boundary targets, view gains, JS divergences,
and fusion weights. A neighboring `.manifest.json` file records the command,
configuration, seed, model/environment settings, and source/input/image hashes.

Efficiency fields include original, weakened, and re-query forward counts,
synchronized generation time, generated-token count, and CUDA process peak
allocation. Adapters evaluate each branch with the complete question and exact
generated token prefix, using full-context recomputation.

See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for the algorithm and
implementation details.

## Repository layout

```text
src/etscd/core.py          Shared ET-SCD controller
src/etscd/config.py        Policy configuration
src/etscd/adapters.py      Backbone adapters
src/etscd/runner.py        Unified inference entry point
src/etscd/weak_images.py   Weakened image preparation
src/etscd/views.py         Regional image views
src/etscd/evaluation.py    POPE/MME metrics and seed aggregation
src/etscd/chair_eval.py    CHAIR/BLEU evaluation
src/etscd/vocabulary.py    Object-token map construction
configs/etscd_tmm.json     Supplement Table VI parameters
tests/                    Controller and pipeline tests
```

Run the test suite with:

```bash
python -m unittest discover -s tests -v
```

Create a source archive with:

```bash
python scripts/build_release.py
```

## Citation and license

Please cite the accompanying ET-SCD manuscript when using this implementation.
Author and title metadata are available in [CITATION.cff](CITATION.cff).

This project uses the Apache-2.0 license; see [LICENSE](LICENSE). CHAIR metric and
object-vocabulary credits are listed in
[src/etscd/metrics/README.md](src/etscd/metrics/README.md).
