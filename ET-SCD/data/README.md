# Data

| File | Contents |
|---|---|
| `chair500.questions.jsonl` | 500 distinct COCO2014 validation image IDs with the prompt `Please describe this image in detail.` |
| `mme_hall240.questions.jsonl` | 240 MME-Hall questions: 60 each for existence, count, position, and color |
| `mme_hall240.labels.jsonl` | MME-Hall answers, category names, and paired-image IDs |
| `coco_object_concepts.json` | 80 canonical COCO object concepts and their aliases |

Place images and COCO annotations in local directories and pass their paths to
the runner and evaluator. Prepare POPE or additional MME-Hall data with:

```bash
python -m etscd.data --benchmark pope --input RAW.jsonl --questions QUESTIONS.jsonl --labels LABELS.jsonl
python -m etscd.data --benchmark mme --input RAW.jsonl --questions QUESTIONS.jsonl --labels LABELS.jsonl
```

Question records contain `id`, `image`, and `question`. Label records contain
`id` and `label`; MME-Hall labels also contain `category` and `image_id`. Each
MME-Hall category/image pair has two questions.

Generate a tokenizer-specific object map from the lexicon with:

```bash
python -m etscd.vocabulary --tokenizer-path TOKENIZER --output data/object_tokens.json
```

Add `--qwen` for the Qwen-VL-Chat tokenizer. For InstructBLIP, use the tokenizer
directory of its Vicuna language model.
