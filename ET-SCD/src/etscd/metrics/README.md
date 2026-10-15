# CHAIR metrics

`chair.py` implements the CHAIR metric and COCO object vocabulary following the
public CHAIR code and FuzzyCD's Python 3 adaptation. Credits to their authors.
The project license is available in [LICENSE](../../../LICENSE).

`etscd.chair_eval` matches caption and image IDs, loads COCO2014 caption and
instance annotations, and computes CHAIRs, CHAIRi, Recall, and F1. BLEU-4 uses
pycocoevalcap's PTBTokenizer and BLEU implementation.

Install the evaluation packages from `requirements-eval.txt`, Java, and the NLTK
resources listed in [ENVIRONMENT.md](../../../ENVIRONMENT.md). Run the evaluator
with:

```bash
python -m etscd.chair_eval --predictions PREDICTIONS.jsonl --questions data/chair500.questions.jsonl --annotation-dir COCO_ANNOTATIONS --output METRICS.json
```
