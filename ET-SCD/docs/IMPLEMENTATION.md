# Implementation

POPE, MME-Hall, and CHAIR use the shared controller in `src/etscd/core.py`.
The default `tmm` profile loads the supplement's Table VI parameters from
`ETConfig` and uses a 64-token caption limit. Select `--profile custom --config
CONFIG.json` to use a custom policy.

## Algorithm

| Paper definition | Implementation |
|---|---|
| Eqs. (2)–(5): contrastive scores, entropy, adaptive candidates | `decode_step`, `normalized_entropy`, `candidate_set` |
| Eqs. (6)–(8): relative support and evidence intervals | `evidence` |
| Eqs. (9)–(10): endpoint worst-case risks | `risks`, `choose_action` |
| Supplement Algorithm S1: current-leader routing and bounded rejection | `decode_step` |
| Eqs. (11)–(12), supplement Eqs. (19)–(23): boundary construction | `boundary_set` |
| Eq. (13): JS-constrained view selection and gain gate | `decode_step` |
| Eqs. (14)–(15): adaptive evidence fusion | `decode_step` |
| Supplement Table VI: policy parameters | `configs/etscd_tmm.json` |
| Supplement S2.4: corner crops and query budgets | `views.corner_views` |

The original and weakened image branches produce logits `le` and `la` with the
same instruction and generated-token prefix. The evidence difference is
`delta = le - la`, and the base decoding score is `le + alpha * delta`.

Normalized full-vocabulary entropy determines the candidate probability mass and
candidate-count cap. Within the candidate set, relative support is
`softmax(delta / tau_s)`. Each support value `s` has interval half-width
`lambda_u * entropy + lambda_b * s * (1 - s)`; endpoints are clipped to `[0, 1]`.

The controller routes the current highest-scoring candidate to Accept, Reject,
or Defer by minimizing its endpoint worst-case risk. Risk ties follow
Accept, Reject, Defer order. Reject removes the current candidate subject to the
rejection budget. Defer constructs a boundary containing the leader, overlapping
competitors, and then additional candidates ordered by support and interval width.
Endpoint contact counts as overlap.

Local evidence is normalized over the boundary set before and after each query.
Original full-vocabulary entropy is used for the local intervals. View selection
maximizes the gain from increased support margin, decreased overlap, and decreased
mean interval width, subject to the JS-divergence bound. JS uses natural logarithms.
A selected view passing the gain threshold updates boundary evidence with the
adaptive fusion weight. The base score selects the output when the query or gain
conditions are unsatisfied.

Token-score ties use ascending token ID. Equal-gain views retain their generation
order.

## Image branches and targets

`etscd-prepare-weak-images` reads normalized questions containing `id`, `image`,
and `question`, and transforms each distinct image once. It writes to a new
directory, preserving relative filenames for `--weakening provided
--weak-image-folder WEAK_IMAGES`.

| Preparation transform | Operation | Default parameters |
|---|---|---|
| `frequency-lowpass` | Apply the radial mask `exp(-(r / cutoff)^order)` to each RGB channel in the Fourier domain; `cutoff = max(2, cutoff_ratio * min(height, width))` | `--fll-cutoff-ratio 0.16 --fll-order 4` |
| `gaussian-blur` | Apply Pillow Gaussian blur to the RGB image | `--blur-radius 3` |
| `edge-overlay` | Blend the RGB image with grayscale gradient magnitude normalized by its 99th percentile | `--edge-strength 0.58` |

Transforms retain image dimensions. JPEG output uses `--jpeg-quality 95` by
default. The preparation manifest records parameters and image hashes. The
runner also supports `--weakening gaussian-blur --blur-radius R` to construct the
weakened branch directly from each original image.

POPE and MME-Hall use two distinct complete-token yes/no spellings. Tokenizer
round-trip checks select the spellings before inference, and the manifest records
their IDs. The complete question is the query target, preserving count, color,
and position conditions.

CHAIR uses a tokenizer-specific object map built from the 80-concept COCO
lexicon. The builder checks lowercase, title-case, and leading-space spellings,
retaining complete single-token aliases. It records the remaining aliases in the
map metadata. The ratio of object tokens in the adaptive candidate set controls
object-sensitive activation. Boundary object concepts supply the query targets.

Regional views use the same four-corner geometry for each target, in order:
top-left, top-right, bottom-left, bottom-right. Crop width and height are the
nearest integers to `crop_ratio` times the image dimensions. Defaults use
`crop_ratio = 0.68`, up to two targets, and four total views per decoding step.
Targets are traversed in order, so the first target uses the four-view default
budget. Each crop is resized to the backbone's native visual input resolution.
Views are evaluated lazily when the controller selects Defer.

## Backbone adapters

Adapters return raw full-vocabulary next-token logits. Each image branch receives
the original instruction and the exact generated token IDs. Context is recomputed
with `use_cache=False` on every forward.

LLaVA uses its native conversation template and image-token preparation. Qwen
uses the checkpoint's native ChatML context helper and image placeholder tokens.
InstructBLIP uses LAVIS's `blip2_vicuna_instruct` / `vicuna7b` model and its
configured language-model tokenizer. Generation stops at the backend EOS IDs,
which are saved in the run manifest.

`--method regular` maximizes original logits. `--method vcd` maximizes
`le + alpha * (le - la)`, the score in Eq. (3). The binary tasks restrict output
selection to their resolved yes/no token pair.

## Evaluation and logs

Inference reads `id`, `image`, and `question` fields. Scoring reads predictions
and a separate label file.

POPE reports metrics separately for each dataset, sampling setting, model, and
method. Five-seed aggregation uses seeds `42`, `55`, `123`, `2024`, and `3407`,
matching sample IDs and configuration. Standard deviation uses `ddof=1`.

MME-Hall evaluates existence, count, position, and color. Each category/image has
two questions. The category score is question accuracy plus paired-image accuracy,
both expressed as percentages. A category score ranges from 0 to 200 and the
four-category total ranges from 0 to 800.

CHAIR evaluation matches predictions to 500 distinct COCO2014 image IDs and the
prompt `Please describe this image in detail.`. It computes CHAIRs, CHAIRi,
Recall, F1, and BLEU-4 using COCO caption and instance annotations.

Prediction records contain candidates, support intervals, risks, actions,
rejected tokens, boundary targets, view gains, JS divergences, and fusion weights.
The manifest records run status, command, configuration, source and input hashes,
seed, model settings, package versions, and image hashes.

Efficiency records count original, weakened, and re-query forwards separately.
Generation time is synchronized with the device. CUDA memory is the process peak
allocation in bytes. The cache mode is `full_context_recompute`.
