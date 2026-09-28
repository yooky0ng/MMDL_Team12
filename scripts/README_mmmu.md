# MMMU: assignment data, Qwen generation recipe, API-free scoring

Activate the existing environment first:

```bash
conda activate qwen-mmmu
cd /home/dikim/models
bash run_mmmu.sh --stage check
bash run_mmmu.sh --stage infer
```

`check` verifies local HF download revision metadata and exactly 30 subjects /
900 unique validation IDs. Metadata checks do not hash the model weight contents.
Model revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`.
Dataset revision: `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`.

The shell script respects existing CUDA_VISIBLE_DEVICES; otherwise it selects
physical GPU 1 (under the runtime's CUDA device ordering). To select another GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_mmmu.sh --stage infer
```

The default backend is now **Transformers**, using
`Qwen3VLForConditionalGeneration.generate()` on the first visible GPU,
bfloat16, SDPA attention and batch size 1. The prompt uses the Qwen function; default scoring extends Qwen rules without an API. This is a backend experiment, not the official vLLM execution.

Both backends default to seed 42, max_new_tokens 32768, max_model_len 128000,
temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0,
presence_penalty 1.5 and min/max_pixels 1003520/4014080. Transformers uses a
custom additive presence penalty on generated tokens only (once per distinct
token). It checks input length plus the requested generation budget against
max_model_len without truncating. No added CoT instruction is used.
GPU utilization 0.90 and automatic tensor parallelism apply only to vLLM.
The Transformers implementation uses one GPU, not tensor parallelism.
Identical seeds/settings do not guarantee identical outputs between backends.

```bash
# Default: Transformers, results_transformers/
bash run_mmmu.sh --stage infer
bash run_mmmu.sh --stage eval
# Previous backend: vLLM, results_official/
bash run_mmmu.sh --backend vllm --stage infer
bash run_mmmu.sh --backend vllm --stage eval
```

Output directories are selected by backend unless --output-dir is provided.
The previous results/ and results_official/ are preserved by the default run.
An existing predictions.jsonl is never overwritten by inference: use a new
`--output-dir` for another run. `run_config.json` records inference settings,
backend, actual dtype and inference duration (Transformers includes per-sample preprocessing). Images and the exact messages are saved.

## Evaluation (API-free by default)

```bash
# Score saved vLLM predictions; no inference and no API requests:
bash run_mmmu.sh --backend vllm --stage eval
# Score saved Transformers predictions:
bash run_mmmu.sh --stage eval
# One command: inference followed by API-free evaluation:
bash run_mmmu.sh --backend vllm
```

`--scoring rules` is the default. `local_scoring.py` defines the frozen parser
`qwen-first-local-v1`. It does not call external services, including when API
credentials happen to be present. It is NOT the full official Qwen Judge pipeline.
The same parser must be reused for fine-tuned checkpoints; do not select rules
per question or adjust them to improve a known score.

### Multiple-choice extraction

1. Apply the vendored Qwen `can_infer` to the full response (choice letters,
   then unique matching choice content). Preserve its results, including Z.
2. If unresolved, remove Markdown bold emphasis and find explicit Answer /
   Final answer / Correct answer expressions and `\boxed{letter}`. Choose the
   last match by actual character position, with a token boundary after the
   letter; answer words such as "Because" must not become option B.
3. If unresolved, apply Qwen rules to the last nonempty paragraph.
4. If unresolved, use the last standalone uppercase valid choice letter.
5. No answer found: null prediction and incorrect. No random guessing.

### Open-answer extraction

Open answers are scored directly, not converted into choices for a Judge.
Candidate extraction is adapted from the user-supplied MMMU-style script:
extract tails after answer/result/therefore/is/etc. markers; use the full response
if there are no such tails; also extract signed decimal, scientific-notation and
thousands-separated numbers. Numeric values are rounded to two decimal places.
Strings are case-folded and compared by contained text with word boundaries.
Gold alternatives expressed as a Python list are accepted. Gold is used only in
comparison, never to choose what to extract from the model response.

Differences from the supplied script: explicit and boxed MC answers follow actual
text order; open sentences are split before lowercasing; numeric matches avoid
partial exponent/thousands matches; text boundaries avoid matching C inside cat.
The pipeline still has limitations: the last mentioned choice may not be the
intended answer, and intermediate numbers or quoted answer text can count as a
match. Rounding to two decimals can accept small numeric differences. These
are documented evaluation rules, not semantic equivalence guarantees.

Legacy predictions omit finish_reason/output token counts. We cannot reproduce
the supplied script's special rejection of length-truncated responses; no such
check or inferred truncation count is claimed. Existing predictions are unchanged.

### Outputs

- `scores_rules.json`: macro/micro accuracy as percentages, 30 subject scores,
  parser version, extraction method counts and extraction failures.
- `scores_rules.csv`: 30 subject rows and one macro-average row.
- `evaluation_rules.jsonl`: every generated answer, extracted answer, ground
  truth, correctness and extraction method (including last-choice fallback).

All 900 pinned validation IDs must be present. Failures remain in the denominator.
Subject macro average equals total correct / 900 because every subject has 30 rows.
These files do not overwrite existing predictions or Judge score files.

### Optional original Judge mode

`--scoring judge` explicitly enables the existing official rule + Judge pathway.
It requires CHATGPT_DASHSCOPE_API_KEY and DASHSCOPE_API_BASE for `--api-type dash`,
or MIT_SPIDER_TOKEN and MIT_SPIDER_URL for `--api-type mit`. Use a full POST endpoint.
The default model is gpt-3.5-turbo-0125. Missing required credentials stop evaluation.
This mode uses the official open-question A=answer/B=Other Answers adaptation and
retains the upstream retry/random-option fallback after exhausted Judge failures.
It writes scores_judge.json, scores_judge.csv and evaluation_judge.jsonl.

Upstream attribution and the exact source commit are in `official_mmmu/README.md`.
Transformers interface: https://huggingface.co/docs/transformers/model_doc/qwen3_vl
