# Qwen upstream files

The Python files and `infer_instruct.sh` in this directory originate from:
https://github.com/QwenLM/Qwen3-VL/tree/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu

Upstream license: Apache-2.0 (see LICENSE).

`../eval_mmmu.py` uses the upstream prompt/vision functions and answer extraction,
including `MMMU_preproc` and LLM Judge fallback. Its local adapter replaces TSV
loading with the assignment's pinned HF validation data. Do not run the upstream
entry point directly for the assignment: it loads MMMU_DEV_VAL.tsv instead.

HF image pixels are exported as PNG, questions retain their original text, and
HF option lists become A/B/C/... fields. Open-question placeholder options are
ignored. The upstream prompt receives no choices for open questions; choices
are synthesized only during evaluation. This preserves the assignment data,
but does not claim byte-identical inputs to the official TSV/JPEG pipeline.

HF adapter exception: ground-truth option mapping uses `question_type`. Every
open question maps to A after MMMU_preproc, including literal answers A/B/C.
Upstream's `answer in A..Z` heuristic would mis-score the HF open answers C
(`validation_Basic_Medical_Science_10`) and B (`validation_Pharmacy_19`).
Local import-only change in run_mmmu.py: vLLM is imported inside run_inference
instead of at module load, so Transformers can reuse the prompt helpers without
requiring vLLM. The prompt and evaluation functions are unchanged.
