# MMDL Team 12 — MMMU Baseline Evaluation

Qwen3-VL-4B-Instruct를 MMMU validation 900문제에서 평가하는 코드와 결과입니다.

## 평가 설정

- 모델: `Qwen/Qwen3-VL-4B-Instruct`
- 모델 revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- 데이터셋: `MMMU/MMMU` validation, 30개 과목, 총 900문제
- 데이터셋 revision: `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`
- dtype: bf16 (양자화 없음)
- 추론 백엔드: vLLM 0.14.0
- 생성 설정: `temperature=0.7`, `top_p=0.8`, `top_k=20`,
  `repetition_penalty=1.0`, `presence_penalty=1.5`, `seed=42`
- 생성 길이: `max_new_tokens=16384`, `max_model_len=32768`
- 이미지 해상도: `min_pixels=1003520`, `max_pixels=4014080`

프롬프트와 생성 설정은
[Qwen3-VL 공식 MMMU 평가 코드](https://github.com/QwenLM/Qwen3-VL/tree/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu)를
참고했습니다. 채점은 외부 LLM judge 없이 규칙 기반 파서로 수행합니다.

## 설치

Python 3.12 환경에서 다음 명령으로 의존성을 설치합니다.

```bash
python -m pip install -r requirements.txt
```

## 실행

`<HF_DATASETS_CACHE>`와 GPU 번호를 실행 환경에 맞게 변경합니다.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mmmu_eval.sh \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --model-revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --data-root <HF_DATASETS_CACHE> \
  --output-dir results/baseline \
  --prompt-style official \
  --max-new-tokens 16384 \
  --max-model-len 32768 \
  --chunk-size 128 \
  --physical-gpu-index 0 \
  --gpu-memory-utilization 0.90
```

모델과 데이터셋은 Hugging Face 표준 캐시에 저장하며 저장소에는 포함하지 않습니다.
스크립트는 30개 과목 로드, 프롬프트 생성, vLLM 추론, 답안 파싱과 점수 집계를
한 번에 수행합니다.

## 결과

- 최종 정확도: **582/900 (64.67%)**
- 공식 수치 67.4와의 차이: **-2.73%p**
- 실행 시간: 3,307.3초 (약 55분 7초)
- peak GPU memory: 88,334 MiB
- 파서: `parser_version: 2`

세부 결과와 추가 실험은 [`results/README.md`](results/README.md), 최종 제출 내용은
[`reports/mmmu_baseline.md`](reports/mmmu_baseline.md)에 기록합니다.
