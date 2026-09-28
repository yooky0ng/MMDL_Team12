# MMDL Team 12 — MMMU Baseline Evaluation

Qwen3-VL-4B-Instruct를 MMMU validation split 900문제에서 평가하는 코드와
결과를 정리한 저장소입니다.

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

프롬프트와 generation recipe는
[Qwen3-VL 공식 MMMU 평가 코드](https://github.com/QwenLM/Qwen3-VL/tree/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu)를
참고했습니다. 채점은 외부 LLM Judge 없이 자체 규칙 기반 파서로 수행합니다.

## 환경

- Python 3.12
- vLLM 0.14.0
- PyTorch 2.9.1
- Transformers 4.57.6
- Datasets 5.0.1
- qwen-vl-utils 0.0.14

패키지는 다음 명령으로 설치합니다.

```bash
python -m pip install -r requirements.txt
```

## 평가 실행

아래 명령에서 `<HF_DATASETS_CACHE>`와 GPU 번호를 실행 환경에 맞게 변경합니다.

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

모델과 데이터셋은 Hugging Face 표준 캐시에 내려받으며 저장소에는 가중치나
데이터를 포함하지 않습니다.

`scripts/run_mmmu_eval.sh`는 CUDA 라이브러리 경로를 준비한 뒤
`scripts/eval_mmmu.py`를 실행합니다. Python 스크립트는 30개 과목을 각각
불러오고, 프롬프트 생성·vLLM 추론·답안 파싱·점수 집계를 수행합니다.

## 결과

최종 baseline은 900문제 중 546문제를 맞혀 60.67%를 기록했습니다. 공식
수치 67.4와의 차이는 -6.73%p입니다. NVIDIA RTX PRO 6000 Blackwell 한 장에서
총 3,708.2초가 걸렸고, peak GPU memory는 88,334 MiB였습니다.

- `results/baseline/summary.json`: 실행 설정과 전체·과목별 결과
- `results/baseline/scores.csv`: 과목별 점수
- `results/ablations/`: 생성 길이와 프롬프트 비교 실험
- `reports/mmmu_baseline.md`: 제출 보고서

원본 생성 응답인 `predictions.jsonl`은 로컬 분석용이며 Git에는 포함하지
않습니다.
