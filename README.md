# MMDL Team 12 — Qwen3-VL MMMU 평가 재현

MMMU validation 900문제에서 원본 Qwen3-VL-4B-Instruct는 **575/900 (63.89%)**,
최종 RLVR 모델(어댑터 스케일 1.0)은 **567/900 (63.00%)**를 기록했습니다.

- [최종 모델 재현](#최종-모델-재현) · 저장소에 포함된 LoRA 어댑터를 병합해 평가합니다.
- [원본 Qwen 베이스라인 재평가](#원본-qwen-베이스라인-재평가) · 같은 MMMU 설정으로 원본 모델을 평가합니다.
- [실험별 점수와 평가 조건](results/README.md) · 전체 점수와 과목별 점수를 확인할 수 있습니다.

## 제공 범위

| 대상 | 저장소에서 제공하는 내용 |
| --- | --- |
| **최종 모델과 원본 Qwen 베이스라인** | 재평가 스크립트와 실행 명령을 제공합니다. 최종 모델의 LoRA 어댑터는 [`models/final_lora/`](models/final_lora/)에 포함되어 있으며, 베이스라인은 공개된 원본 Qwen 모델을 사용합니다. |
| **비교 실험** | 학습 코드, 학습 데이터와 문항 선별 기록, 전체·과목별 평가 점수를 제공합니다. 비교 실험의 어댑터는 로컬에 보관하며, 해당 모델의 직접 재평가에는 실험별 어댑터가 필요합니다. |

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

평가는 Linux, Python 3.12, CUDA를 사용할 수 있는 NVIDIA GPU 환경에서 실행했습니다.
원본 Qwen과 MMMU 데이터는 고정된 revision에서 받아 Hugging Face 캐시에 저장합니다.
최종 모델 재현에는 CPU에서 모델을 병합할 메모리와 병합본을 저장할 디스크 공간도
필요합니다. 기존 병합본은 약 8.3 GiB였습니다. 평가에는 아래 의존성을 설치합니다.

```bash
python -m pip install -r requirements.txt
```

## 원본 Qwen 베이스라인 재평가

저장소 루트에서 실행합니다. 다른 GPU를 쓰려면 첫 줄의 번호를 변경합니다.

```bash
GPU=0
CUDA_VISIBLE_DEVICES="$GPU" bash scripts/run_mmmu_eval.sh \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --model-revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --data-root "${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}" \
  --output-dir results/baseline/baseline_qwen3vl_4b_gpu050_mmmu_full900_reproduction \
  --prompt-style official \
  --max-new-tokens 16384 \
  --max-model-len 32768 \
  --chunk-size 128 \
  --seed 42 \
  --physical-gpu-index "$GPU" \
  --gpu-memory-utilization 0.50
```

모델과 데이터셋은 Hugging Face 표준 캐시에 저장하며 저장소에는 포함하지 않습니다.
스크립트는 30개 과목 로드, 프롬프트 생성, vLLM 추론, 답안 파싱과 점수 집계를
한 번에 수행합니다.

## 결과

- 최종 정확도: **575/900 (63.89%)**
- 공식 수치 67.4와의 차이: **-3.51%p**
- 실행 시간: 3,458.6초 (약 57분 39초)
- peak GPU memory: 85,723 MiB
- 파서: `parser_version: 2`

세부 결과와 추가 실험은 [`results/README.md`](results/README.md), 이전 0.90 실행 분석은
[`reports/mmmu_baseline.md`](reports/mmmu_baseline.md)에 기록합니다.

## 최종 모델 재현

발표에서는 `results/baseline/baseline_qwen3vl_4b_gpu050_mmmu_full900/`의 원본 Qwen
평가(575/900, 63.89%)를 비교 기준으로 사용합니다. 최종 모델의 결과는
`results/rlvr/base_m3cot_rlvr_hard104_lr5e6_mmmu_full900/`에 있습니다(567/900, 63.00%).
두 실행의 vLLM GPU 메모리 설정은 0.50입니다.

최종 모델은 M3CoT 104문항으로 원본 Qwen에 강화학습을 적용해 얻은 LoRA 어댑터를
스케일 1.0으로 병합한 모델입니다. 어댑터 설정과 가중치는 이 저장소의
[`models/final_lora/`](models/final_lora/)에 있습니다. 평가 의존성을 설치한
환경에서 아래 명령으로 원본 Qwen에 어댑터를 병합하고 MMMU를 평가합니다.
결과는 새 평가 폴더에 저장됩니다.

```bash
python -m pip install peft==0.21.3
python scripts/merge_lora_simple.py \
  --adapter models/final_lora \
  --scale 1.0 \
  --output outputs/final_rlvr_scale1.0

GPU=0
CUDA_VISIBLE_DEVICES="$GPU" bash scripts/run_mmmu_eval.sh \
  --model-path outputs/final_rlvr_scale1.0 \
  --model-revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --data-root "${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}" \
  --output-dir results/rlvr/base_m3cot_rlvr_hard104_lr5e6_mmmu_full900_reproduction \
  --prompt-style official \
  --max-new-tokens 16384 \
  --max-model-len 32768 \
  --chunk-size 128 \
  --seed 42 \
  --physical-gpu-index "$GPU" \
  --gpu-memory-utilization 0.50
```

병합에는 [merge_lora_simple.py](scripts/merge_lora_simple.py)를 사용합니다.
[run_mmmu_eval.sh](scripts/run_mmmu_eval.sh)는
[eval_mmmu.py](scripts/eval_mmmu.py)를 실행해 MMMU 900문제를 평가합니다.

## 재학습 데이터

재학습에는 학습 JSON 외에도 Point-CoT 원본과 `source_images/`, MMVU 이미지,
GQA 선별 이미지, M3CoT 이미지가 필요합니다. Point-CoT와 MMVU의 복원 설명은
`datasets/point_cot/README.md`와 `datasets/mmvu_48k/README.md`에 있습니다.
GQA와 M3CoT의 데이터 준비 코드는 `scripts/data_preparation/`에 있습니다.
다른 환경에서 재학습할 때는 학습 JSON에 저장된 이미지 경로를 해당 환경에 맞게
지정해야 합니다.
