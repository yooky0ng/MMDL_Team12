# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: _(MMDL Team 12)_
- **팀원**: _(김다인, 최재원, 하유경)_
- **작성일**: _(2026.09.28.)_
- **재현 커맨드**: `(예: bash scripts/run_mmmu_eval.sh)`

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (ebb281ec70b05090aa6165b016eac8ec08e71b17) |
| 추론 백엔드 | `vLLM 0.14.0` |
| 사용 GPU | `NVIDIA RTX PRO 6000 Blackwell, VRAM: _(확인 필요)_ GB` |
| 실측 peak VRAM | _(88,334 MiB (약 86.26 GiB))_ |
| 총 소요 시간 | _(3,708.2초 (약 61분 48초))_ |
| 의존성 | _([`requirements.txt`](../requirements.txt))_ |
| 실행 커맨드 | 아래 재현 커맨드 참고 |

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

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문** (변수 부분은 `{}`로 표시):

```
_(Question: {question}
Options:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}
Please select the correct answer from the options above.)_
```

- **출처**: _(Qwen3-VL 공식 MMMU 평가 코드의 build_mmmu_prompt()를 참고하여 구현하였습니다.)_
- **선택 이유**: _(왜 이 프롬프트를 골랐는지)_


- **출처**: (예시) 오픈소스 평가 툴킷 XYZ의 프롬프트 생성 함수에서 차용, 문구 일부만 수정
- **선택 이유**: (예시) 모델이 장황한 설명 없이 선택지 하나로 바로 답하도록 유도하기 위해 간결한 지시문 사용


## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| `do_sample` | 별도 인자 없음 (`vLLM SamplingParams`에서 sampling 방식 사용) |
| `temperature` | 0.7 |
| `top_p` | 0.8 |
| `top_k` | 20 |
| `repetition_penalty` | 1.0 |
| `presence_penalty` | 1.5 |
| `seed` | 42 |

- **출처**: _([Qwen3-VL 공식 MMMU Instruct inference 설정](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/infer_instruct.sh)  )_

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | 16,384 |
| 이미지 해상도 처리 (`min_pixels`/`max_pixels` 등) | `min_pixels=1,003,520`, `max_pixels=4,014,080`, `max_model_len=32,768` |

**선택 근거** (본인이 사용한 인프라 제약과 어떻게 연결되는지 — 속도/VRAM/응답 잘림 등 trade-off): 생성 길이의 영향을 확인하기 위해 동일한 객관식 120문제(30개 과목에서 과목별 4문제)에 대해 `max_new_tokens`를 128, 4096, 32768로 변경하여 비교하였습니다.

| `max_new_tokens` | Accuracy | Parsing failure | Length limit 도달 | 소요 시간 |
|---|---|---|---|---|
| 128 | 14.17% | 90/120 | 112/120 | 99.3초 |
| 4,096 | 54.17% | 17/120 | 20/120 | 290.4초 |
| 32,768 | 59.17% | 7/120 | 17/120 | 4,536.3초 |

128토큰에서는 대부분의 응답이 생성 길이 제한에 도달하여 정확도가 크게 낮아졌습니다. 4,096토큰으로 확대했을 때 정확도가 54.17%까지 증가하고 length limit 및 parsing failure가 크게 감소하였습니다. 32,768토큰에서는 정확도가 59.17%로 추가 상승했지만 실행 시간이 크게 증가하였습니다. 이에 전체 900문제 평가에서는 성능과 실행 비용 사이의 trade-off를 고려하여 중간 생성 예산인 16,384토큰을 사용하였습니다.
단, 16,384토큰 조건을 동일한 120문제 subset에서 별도로 비교한 실험은 수행하지 않았으므로 16,384가 최적값임을 실험적으로 확정한 것은 아닙니다.

## 4. 채점(파싱) 방식

- 사용한 파서/로직: _(자체 규칙 기반 파서 ([`scripts/eval_mmmu.py`](../scripts/eval_mmmu.py)))_
- 동작 방식 요약: 객관식 문제에서는 다음 순서로 모델의 자유 텍스트 응답에서 선택지를 추출하였습니다.
1. answer is A, final answer: B 등 명시적인 final-answer 표현을 우선 탐색한다.
2. \boxed{A} 형태의 응답을 탐색한다.
3. 위 형식이 없는 경우 (A), 독립적인 A, A. 등의 선택지 표기를 탐색한다.
4. 여러 선택지 후보가 검출되면 응답 내에서 가장 마지막에 등장한 후보를 선택한다.
5. 선택지 문자를 찾지 못하고 응답이 충분히 긴 경우, 선택지의 실제 텍스트가 응답에 포함되어 있는지를 fallback으로 확인한다.
6. 위 규칙으로도 답을 추출하지 못하면 None으로 처리하고 오답으로 채점한다.
또한 모델 응답이 생성 길이 제한으로 종료(finish_reason == "length")되었고 answer, final answer, \boxed{} 등의 명시적인 최종 답안 표현이 없는 경우에는, 추론 과정 중 우연히 등장한 선택지 문자가 정답으로 잘못 파싱되는 것을 방지하기 위해 parsing failure로 처리합니다.
주관식 문제에서는 응답을 소문자화하고 불필요한 구두점을 정리한 뒤, is, thus, therefore, final, answer, result 등의 표현 뒤에 등장하는 응답을 우선 추출합니다. 숫자, 쉼표가 포함된 숫자, 소수 및 scientific notation을 추가로 추출하고 수치는 소수점 둘째 자리까지 정규화합니다. 이후 정규화된 예측값과 gold answer를 비교하여 정답 여부를 결정합니다.

## 5. 결과

| No. | Subject | Data Num | Acc |
|---|---|---|---|
| 1 | Accounting | 30 | 70.00% |
| 2 | Agriculture | 30 | 53.33% |
| 3 | Architecture_and_Engineering | 30 | 46.67% |
| 4 | Art | 30 | 53.33% |
| 5 | Art_Theory | 30 | 76.67% |
| 6 | Basic_Medical_Science | 30 | 73.33% |
| 7 | Biology | 30 | 56.67% |
| 8 | Chemistry | 30 | 46.67% |
| 9 | Clinical_Medicine | 30 | 66.67% |
| 10 | Computer_Science | 30 | 56.67% |
| 11 | Design | 30 | 70.00% |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 36.67% |
| 13 | Economics | 30 | 76.67% |
| 14 | Electronics | 30 | 43.33% |
| 15 | Energy_and_Power | 30 | 43.33% |
| 16 | Finance | 30 | 73.33% |
| 17 | Geography | 30 | 63.33% |
| 18 | History | 30 | 66.67% |
| 19 | Literature | 30 | 76.67% |
| 20 | Manage | 30 | 56.67% |
| 21 | Marketing | 30 | 86.67% |
| 22 | Materials | 30 | 30.00% |
| 23 | Math | 30 | 50.00% |
| 24 | Mechanical_Engineering | 30 | 46.67% |
| 25 | Music | 30 | 20.00% |
| 26 | Pharmacy | 30 | 80.00% |
| 27 | Physics | 30 | 76.67% |
| 28 | Psychology | 30 | 80.00% |
| 29 | Public_Health | 30 | 90.00% |
| 30 | Sociology | 30 | 53.33% |
| | **Overall (macro avg)** | **900** | **60.67%** |

계산식: `Overall = mean(30개 과목 accuracy)`

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 60.67 |
| 차이 (Δ) | -6.73%p |

## 7. 격차 분석

_(우리 재현 결과는 60.67%로 공식 성능 67.4보다 6.73%p 낮았다. 생성 길이의 영향을 확인하기 위해 동일한 객관식 120문제에서 max_new_tokens를 128, 4,096, 32,768로 변경한 결과 정확도는 각각 14.17%, 54.17%, 59.17%였고, length limit 도달은 112건, 20건, 17건으로 감소하였다. 이는 생성 예산이 평가 성능에 직접 영향을 줄 수 있음을 보여준다. 최종 평가에서는 16,384토큰을 사용했지만 여전히 900문제 중 96개 응답이 길이 제한에 도달했고, 객관식 43건에서 parsing failure가 발생하였다. 또한 본 평가는 외부 LLM Judge 없이 규칙 기반 파서만 사용하므로 자유 형식 응답을 정답으로 추출하지 못하는 경우가 있을 수 있다. 따라서 생성 길이 제한과 답안 파싱 방식의 차이가 공식 수치와의 격차에 일부 영향을 주었을 가능성이 있다.)_


## 8. 기타 특이사항 / 한계 (Optional)

- 생성 길이 비교 실험은 동일한 120개 객관식 subset에서 128, 4,096, 32,768토큰을 비교하였으나, 최종 선택값인 16,384토큰을 동일 subset에서 별도로 평가하지 않았다. 따라서 16,384는 실험적으로 확인한 최적값이 아니라 전체 평가 시간과 성능을 고려한 engineering choice이다.
- 초기 direct prompt 실험은 900문제 전체에서 수행되었고, official prompt의 128-token 실험은 120문제 subset에서 수행되었다. 평가 문항 집합이 서로 다르므로 두 실험의 정확도를 이용해 프롬프트 자체의 효과를 직접 비교하기는 어렵다.
- 최종 평가에서 96/900개의 응답이 생성 길이 제한에 도달하였고, 객관식 43/900건에서 parsing failure가 발생하였다. 향후 생성 예산과 parsing 방식을 추가 검토할 필요가 있다.
