# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: 012
- **팀원**: 김다인, 최재원, 하유경
- **작성일**: 2026.09.28.
- **재현 커맨드**: `bash scripts/run_mmmu_eval.sh` (전체 커맨드는 1절에 기재)

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (ebb281ec70b05090aa6165b016eac8ec08e71b17) |
| 추론 백엔드 | `vLLM 0.14.0` 900개의 멀티모달 요청을 단일 대용량 GPU에서 효율적으로 배치 처리하기 위해 vLLM을 추론 백엔드로 선택하였다. |
| 사용 GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition 1장 (97,887 MiB, 약 95.6 GiB) |
| 실측 peak VRAM | 약 92.62 GB (88,334 MiB) |
| 총 소요 시간 | 3,307.3초 (약 55분 7초, 900문제 기준) |
| 의존성 | [`requirements.txt`](../requirements.txt) (Python 3.12)|
| 실행 커맨드 | 아래 재현 커맨드 참고 (<GPU_ID>: GPU 번호 (두 곳에 같은 값), <HF_DATASETS_CACHE>: MMMU 데이터 폴더 경로) |

```bash
# 환경 설치 (Python 3.12, 환경 이름은 자유롭게 변경 가능)
conda create -n mmdl python=3.12 -y && conda activate mmdl
pip install -r requirements.txt

# 평가 실행
CUDA_VISIBLE_DEVICES=<GPU_ID> bash scripts/run_mmmu_eval.sh \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --model-revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --data-root <HF_DATASETS_CACHE> \
  --output-dir results/baseline \
  --prompt-style official \
  --max-new-tokens 16384 \
  --max-model-len 32768 \
  --chunk-size 128 \
  --physical-gpu-index <GPU_ID> \
  --gpu-memory-utilization 0.90
```


## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문** (변수 부분은 `{}`로 표시):

```
Question: {question}
Options:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}
Please select the correct answer from the options above.
```

주관식 문제:

```text
Question: {question}
```

- **출처**: [Qwen3-VL 공식 MMMU 평가 코드의 `build_mmmu_prompt()`](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py#L23-L43) (+ 일부 수정)
- **선택 이유:** Qwen이 공개한 MMMU 평가 코드의 프롬프트 형식을 기준으로 사용하여, 임의의 프롬프트 설계가 성능에 미치는 영향을 줄이고 공식 결과와 가능한 한 유사한 조건에서 비교하기 위해 선택하였다. <br>
이후 Hugging Face MMMU/MMMU 데이터 구조와 평가 환경에 맞추어 일부 구현을 수정하였다. 우선 이미지 입력 방식을 Hugging Face MMMU/MMMU의 데이터 구조에 맞추기 위해 문항 내 `<image N>` 표기를 `[Image N]`으로 변환하였다. Hugging Face MMMU/MMMU에서는 실제 이미지가 `image_1`부터 `image_7`까지 별도의 필드로 제공되므로, 실제 이미지들은 텍스트와 분리된 multimodal content로 전달하고 질문 문자열에는 이미지의 위치를 나타내는 `[Image N]` 참조 표기만 남기도록 수정하였다.<br>
또한 모델이 응답을 생성하는 형식에 맞춰 입력을 구성하기 위해 모델의 chat template에 `add_generation_prompt=True`를 적용하였다. 객관식 선택지는 각 문항의 실제 선택지 수를 반영하기 위해 A부터 해당 문항의 마지막 선택지까지 동적으로 생성하도록 수정하였다.<br>
마지막으로 Qwen 공식 프롬프트 코드에는 선택적으로 `Hint:`를 추가하는 로직이 포함되어 있으나, 본 평가에 사용한 Hugging Face MMMU/MMMU 데이터에는 `hint` 필드가 존재하지 않았다. 따라서 평가 데이터에 존재하지 않는 정보를 별도로 고려하지 않도록 실제 입력 프롬프트에서는 `Hint:` 관련 로직을 제외하였다.



## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| `do_sample` | 해당 없음 (vLLM에 별도 인자 없음, `temperature=0.7`로 sampling 수행) |
| `temperature` | 0.7 |
| `top_p` | 0.8 |
| `top_k` | 20 |
| `repetition_penalty` | 1.0 |
| `presence_penalty` | 1.5 |
| `seed` | 42 |

- **출처**: [Qwen3-VL 공식 MMMU 평가 코드](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py)의 vLLM `SamplingParams` 및 `vLLM(seed=42)` 설정을 사용하였다.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | 16,384 |
| 이미지 해상도 처리 (`min_pixels`/`max_pixels` 등) | `min_pixels=1,003,520`, `max_pixels=4,014,080` |

**선택 근거** (본인이 사용한 인프라 제약과 어떻게 연결되는지 — 속도/VRAM/응답 잘림 등 trade-off): 이미지 해상도 범위는 [Qwen 공식 MMMU 평가](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py#L30-L31) 설정을 따랐다. 생성 길이는 동일한 객관식 120문제를 대상으로 parser v2에서 비교하였다. `max_new_tokens`가 4,096일 때 67/120(55.83%), 16,384일 때 74/120(61.67%), 32,768일 때 76/120(63.33%)이었다. 16,384에서 32,768로 늘렸을 때 정답은 2문제 증가했지만 32,768 실험에는 1,622.9초가 소요되었다. 따라서 전체 900문제 평가 시간과 이후 fine-tuned checkpoint 재평가 비용을 고려하여 16,384를 선택하였다. 최종 실행의 peak VRAM은 88,334 MiB였다. 

## 4. 채점(파싱) 방식

- 사용한 파서/로직: [`scripts/eval_mmmu.py`](../scripts/eval_mmmu.py)의 자체 규칙 기반 parser v2. 기본 구조는 [MMMU 공식 평가 유틸리티](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py)를 참고하였으며 외부 LLM judge는 사용하지 않았다.
- 동작 방식 요약: 객관식은 먼저 `Final answer: B`, `answer is B`, `\boxed{B}`와 같이 명시된 선택지를 탐색한다. 없으면 `(B)`, 독립된 `B`, `B.` 등을 찾고 여러 후보가 있으면 응답에서 가장 마지막에 나온 후보를 선택한다. 그래도 찾지 못하고 응답이 5단어보다 긴 경우 선택지 본문의 등장 위치를 이용해 fallback parsing을 수행한다. 길이 제한으로 종료된 응답에 `answer is`, `answer:`, `\boxed{` 같은 명시적 답 표현이 하나도 없으면 풀이 과정의 선택지를 정답으로 인정하지 않는다. 끝까지 추출하지 못한 경우 parsing failure로 처리하여 오답으로 기록한다. 주관식은 `answer`, `result`, `therefore`, `=` 등의 뒤쪽 표현 및 응답 내 숫자를 후보로 추출하고 문자열은 소문자화하며 숫자는 쉼표 제거 후 소수 둘째 자리까지 정규화하여 정답과 비교한다.

## 5. 결과

| No. | Subject | Data Num | Acc |
|---|---|---|---|
| 1 | Accounting | 30 | 80.00% |
| 2 | Agriculture | 30 | 46.67% |
| 3 | Architecture_and_Engineering | 30 | 63.33% |
| 4 | Art | 30 | 56.67% |
| 5 | Art_Theory | 30 | 76.67% |
| 6 | Basic_Medical_Science | 30 | 70.00% |
| 7 | Biology | 30 | 50.00% |
| 8 | Chemistry | 30 | 53.33% |
| 9 | Clinical_Medicine | 30 | 66.67% |
| 10 | Computer_Science | 30 | 56.67% |
| 11 | Design | 30 | 80.00% |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 40.00% |
| 13 | Economics | 30 | 86.67% |
| 14 | Electronics | 30 | 56.67% |
| 15 | Energy_and_Power | 30 | 63.33% |
| 16 | Finance | 30 | 76.67% |
| 17 | Geography | 30 | 46.67% |
| 18 | History | 30 | 66.67% |
| 19 | Literature | 30 | 76.67% |
| 20 | Manage | 30 | 63.33% |
| 21 | Marketing | 30 | 90.00% |
| 22 | Materials | 30 | 53.33% |
| 23 | Math | 30 | 63.33% |
| 24 | Mechanical_Engineering | 30 | 43.33% |
| 25 | Music | 30 | 23.33% |
| 26 | Pharmacy | 30 | 76.67% |
| 27 | Physics | 30 | 76.67% |
| 28 | Psychology | 30 | 76.67% |
| 29 | Public_Health | 30 | 90.00% |
| 30 | Sociology | 30 | 70.00% |
| | **Overall (macro avg)** | **900** | **64.67%** |

계산식: `Overall = mean(30개 과목 accuracy)`

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 64.67 |
| 차이 (Δ) | -2.73%p |

## 7. 격차 분석

재현 점수는 64.67%로 공식 결과(67.4%)보다 2.73%p 낮았다. 최종 실행에서 88개 응답이 16,384토큰 제한에 도달했으며, 이 중 17개(19.32%)만 정답이었다. 반면 정상 종료된 812개 중 565개(69.58%)가 정답이었다. 그러나 동일한 객관식 120문제에서 16,384토큰과 32,768토큰 설정의 정답 수가 각각 74개와 76개로 유사해, 토큰 제한만으로 성능 차이를 설명하기는 어려웠다.<br>
또한 Qwen 공식 평가 코드는 규칙 기반 답안 추출이 실패하면 GPT judge가 응답을 읽고 선택지를 고르지만, 본 파이프라인은 재현성과 비용을 고려해 규칙 기반 파서만 사용하고 parsing failure를 오답으로 처리하였다. 공식 규칙으로는 객관식의 56.55%(479/847)가 judge 판정 대상이어서, 이 구간에서 두 방식의 결과가 달라질 수 있다. 최종 실행에서 parsing failure는 45건이었으며, 주관식 정확도는 22/53(41.51%)였다. 따라서 공식 결과와의 차이는 응답 절단, 답안 추출 방식의 차이 등이 복합적으로 영향을 미친 결과로 판단된다.


## 8. 기타 특이사항 / 한계 (Optional)

- 초기 파서(v1)는 선택지 문자를 대소문자 구분 없이 찾았기 때문에, `answer is approximately`의 첫 글자 `a`를 선택지 A로 잘못 인식하는 경우가 있었다. 선택지 문자를 대문자로 제한하도록 수정한 파서(v2)로 v1 실행의 응답 900개를 다시 채점하면 정답 수가 546개에서 563개로 17개 늘어난다.

- 최종 제출 점수 582/900(64.67%)은 재채점 결과가 아니라, 파서 v2를 적용한 뒤 900문제 전체를 새로 생성하고 채점한 결과다. 수정 전 결과는 [`results/ablations/official_prompt_16384_full900_parser_v1/`](../results/ablations/official_prompt_16384_full900_parser_v1/)에 비교용으로 보관했다.

- 시간 제약으로 인해 `transformers.generate()` 기반 파이프라인을 동일 조건에서 끝까지 비교하지 못한 점은 한계로 남는다. vLLM과 동일한 프롬프트, sampling 설정, 파서 조건에서 Transformers 결과까지 비교했다면 추론 백엔드에 따른 성능 및 실행 시간 차이까지 더 명확히 분석할 수 있었을 것이다.
