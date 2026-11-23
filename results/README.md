# 평가 결과

## 최종 베이스라인

평가에 사용할 수 있는 GPU 메모리가 제한적이어서 원본 Qwen을 vLLM GPU 메모리 설정
`0.50`으로 평가했습니다. [이 실행 결과](baseline/baseline_qwen3vl_4b_gpu050_mmmu_full900/summary.json)를
발표와 실험 비교의 최종 베이스라인으로 사용합니다.

- 정확도 **575/900 (63.89%)**
- 실행 시간 **3,458.6초 (약 57분 39초)**
- 관측 최대 GPU 메모리 **85,723 MiB**
- 길이 상한 도달 **94/900**, 답안 파싱 실패 **40/900**

이전 GPU 메모리 설정 0.90 실행은 [이전 0.90 실행 결과](baseline/summary.json)에 보관합니다.
이전 실행은 582/900 (64.67%)였으며, 아래 추가 실험에서 비교한 결과입니다.

`summary.json`에는 실행 설정과 전체·과목별 결과가, `scores.csv`에는 과목별
정답 수와 정확도가 들어 있습니다. 원본 응답인 `predictions.jsonl`은 로컬 분석용으로
보관하고 Git에는 포함하지 않습니다.

## 발표 실험 결과

완료된 제출 결과는 베이스라인(`baseline/`), 지도 미세조정(`sft/`), 강화학습(`rlvr/`) 아래에 있습니다.

| 번호 | 실험 | MMMU 정답 및 정확도 | GPU 메모리 설정 | 결과 파일 |
| ---: | --- | ---: | ---: | --- |
| 1 | 원본 Qwen 베이스라인 | 575/900 · 63.89% | `0.50` | [전체 결과](baseline/baseline_qwen3vl_4b_gpu050_mmmu_full900/summary.json) · [과목별 점수](baseline/baseline_qwen3vl_4b_gpu050_mmmu_full900/scores.csv) |
| 2 | Mixed 30K SFT | 461/900 · 51.22% | `0.50` | [전체 결과](sft/mixed_sft_lora_mmmu_full900/summary.json) · [과목별 점수](sft/mixed_sft_lora_mmmu_full900/scores.csv) |
| 3 | Mixed 10K SFT | 448/900 · 49.78% | `0.50` | [전체 결과](sft/mixed_10k_reasoning_lora_mmmu_full900/summary.json) · [과목별 점수](sft/mixed_10k_reasoning_lora_mmmu_full900/scores.csv) |
| 4 | Point-CoT + GQA Plain | 406/900 · 45.11% | `0.50` | [전체 결과](sft/point_gqa_4k_plain_mmmu_full900/summary.json) · [과목별 점수](sft/point_gqa_4k_plain_mmmu_full900/scores.csv) |
| 5 | Point-CoT + GQA Grounded | 386/900 · 42.89% | `0.50` | [전체 결과](sft/point_gqa_4k_grounded_mmmu_full900/summary.json) · [과목별 점수](sft/point_gqa_4k_grounded_mmmu_full900/scores.csv) |
| 6 | MMVU 4K · 학습률 `5e-5` | 486/900 · 54.00% | `0.50` | [전체 결과](sft/mmvu_only_4k_local_base_mmmu_full900/summary.json) · [과목별 점수](sft/mmvu_only_4k_local_base_mmmu_full900/scores.csv) |
| 7 | MMVU 4K · 학습률 `1e-5` | 554/900 · 61.56% | `0.50` | [전체 결과](sft/mmvu_only_4k_lr1e5_mmmu_full900/summary.json) · [과목별 점수](sft/mmvu_only_4k_lr1e5_mmmu_full900/scores.csv) |
| 8 | Grounded + Final answer | 550/900 · 61.11% | `0.50` | [전체 결과](sft/point_gqa_4k_grounded_final_answer_lr1e5_restart_mmmu_full900/summary.json) · [과목별 점수](sft/point_gqa_4k_grounded_final_answer_lr1e5_restart_mmmu_full900/scores.csv) |
| 9 | M3CoT 1.6K SFT | 528/900 · 58.67% | `0.50` | [전체 결과](sft/m3cot_1600_lr1e5_mmmu_full900/summary.json) · [과목별 점수](sft/m3cot_1600_lr1e5_mmmu_full900/scores.csv) |
| 10 | MMVU LoRA → M3CoT RLVR | 538/900 · 59.78% | `0.50` | [전체 결과](rlvr/mmvu_m3cot_rlvr_final104_mmmu_full900/summary.json) · [과목별 점수](rlvr/mmvu_m3cot_rlvr_final104_mmmu_full900/scores.csv) |
| 11 | **최종 모델 · 원본 Qwen → M3CoT 강화학습 (스케일 1.0)** | 567/900 · 63.00% | `0.50` | [전체 결과](rlvr/base_m3cot_rlvr_hard104_lr5e6_mmmu_full900/summary.json) · [과목별 점수](rlvr/base_m3cot_rlvr_hard104_lr5e6_mmmu_full900/scores.csv) |

어댑터 스케일 0.5는 11번과 같은 어댑터를 사용한 병합 비교입니다. MMMU에서 568/900 (63.11%)를 기록했습니다. [전체 결과](rlvr/rlvr_lr5e6_scale0.5_mmmu_full900/summary.json) · [과목별 점수](rlvr/rlvr_lr5e6_scale0.5_mmmu_full900/scores.csv)

## 추가 실험

`ablations/`은 생성 길이, 프롬프트와 파서 영향을 확인한 진단용 결과입니다.

| 폴더 | 파서 | 문제 수 | 설정 | 정확도 |
| --- | --- | ---: | --- | ---: |
| `direct_prompt_128_full900/` | v2 | 900 | 직접응답 프롬프트, 128토큰 | 467/900 (51.89%) |
| `official_prompt_128_mc120/` | v2 | 120 | 공식 프롬프트, 128토큰 | 17/120 (14.17%) |
| `official_prompt_4096_mc120/` | v2 | 120 | 공식 프롬프트, 4,096토큰 | 67/120 (55.83%) |
| `official_prompt_32768_mc120/` | v2 | 120 | 공식 프롬프트, 32,768토큰 | 76/120 (63.33%) |
| `official_prompt_16384_full900_parser_v1/` | v1 | 900 | 공식 프롬프트, 16,384토큰 | 546/900 (60.67%) |

120문제 실험은 과목별 객관식 4개씩 동일한 문항을 사용했습니다. 이전 0.90
베이스라인에서 같은 120문제의 결과는 74/120(61.67%)입니다. 조건별 sampling을
한 번씩 실행했으므로 작은 점수 차이는 생성 길이만의 효과로 단정하지 않고,
16,384토큰 생성 예산을 정하기 위한 참고 결과로 사용했습니다.

`official_prompt_16384_full900_parser_v1/`은 파서 수정 전 결과입니다. 같은 응답을
v2 파서로 재채점하면 546개에서 563개로 17개 증가합니다. 이전 0.90 베이스라인은 파서
v2를 적용한 뒤 900문제 전체를 다시 생성해 얻었습니다.
