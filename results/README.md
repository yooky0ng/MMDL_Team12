# 평가 결과

## 최종 Baseline

`baseline/`에는 Qwen3-VL-4B-Instruct로 MMMU validation 900문제 전체를
평가한 최종 제출 결과가 들어 있다.

- 추론 백엔드: vLLM 0.14.0
- 프롬프트: Qwen MMMU 프롬프트
- `max_new_tokens`: 16,384
- 정확도: 546/900 (60.67%)
- 공식 수치 67.4와의 차이: -6.73%p
- 실행 시간: 3,708.2초 (약 61분 48초)
- peak GPU memory: 88,334 MiB
- 길이 제한 도달: 96/900
- 객관식 파싱 실패: 43/900
- Git에 포함하는 파일: `summary.json`, `scores.csv`
- 원본 응답인 `predictions.jsonl`은 로컬에 보관하고 `.gitignore`로 제외한다.

`summary.json`에는 실행 설정과 전체·과목별 결과가, `scores.csv`에는 과목별
정답 수와 정확도가 들어 있다.

## 프롬프트

최종 baseline은 `--prompt-style official`로 실행했으며, 객관식 문제에는 다음
템플릿을 사용했다.

```text
Question: {question}
Options:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}
Please select the correct answer from the options above.
```

프롬프트는 [Qwen3-VL 공식 MMMU 평가 코드의 `build_mmmu_prompt()`](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py#L23-L43)를
참고했다. 이미지는 텍스트와 별도의 multimodal content로 모델에 전달한다.

최초 직접응답 실험에서는 공식 문구 뒤에 다음 지시를 추가했지만, 최종
baseline에는 포함하지 않았다.

```text
Respond with only the option letter (for example, A). Do not include an explanation.
```

## 추가 실험

`ablations/`에는 공식 점수와의 격차를 분석하기 위해 실행한 추가 실험이
들어 있다. 이 결과들은 최종 baseline이 아니다.

- `direct_prompt_128_full900/`: 최초 직접응답 프롬프트, 900문제, 52.78%
- `official_prompt_128_mc120/`: 객관식 120문제, 128토큰, 14.17%
- `official_prompt_4096_mc120/`: 동일한 120문제, 4,096토큰, 54.17%
- `official_prompt_32768_mc120/`: 동일한 120문제, 32,768토큰, 59.17%

120문제 실험은 30개 과목에서 앞쪽 객관식 4개씩을 선택해 생성 길이의 영향을
비교한 진단용 subset이다. 생성 예산을 늘리면 정확도가 높아졌지만 실행 시간도
증가했기 때문에, 전체 900문제 baseline에는 16,384토큰을 사용했다.

최종 재현 명령과 전체 설정은 `reports/mmmu_baseline.md`에 기록한다.
