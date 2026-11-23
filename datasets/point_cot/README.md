# Point-CoT 전처리 과정

## 1. 목표

Point-RFT의 Point-CoT 데이터셋을 Qwen3-VL-4B-Instruct의 SFT 학습에 사용할 수 있는 형식으로 변환한다.

원본 `Point-CoT.json`은 보존하고, 최종 학습 데이터는 별도 파일로 관리한다.

## 2. 원본 데이터 구조

주요 필드는 다음과 같다.

- `image`: 이미지 경로
- `source`: 데이터 출처
- `intra_source_id`: 원본 데이터셋 내부 ID
- `question`: 질문
- `original_answer`: 최종 정답
- `steps_text`: 단계별 추론
- `steps_point`: 단계별 point 좌표 목록
- `steps_point_description`: point가 가리키는 시각적 요소 설명

전체 샘플 수는 **71,156개**다.

## 3. 품질 검사 결과

- 필수 필드 누락: 없음
- `steps_text` 형식 오류: 없음
- `steps_point` 형식 오류: 없음
- reasoning 단계와 point 단계 수 불일치: 없음
- 빈 `original_answer`: 214개
- DVQA grounding 실패: 17개
- PlotQA grounding 실패: 29개

## 4. 이미지 확보 및 연결

DVQA, PlotQA, ChartQA, TallyQA, MathV360K 계열의 이미지를 확보하고 Point-CoT 경로와 연결했다. Parquet 내부 이미지 바이트는 로컬 이미지 파일로 추출했다.

최종 학습 대상 70,896개에 대해 이미지 누락은 **0개**다.

## 5. 제외 기준

총 260개 샘플을 학습 대상에서 제외했다.

1. `original_answer`가 빈 샘플: 214개
2. DVQA point grounding 실패: 17개
3. PlotQA point grounding 실패: 29개

원본 `Point-CoT.json`은 삭제하지 않는다. 제외는 학습용 데이터 생성 단계에서만 적용한다.

## 6. 전처리 결과

- 원본 샘플: 71,156개
- 제외 샘플: 260개
- 최종 학습 샘플: **70,896개**
- 이미지 누락: **0개**

## 7. 좌표 처리 원칙

좌표는 임의로 보정하거나 정규화하지 않았다. 실제 이미지와 좌표가 맞지 않는 샘플은 제외하고, 정상 좌표는 원래 값과 설명을 유지했다.

## 8. Qwen3-VL 변환 형식

기존 Point-CoT 필드는 유지하고 각 샘플에 `qwen3vl` 필드를 추가했다.

```json
{
  "qwen3vl": {
    "image": "실제 로컬 이미지 경로",
    "conversations": [
      {"from": "human", "value": "<image>\\n질문"},
      {"from": "gpt", "value": "<think>추론과 좌표</think>\\n<answer>최종 답변</answer>"}
    ]
  }
}
```

## 9. 최종 학습 데이터

```text
processed/Point-CoT_qwen3vl_integrated.json
```

이 파일은 70,896개 샘플과 원본 Point-CoT 필드, Qwen3-VL용 대화 형식을 포함한다.

## 10. 변환 검증

- 변환 샘플 수: 70,896개
- 기존 필드 손실: 없음
- 대화 구조 오류: 0개
- 이미지 경로 누락: 0개
- 빈 assistant 응답: 0개

## 11. 관련 스크립트

```text
scripts/convert_qwen3vl.py
scripts/add_qwen3vl_integrated.py
```

## 12. 데이터 다운로드

대용량 JSON과 이미지 파일은 GitHub에 저장하지 않고 Google Drive에서 다운로드한다.

- [Point-CoT 데이터 및 이미지 다운로드](https://drive.google.com/file/d/1ss1fvp2s9S1rTIDzNEiQZvtcXFdYJVse/view?usp=sharing)

다운로드 후 다음 위치에 둔다.

```text
processed/Point-CoT_qwen3vl_integrated.json
source_images/
```

대용량 파일은 `.gitignore`로 Git 추적에서 제외한다.
