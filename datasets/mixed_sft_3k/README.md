# Mixed 30K 학습 데이터

Point-CoT 24,000개와 MMVU 6,000개를 합친 SFT 데이터입니다. [mixed_30k.json](mixed_30k.json)이 실제 학습 입력이고, [sample_manifest.jsonl](sample_manifest.jsonl)에 선별한 샘플이 기록돼 있습니다.

폴더 이름은 `mixed_sft_3k`이며, 실험 2의 실제 학습 규모는 30,000개입니다.

Point-CoT와 MMVU 샘플은 혼합 JSON의 `source` 필드로 확인할 수 있습니다.
