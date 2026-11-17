# MMVU-Train 48K

Qwen3-VL-4B-Instruct의 LoRA SFT 실험을 위한 MMVU 공개 학습 데이터의 전처리 부분집합이다. 출처는 [BAAI MMVU](https://github.com/BAAI-DCAI/MMVU)의 [MMR-data](https://huggingface.co/datasets/BAAI/Multimodal-Robustness-Benchmark/tree/main/MMR-data)이며, 같은 약칭의 비디오 이해 데이터셋과 구분한다.

## 규모와 추출 방법

전처리한 원본 112,878개를 Train 107,246개와 Validation 5,632개로 분리한 뒤 Train에서 추출했다. **48K는 이미지 수가 아니라 대화 레코드 48,000개**를 뜻한다.

| 항목 | 검증 결과 |
| --- | ---: |
| Train 대화 레코드 | 48,000 |
| Train 질문·답변 쌍 | 327,501 |
| Train 고유 이미지 | 19,340 |
| Validation 대화 레코드 | 5,632 |
| Validation 고유 이미지 | 2,250 |
| Train/Validation 공통 이미지 경로 | 0 |

이미지 경로로 대화를 그룹화하고 정렬한 그룹을 Seed 42로 섞은 뒤, 목표 개수를 초과하지 않는 이미지 그룹 전체를 중복 없이 선택했다. 동일 이미지에 연결된 대화는 함께 선택하며 원래 레코드 순서와 모든 대화 턴을 보존했다. 제공된 추출 스크립트는 이 결정적 절차를 기록한다. 이번 업로드에서는 재생성하거나 재샘플링하지 않았다.

논문에서 사용한 48K의 정확한 샘플 목록과 선정 기준은 확인되지 않았으므로 논문과 동일한 샘플을 재현했다고 주장하지 않는다. Positive/Negative 개별 라벨도 아직 검증되지 않았으므로 클래스 비율이나 균형 샘플링을 주장하지 않는다. **MMMU Validation은 학습에 사용하지 않는다.** 이 폴더의 `val.json.gz`는 MMVU에서 분리한 검증 데이터다.

## 파일

- `train_48k.json.gz`: 학습용 JSON 배열. 각 레코드는 `image` 상대 경로와 `human`/`gpt`가 번갈아 등장하는 `conversations`를 포함한다.
- `val.json.gz`: 보류한 MMVU Validation JSON 배열.
- `manifest.json`: 개수, Seed, 추출 방식, 원본 JSON 및 압축 데이터의 SHA-256 기록.
- `selected_images.txt`: Train 고유 이미지 상대 경로의 정렬된 목록.
- `../../scripts/make_mmvu_48k.py`: 전처리된 전체 Train/Validation으로부터 추출하는 원본 스크립트. 전체 전처리 입력 JSON은 이 저장소에 포함하지 않으며, 재실행 시 manifest의 입력 해시와 일치하는 파일이 필요하다. 기본 경로는 원래 전처리 작업 디렉터리 기준이므로 다른 위치에서는 `--train`, `--val`, `--images-dir`, `--output-dir`를 지정한다.

## GPU 학습 준비

실제 이미지 파일은 Git 저장소에 포함하지 않는다. [공식 MMR-data 이미지 아카이브](https://huggingface.co/datasets/BAAI/Multimodal-Robustness-Benchmark/blob/main/MMR-data/MMR-data-images.zip)를 별도로 확보해야 한다. 이미지 참조와 `<image>` 토큰은 원본 그대로 보존했다. 예를 들어 `image` 값은 다음처럼 시작하는 **상대 경로**다.

```text
share/project/yexin/datasets/DataV2/MMR-data-images/coco_2017/000000263744.jpg
```

학습 로더의 이미지 루트 아래에 이 상대 경로가 해석되도록 디렉터리 또는 심볼릭 링크를 구성한다. 위 접두사는 원본 데이터의 상대 참조이며 개인 PC의 절대 경로가 아니다. Train과 Validation의 모든 참조 이미지가 실제로 존재하는지 GPU 학습 전에 확인한다. 이번 검증은 JSON과 참조 경로를 대상으로 했으며 이미지 파일의 존재·디코딩은 검증하지 않았다.

`.json.gz`는 압축 파일이다. 저장소 루트에서 다음과 같이 해제하고, 학습 프레임워크에 필요한 대화 형식으로 연결하거나 변환한다.

```python
import gzip
import shutil
from pathlib import Path

root = Path("datasets/mmvu_48k")
for name in ("train_48k", "val"):
    with gzip.open(root / f"{name}.json.gz", "rb") as src:
        with (root / f"{name}.json").open("wb") as dst:
            shutil.copyfileobj(src, dst)
```

압축 해제 파일과 이미지는 로컬 학습용으로만 보관한다. 모델/processor, Qwen3-VL 대화 템플릿과 `<image>` 처리, 이미지 루트, LoRA 및 GPU 학습 설정은 별도로 준비해야 한다. Validation은 학습 입력에 합치지 않는다.

업로드 전 Python으로 gzip/JSON, 대화 역할·QA 쌍, 이미지 경로·토큰, 이미지 분리, 이미지 목록 및 manifest 해시를 검증했다. 원본 전처리 Train/Validation의 입력 해시와 비교하고, 선택된 이미지 그룹의 모든 레코드 및 Validation이 원본과 동일한 것도 확인했다. 원본 파일의 바이트를 변경하지 않고 복사했다.
