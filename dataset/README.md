# 데이터셋 구성 및 결과

## 학습 데이터 수량

| 구분 | 주제 | 기존 학습 데이터 | 추가 데이터 | 최종 합본 | 라벨별 수량 |
|:---|:---|---:|---:|---:|:---|
| project2-data | documents | 360 | 300 | **660** | 공지·매뉴얼·회의록 각 220 |
| project2-data | inquiries | 360 | 300 | **660** | 배송·환불·계정 각 220 |
| project2-advanced-data | documents | 180 | 150 | **330** | 공지·매뉴얼·회의록 각 110 |
| project2-advanced-data | inquiries | 180 | 150 | **330** | 배송·환불·계정 각 110 |

project2-data의 경우 기존 학습 데이터와 별도로 작성한 합성 데이터를 추가하였습니다. 고객 문의 데이터는 주된 요청을, 문서 데이터는 글의 목적을 기준으로 분류하도록 구성하였습니다.

project2-advanced-data의 경우에도 기존 학습 데이터와 별도로 작성한 합성 데이터를 추가하였습니다. 기존 Advanced 데이터에는 종료된 상담이나 참고자료와 현재 요청·본문을 구분해야 하는 사례들이 포함되어 있습니다.

## 검사 결과

두 프로젝트 모두 아래에 있는 검사를 통과하였습니다.

- 기존 데이터와 추가 데이터의 동일 문장·ID·그룹 겹침
- 분할 간 동일 문장 및 그룹 겹침
- 동일 문장의 라벨 충돌
- 빈 문장, 잘못된 라벨 및 관리용 ID·group_id·source 누락
- 추가 데이터의 전체 수량 및 라벨별 수량
- 추가 데이터와 기존 전체 데이터, 추가 데이터 내부의 문자 유사도 0.90 이상 후보 검사

이때 라벨의 경우 project2-data에서는 같은 그룹의 라벨이 같아야 하지만, project2-advanced-data에서는 같은 상황에서의 서로 다른 요청이나 문서 목적이 묶여 있기에 같은 그룹에 여러 라벨이 있을 수 있습니다. 다만 두 프로젝트 모두 같은 그룹이 서로 다른 분할에 겹치는 것은 허용하지 않습니다.

동일 문장 검사는 공백과 문장부호 제거 등을 포함한 코드의 정규화 규칙을 적용한 텍스트 기준으로 하였습니다. 문자 유사도 검사도 진행하였으나, 이 검사만으로 의미가 비슷한 사례나 모든 데이터 오염 가능성을 판별할 수는 없습니다.

## 디렉토리 구성

### project2-data 디렉토리 구성

기준 경로는 `project2-data/data/`입니다.

| 파일·폴더 | 내용 |
|:---|:---|
| `prepare_dataset.py` | 중복·오류 검사 및 학습용 합본 CSV 생성 |
| `documents/` | 기존 문서 데이터 및 추가 데이터 `add_train.csv` |
| `inquiries/` | 기존 문의 데이터 및 추가 데이터 `add_train.csv` |
| `dataset_result_data/documents_train.csv` | 문서 학습 데이터 합본 660행 |
| `dataset_result_data/inquiries_train.csv` | 고객 문의 학습 데이터 합본 660행 |

### project2-advanced-data 디렉토리 구성

기준 경로는 `project2-advanced-data/data/`입니다.

| 파일·폴더 | 내용 |
|:---|:---|
| `prepare_advanced.py` | 중복·오류 검사 및 학습용 합본 CSV 생성 |
| `documents/` | 기존 문서 데이터 및 추가 데이터 `add_train.csv` |
| `inquiries/` | 기존 문의 데이터 및 추가 데이터 `add_train.csv` |
| `dataset_result_data_advanced/documents_train.csv` | 문서 학습 데이터 합본 330행 |
| `dataset_result_data_advanced/inquiries_train.csv` | 고객 문의 학습 데이터 합본 330행 |

## 실행 방법

각 프로젝트의 `data` 폴더에서 실행합니다.

project2-data의 경우:

```powershell
uv run python .\prepare_dataset.py
```

project2-advanced-data의 경우:

```powershell
uv run python .\prepare_advanced.py
```

검사를 통과하면 해당 결과 폴더에 합본 CSV가 생성됩니다. 다시 실행하면 같은 경로의 결과 CSV를 덮어씁니다. 매번 기존 학습 데이터와 추가 데이터를 합치기 때문에 추가 데이터가 누적되지는 않습니다.

원본 train·validation·test CSV와 생성용 JSONL은 수정하지 않습니다.

## 모델 담당자 사용 방법

- project2-data는 `dataset_result_data`의 주제별 학습 CSV를 사용합니다.
- project2-advanced-data는 `dataset_result_data_advanced`의 주제별 학습 CSV를 사용합니다.
- 합본에는 추가 데이터가 이미 포함되어 있으므로 다시 합치지 않습니다.
- 검증·테스트·생성용 파일은 각 프로젝트의 기존 파일을 그대로 사용합니다.
- 모델 입력에는 `text`만 사용하고, `label`은 정답으로 사용합니다.
- `id`, `group_id`, `source`는 관리용이며, `source`로 기존·추가 데이터의 출처를 구분합니다.