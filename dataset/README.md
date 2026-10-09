# **데이터셋 구성 및 결과**

## 학습 데이터 수량
| 구분 | 주제 | 기존 학습 데이터 | 추가 데이터 | 최종 합본 | 라벨별 수량 |
|:---:|:---:|---:|---:|---:|:---|
| project2-data | documents | 360 | 300 | **660** | 공지·매뉴얼·회의록 각 220 |
| project2-data | inquiries | 360 | 300 | **660** | 배송·환불·계정 각 220 |
| project2-advanced-data | documents | 180 | 150 | **330** | 공지·매뉴얼·회의록 각 110 |
| project2-advanced-data | inquiries | 180 | 150 | **330** | 배송·환불·계정 각 110 |

### project2-data 경우 기존 학습 데이터와 기존 학습데이터에서 난도를 높인 합성데이터를 추가하였습니다. 고객문의 데이터는 주된 요청을, 문서 데이터에서는 글의 목적을 기준으로 분류를 하도록 구성하였습니다. 
### project2-advanced-data 경우에 project2-data처럼 기존 학습데이터에 난도를 높인 합성데이터를 추가하였으며 추가한 데이터는 종료된 상담이나 참고자료와 현재 본문을 구분해야 하는 사례들이 포함되어 있습니다. 

## 검사결과 

모두 
- 기존 데이터와 추가 데이터의 동일 문장·ID·그룹 겹침
- 분할 간 동일 문장 및 그룹 겹침
- 동일 문장의 라벨 충돌
- 빈 문장, 잘못된 라벨 및 관리용 ID 누락
상단에 있는 검사를 통과 하였습니다. 이때 라별 경우는 project2-data의 데이터 경우에서는 같은 그룹의 같은라벨이여야 하지만 project2-advanced-data 데이터 경우에서는 같은 상황에서의 서로 다른 요청.문서 목적이 묶여서 있기에 같은 그룹에 여러 라벨이 있을 수 있습니다. 
동일문장 검사는 공백과 문장부호를 제거한 오직 텍스트 기준으로 하였으며 의미가 비슷한 사례를 판별하거나 오염가능성을 판별하는 검사는 하지 않았습니다. 

## 디렉토리 구성도

### project2-data 디렉토리 구성

```text
project2-data/data/
├── prepare_dataset.py         # project2-data 데이터의 중복·오류 검사 및 학습용 합본 CSV 생성
├── documents/                 # 기존 문서 데이터 + add_train.csv
├── inquiries/                 # 기존 문의 데이터 + add_train.csv
└── dataset_result_data/ # project2-data 기존·추가 학습 데이터를 합친 결과 폴더
    ├── documents_train.csv # 문서 학습 데이터 합본 (660행)
    └── inquiries_train.csv # 고객 문의 학습 데이터 합본 (660행)
```

### project2-advanced-data 디렉토리 구성

```text
project2-advanced-data/data/    
├── prepare_advanced.py        # project2-advanced-data 데이터의 중복·오류 검사 및 학습용 합본 CSV 생성
├── documents/                 # 기존 문서 데이터 + add_train.csv
├── inquiries/                 # 기존 문의 데이터 + add_train.csv
└── dataset_result_advanced_data/ # project2-advanced-data 기존·추가 학습 데이터를 합친 결과 폴더
    ├── documents_train.csv # 문서 학습 데이터 합본 (330행)
    └── inquiries_train.csv # 고객 문의 학습 데이터 합본 (330행)
```

## 모델 담당자 사용 방법

- project2-data은 dataset_result_basic의 주제별 학습 CSV를 사용합니다.
- project2-advanced-data는 dataset_result_advanced의 주제별 학습 CSV를 사용합니다.
- 합본에는 추가 데이터가 이미 포함되어 있으므로 다시 합치지 않습니다.
- 검증·테스트·생성용 파일은 기존 파일을 그대로 사용합니다.
- 모델 입력에는 text만 사용하고, label은 정답으로 사용합니다.
- id, group_id, source는 관리용이며, source로 기존·추가 데이터의 출처를 구분합니다.