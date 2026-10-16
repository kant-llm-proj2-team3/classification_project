# 분류 데이터

폴더는 `토픽 / 데이터 종류` 두 단계입니다.
토픽은 `inquiries`(문의)와 `documents`(문서)이며 두 토픽의 구조는 같습니다.

| 토픽 | 라벨 |
|---|---|
| `documents` | `announcement`, `manual`, `minutes` |
| `inquiries` | `shipping`, `refund`, `account` |

```text
dataset/
├── inquiries/
│   ├── basic/  basic_add/  basic_merge/
│   ├── advanced/  advanced_add/  advanced_merge/
│   └── total_merge/
├── documents/
│   └── (inquiries와 같은 구조)
├── scripts/
│   └── prepare_dataset.py
└── README.md
```

| 폴더 | 내용 | train | validation | test |
| --- | --- | ---: | ---: | ---: |
| basic | 일반 원본 | 360 | 120 | 120 |
| basic_add | 일반 추가본만 | 300 | 100 | 100 |
| basic_merge | 일반 원본 + 추가본 | 660 | 220 | 220 |
| advanced | 심화 원본 | 180 | 60 | 60 |
| advanced_add | 심화 추가본만 | 150 | 51 | 51 |
| advanced_merge | 심화 원본 + 추가본 | 330 | 111 | 111 |
| total_merge | basic_merge + advanced_merge | 990 | 331 | 331 |

수량은 각 토픽의 데이터 행 수이며 CSV 헤더는 제외합니다.
`add`는 추가본, `merge`는 합본입니다. `total_merge`는 서비스에 줄 최종 모델 후보를 학습하는 데이터입니다.

## 파일

각 데이터 폴더에는 동일한 이름의 파일이 있습니다.

- `train.csv`, `validation.csv`, `test.csv` — 열: `id`, `text`, `label`, `group_id`, `source`
- `train.jsonl`, `validation.jsonl`, `test.jsonl` — laya/CLM 학습용 형식. (`jev/scripts/util/data_convert.py`가 CSV에서 변환)
- `labels.json` — 라벨 순서와 설명. 학습 코드가 이 순서를 기준으로 삼습니다.

`basic`, `basic_merge`, `advanced`, `advanced_merge`에는 생성 평가용 파일도 있습니다.

- `generation_development.jsonl` — 정책·프롬프트를 다듬을 때 보는 개발용 입력 (토픽당 약 15건)
- `generation_eval.jsonl` — 생성 품질 평가용 입력 (`id`, `text`, `expected_label`, `group_id` 등, 약 24건)

> 생성 모델 비교(`fastapi/evaluation/compare.py`)는 위 파일이 아니라 각 종류의 `test.csv`에서 표본을 뽑아 씁니다.

## 데이터 검증·재생성

`prepare_dataset.py`는 basic/advanced 원본과 추가본의 구조·분할 크기·중복(id, group_id, 문장)을 검사하고 `*_merge`의 CSV를 다시 만듭니다.

```commandline
uv run python dataset/scripts/prepare_dataset.py --check-only          # 쓰지 않고 검증만
uv run python dataset/scripts/prepare_dataset.py --variant basic       # basic 계열만 (basic | advanced | all)
```

## 평가 규칙
- 학습은 `train`만 쓰고, 모델 선택은 `validation`으로 합니다. `test`는 최종 평가에서만 한 번 씁니다.
- 같은 `group_id`는 분할을 넘어 섞이지 않아야 합니다. (basic/advanced/merge 간에는 validation·test 구성이 달라 점수를 직접 비교하기 어렵습니다)
