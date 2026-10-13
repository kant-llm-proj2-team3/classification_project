# 분류 데이터

폴더는 `토픽 / 데이터 종류` 두 단계입니다.
토픽은 `inquiries`(문의)와 `documents`(문서)이며 두 토픽의 구조는 같습니다.

```text
dataset/
├── inquiries/
│   ├── basic/
│   ├── basic_add/
│   ├── basic_merge/
│   ├── advanced/
│   ├── advanced_add/
│   └── advanced_merge/
├── documents/
│   ├── basic/
│   ├── basic_add/
│   ├── basic_merge/
│   ├── advanced/
│   ├── advanced_add/
│   └── advanced_merge/
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

수량은 각 토픽의 데이터 행 수이며 CSV 헤더는 제외합니다.
`add`는 추가본, `merge`는 합본입니다.

각 데이터 폴더에는 동일한 이름의 파일이 있습니다.

- `train.csv`, `validation.csv`, `test.csv`
- `train.jsonl`, `validation.jsonl`, `test.jsonl`
- `labels.json`



