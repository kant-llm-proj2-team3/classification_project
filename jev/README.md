# Jev 계열
담당자: 송찬영
---
## 파일 구조
```text
/jev
├── scripts/
│   └── data_convert.py
└── steps/
```
---
## 폴더 설명
- `steps`: laya 학습 관련 디렉토리
- `scripts`: 기타 스크립트 디렉토리
### 스크립트 설명
#### `steps`
#### `scripts`
- `data_convert.py`: laya학습을 위한 데이터 변환 스크립트로, 심화 데이터와 일반 데이터를 구분해서 한 번에 변환해준다.
##### 사용법
```commandline
# 일반 데이터들 변환
uv run python ./jev/scripts/data_convert.py
# 심화 데이터들 변환
uv run python ./jev/scripts/data_convert.py --advanced true
```