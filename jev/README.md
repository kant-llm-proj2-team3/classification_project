# Jev 계열 (laya · CLM · jev)
담당자: 송찬영

언어 기반 분류 모델(System One 계열)의 학습·평가 디렉토리입니다. 결과는 baseline/BERT와 같은 형식으로 `artifacts/<토픽>/<데이터셋>/<모델>/`에 저장되어 같은 표·그래프·모델 추천에 들어갑니다.

| 모델 | 설명 | 학습 | 결과 폴더 |
|---|---|---|---|
| `laya` / `laya_zeroshot` | laya(multilingual, mmBERT 기반) 파인튜닝 / 학습 없이 평가 | 인코더 + 판단 헤드 전체 | `laya`, `laya_zeroshot` |
| `clm` / `clm_zeroshot` | CLM. 인코더는 Qwen3-8B Q4_K_M(고정)을 llama.cpp 서버로 구동, 판단 헤드(약 2천만 파라미터)만 학습 / 공식 헤드 그대로 평가 | 헤드만 | `clm`, `clm_zeroshot` |
| `jev` | TypeSafe System One API 호출 결과. laya·CLM의 baseline(참고용, 모델 추천에서 제외) | 없음 | `jev` |

## 파일 구조
```text
jev/
├── scripts/
│   ├── step01_read_data.py       JSONL 한 행 읽기·미리보기
│   ├── step02_check_data.py      모든 토픽/데이터셋 JSONL 구조·라벨·중복 검사
│   ├── step03_finetune_laya.py   laya 파인튜닝 동작 확인용 (결과는 jev/tmp)
│   ├── step04_train_laya.py      laya zero-shot 평가 + 파인튜닝 (artifacts 저장)
│   ├── step05_check_clm_env.py   CLM 구동·파인튜닝 가능한 환경인지 점검
│   ├── step05_train_clm.py       CLM zero-shot 평가 + 판단 헤드 파인튜닝
│   ├── step06_eval_jev.py        jev(TypeSafe API) 평가
│   ├── settings.py               laya 경로·설정 (import 시 모델을 내려받음)
│   ├── clm_settings.py           CLM 경로·설정 (.env의 CLM_* 값, 없으면 기본값)
│   ├── jev_settings.py           jev 설정 (.env의 JEV_* 값)
│   └── util/                     data_convert, clm_encoder, clm_head, jev_client, clm_reference.json 등
└── tmp/                          임시 산출물 (Git 제외): 임베딩 캐시, jev 응답 캐시, llama-server 로그
```
모든 명령은 저장소 루트에서 실행하며, Windows에서는 `PYTHONUTF8=1`을 권장합니다. 에폭·배치·조기 종료·데이터셋 ON/OFF는 루트의 `baseline_bert_config.py`에서 바꿉니다.

## 사용법

### 데이터 확인
```commandline
uv run python ./jev/scripts/step01_read_data.py
uv run python ./jev/scripts/step02_check_data.py
```
laya/CLM 학습용 JSONL은 `jev/scripts/util/data_convert.py`가 `dataset`의 CSV와 `labels.json`으로부터 만듭니다.
```commandline
uv run python ./jev/scripts/util/data_convert.py
uv run python ./jev/scripts/util/data_convert.py --advanced true   # 심화 데이터
```

### laya
```commandline
uv run python ./jev/scripts/step04_train_laya.py                                  # ON인 데이터셋 전부 (zero-shot → 파인튜닝)
uv run python ./jev/scripts/step04_train_laya.py --topic inquiries --dataset basic
uv run python ./jev/scripts/step04_train_laya.py --zero-shot-only
uv run python ./jev/scripts/step04_train_laya.py --test                           # best_model로 test 평가 (모델마다 1회)
```

### CLM
인코더(llama.cpp `llama-server`, Qwen3-8B Q4_K_M GGUF)는 프로젝트 밖(`CLM_TOOLS_DIR`, 기본 `~/tools`)에 둡니다. 코드가 떠 있는 서버를 찾아 쓰거나 직접 띄웁니다. `best_model`에는 판단 헤드(`head.safetensors`)와 인코더 정보(`encoder.json`)만 저장됩니다.
```commandline
uv run python ./jev/scripts/step05_check_clm_env.py            # 환경 점검 (--static: 서버 없이 파일·설치만, --for run: 구동만)
uv run python ./jev/scripts/step05_train_clm.py                # ON인 데이터셋 전부
uv run python ./jev/scripts/step05_train_clm.py --topic documents --dataset basic
uv run python ./jev/scripts/step05_train_clm.py --test         # best_model로 test 평가 (모델마다 1회)
```

### jev (TypeSafe API, 유료 호출)
API 키는 `.env`의 `JEV_API_KEY`에 둡니다. 같은 문장은 데이터셋이 달라도 한 번만 호출하고(`jev/tmp/jev_responses` 캐시), 중간에 끊겨도 이어서 할 수 있습니다.
```commandline
uv run python ./jev/scripts/step06_eval_jev.py --check      # 키·연결만 확인 (문장 전송 없음)
uv run python ./jev/scripts/step06_eval_jev.py --dry-run    # 요청 수와 예상 비용만 출력
uv run python ./jev/scripts/step06_eval_jev.py              # ON인 데이터셋의 validation 평가
uv run python ./jev/scripts/step06_eval_jev.py --test       # best_model로 test 평가 (모델마다 1회)
```

## 결과 요약 (`advanced_merge`, Macro-F1)
| 모델 | documents (val / test) | inquiries (val / test) |
|---|---|---|
| laya_zeroshot | 0.377 / 0.488 | 0.588 / 0.533 |
| laya | 1.000 / 0.973 | 1.000 / 1.000 |
| clm_zeroshot | 0.412 / 0.379 | 0.410 / 0.357 |
| clm | 0.937 / 0.928 | 0.964 / 0.910 |
| jev | 0.928 / 0.919 | 0.982 / 0.982 |

laya는 inquiries의 서비스 분류 모델로 선정되어 FastAPI가 사용합니다. CLM을 서비스에서 쓰려면 llama.cpp 인코더 서버가 함께 떠 있어야 합니다.

## 규칙
- test는 모델마다 한 번만 평가하고, 점수를 보고 모델·설정을 다시 고르지 않습니다.
- 학습 결과·로그는 `artifacts/`와 `logs/`에 남고, 가중치 파일은 Git에 올라가지 않습니다.
