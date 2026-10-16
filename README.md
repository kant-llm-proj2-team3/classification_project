# classification_project
## KANT 프로젝트2 - 3조(좋은 팀 이름 제안해 조)

문서(`documents`)와 고객 문의(`inquiries`) 텍스트를 3개 라벨로 분류하고, 분류 결과에 맞는 정책으로 답변(문의) 또는 요약(문서)을 생성하는 챗봇 서비스입니다.
여러 분류 모델(baseline, BERT 계열, laya, CLM, jev)을 같은 데이터로 비교해 서비스에 쓸 모델을 고르고, FastAPI + Streamlit으로 서비스합니다.

| 토픽 | 라벨 | 서비스에서 하는 일 |
|---|---|---|
| `documents` (문서) | `announcement`, `manual`, `minutes` | 문서를 항목형으로 요약 (`answer` + `summary_items`) |
| `inquiries` (문의) | `shipping`, `refund`, `account` | 문의에 답하고 다음 행동 안내 (`answer` + `next_steps`) |

## 팀원
- 권오륜 - [https://github.com/Riddlerio](https://github.com/Riddlerio)
- 송찬영 - [https://github.com/scy0416](https://github.com/scy0416)
- 이정원 - [https://github.com/piglove06](https://github.com/piglove06)
- 최선영 - [https://github.com/chsopo89](https://github.com/chsopo89)

## 전체 흐름

```text
dataset ──► baseline / bert / jev(laya, clm) 학습·평가 ──► artifacts ──► recommendation.json
                                                                              │ (서비스 모델 선정)
사용자 ──► chatbot (Streamlit) ──► fastapi (/chat, /compare) ──► 분류(classifier) ──► 정책·프롬프트(policy)
                                        │                                           ──► 생성(OpenAI / Ollama)
                                        └─► tracking (SQLite, logs/tracking.db) ◄── /admin ◄── 트레이싱 페이지
```

## Quick Start

### 1. 환경 설정
Python 3.12, [uv](https://docs.astral.sh/uv/)가 필요합니다. PyTorch는 CUDA 12.8 빌드(`pytorch-cu128`)를 사용합니다.

```commandline
uv sync
```

`.env.example`을 복사해 `.env`를 만들고 필요한 값을 채웁니다. (`.env`는 Git에 올라가지 않습니다)

| 용도 | 주요 키 |
|---|---|
| 답변 생성 (OpenAI) | `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_REASONING_EFFORT` |
| 답변 생성 (로컬 Ollama, 비교용) | `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_NUM_CTX`, `OLLAMA_PRELOAD` |
| 서비스 분류 모델 지정 | `CLASSIFIER_MODEL_DOCUMENTS`, `CLASSIFIER_MODEL_INQUIRIES`, `CLASSIFIER_DEVICE` (비우면 추천 결과를 따름) |
| 요청 추적 | `TRACKING_STORE_TEXT`, `TRACKING_DB_PATH`, `TRACKING_RETENTION_DAYS`, `TRACKING_ADMIN_PASSWORD` |
| jev API 평가 | `JEV_API_KEY`, `JEV_MODEL` |
| CLM 인코더 | `CLM_TOOLS_DIR`, `CLM_REPO`, `CLM_LLAMA_SERVER`, `CLM_ENCODER_URL` 등 |

### 2. 서비스 실행
서비스 분류 모델은 학습 결과(`artifacts/<토픽>/advanced_merge/<모델>/best_model`)가 있어야 불러올 수 있습니다. 모델 가중치(`*.safetensors`, `*.joblib`, `*.pt`)는 Git에 올라가지 않으므로 별도로 받아 해당 위치에 둬야 합니다.

```powershell
# 터미널 1: API 서버
uv run uvicorn main:app --app-dir fastapi --host 127.0.0.1 --port 8000
# 터미널 2: 챗봇 UI
uv run streamlit run chatbot/app.py
```

챗봇 설정 페이지에서 API 사용을 켜고 주소(`http://127.0.0.1:8000`)를 지정합니다. 자세한 내용은 [fastapi/README.md](fastapi/README.md), [chatbot/README.md](chatbot/README.md)를 보세요.

### 3. 모델 학습·평가
```commandline
# baseline + bert + roberta + koelectra 학습/추천/test/그래프 (입력 메뉴)
uv run python baseline/steps/run_menu.py
```
laya / CLM / jev 계열은 [jev/README.md](jev/README.md)를 보세요.

## 프로젝트 구조
```text
classification_project/
├── baseline/            TF-IDF baseline, 학습 공통 코드, 메뉴(run_menu.py), 추천·그래프
├── bert/                BERT / RoBERTa / KoELECTRA 학습 스크립트
├── jev/                 laya, CLM 학습·평가, jev(TypeSafe API) 평가
├── dataset/             토픽별 분류 데이터, 생성 평가 데이터, 데이터 검증 스크립트
├── fastapi/             API 서버 (분류 → 정책 → 생성), 요청 추적, /admin, 생성 모델 비교 평가
├── chatbot/             Streamlit UI (채팅, 설정, 트레이싱 조회)
├── locust/              /chat 부하 테스트 시나리오와 질문 CSV
├── scripts/             부하 테스트 실행·보고서 스크립트
├── artifacts/           학습·평가 결과 (모델별 best_model, 추천, 그래프, 생성 모델 비교)
├── logs/                실행 로그, 요청 추적 DB (Git 제외)
├── baseline_bert_config.py   학습·추천 공통 설정 (데이터셋 ON/OFF, 학습률, 외부 모델 등)
├── .env.example · pyproject.toml · uv.lock · .python-version
└── README.md
```

### 디렉토리 설명
| 디렉토리 | 설명 | 관리책임자 | 문서 |
|---|---|---|---|
| `baseline`, `bert` | baseline·BERT 계열 학습, 모델 추천, test 평가, 시각화 | 이정원 | [baseline](baseline/README.md), [bert](bert/README.md) |
| `jev` | laya·CLM 파인튜닝, jev API 평가 (언어 기반 분류 모델) | 송찬영 | [jev](jev/README.md) |
| `fastapi`, `chatbot` | 서비스 서버, 요청 추적, UI, 생성 모델 비교 평가 | 권오륜 | [fastapi](fastapi/README.md), [chatbot](chatbot/README.md) |
| `dataset` | 데이터셋 보관·검증 | 최선영 | [dataset](dataset/README.md) |
| `locust`, `scripts` | 부하 테스트 | - | [locust](locust/README.md) |
| `artifacts` | 학습·평가 산출물 | - | [artifacts](artifacts/README.md) |

## 분류 모델 비교 결과 요약
`advanced_merge`(심화 원본 + 추가본) 기준, test Macro-F1입니다. 자세한 내용과 다른 데이터셋 결과는 [artifacts/README.md](artifacts/README.md)와 `artifacts/viz/charts/`를 보세요.

| 모델 | documents | inquiries | 비고 |
|---|---:|---:|---|
| baseline (TF-IDF + 로지스틱 회귀) | 0.875 | 0.809 | 참고용 |
| bert (klue/bert-base) | 1.000 | 0.973 | |
| roberta (klue/roberta-base) | 0.991 | 0.982 | |
| koelectra (small-v3) | 0.991 | 1.000 | |
| laya (파인튜닝) | 0.973 | 1.000 | |
| clm (판단 헤드 파인튜닝) | 0.928 | 0.910 | |
| jev (TypeSafe API, 학습 없음) | 0.919 | 0.982 | 참고용 |

서비스 기본 모델은 `artifacts/<토픽>/advanced_merge/recommendation.json`의 `recommended_model`입니다. (현재 documents: bert, inquiries: laya. 상위 모델이 test에서 동점·만점이라 변별력이 낮으므로 test 점수가 모델 선택에 쓰였다는 점을 감안해야 합니다)

## 생성 모델 비교 (OpenAI vs 로컬 Ollama)
같은 분류 결과·프롬프트로 두 생성 모델을 호출하고 사람이 0~2점으로 채점했습니다. (test에서 종류별 10건, 총 160호출)
openai 총점 평균 7.41/8, ollama 6.78/8로 openai가 높았고, ollama는 오류 1건·없는 사실 0점 5건이 있었습니다.
상세는 `artifacts/generation_compare/compare_20261015_113659/report.md`, 실행 방법은 [fastapi/README.md](fastapi/README.md)를 보세요.
