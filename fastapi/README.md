# FastAPI 서버

문서/문의 텍스트 한 건을 받아 **분류 → 정책 적용 → 답변 생성 → 응답 조립**을 수행하는 API 서버입니다.
요청 추적(SQLite), 추적 조회용 `/admin` API, 생성 모델 비교 평가 도구도 이 디렉토리에 있습니다.

## 파일 구성

```text
fastapi/
├── main.py                  앱 생명주기(모델 로딩), /health, /chat, /compare
├── chain.py                 LCEL 체인: classify → apply_policy → generate → respond
├── classifier.py            토픽별 분류 모델 선택·로딩·추론 (recommendation.json 또는 .env 지정)
├── classifier_backends.py   백엔드: LayaBackend, ClmBackend, HfBackend(bert/roberta/koelectra)
├── policy.py                라벨별 정책, 토픽별 프롬프트, 요약 항목 구성 검사
├── generator.py             생성 모델: OpenAIGenerator, OllamaGenerator
├── admin.py                 추적 조회·집계·내보내기 API (/admin/*)
├── admin_auth.py            /admin 서명 인증 (HMAC-SHA256)
├── tracking/                요청 추적
│   ├── middleware.py        모든 요청의 시작·종료·상태 기록, X-Request-ID 발급
│   ├── callbacks.py         LangChain 콜백으로 단계별 span 수집
│   ├── context.py           요청 단위 RequestRecord (ContextVar)
│   └── store.py             SQLite 저장소 (requests, spans, 집계용 뷰)
└── evaluation/              생성 모델 비교, 서빙 경로 점검
    ├── compare.py · report.py · serving_check.py · eval_common.py
```

## 실행

저장소 루트에서 실행합니다. `main.py`에 서버 시작 코드가 없으므로 uvicorn으로 띄웁니다.

```powershell
uv sync
uv run uvicorn main:app --app-dir fastapi --host 127.0.0.1 --port 8000
```

- API 문서: `http://127.0.0.1:8000/docs`
- 시작할 때 토픽(`documents`, `inquiries`)마다 분류 모델을 한 번 불러옵니다. 로딩에 실패한 토픽은 서버가 죽지 않고 `/health`에 `error`로 표시되며 해당 토픽은 분류하지 않습니다.
- Ollama는 시작 때 모델을 미리 올립니다(`OLLAMA_PRELOAD`, 기본 켜짐). 첫 요청의 로딩 지연이 비교를 왜곡하지 않게 하기 위해서입니다.
- 필요한 설정은 루트의 `.env`(예시: `.env.example`)에 둡니다. API 키는 응답·로그·추적에 남지 않습니다.

## API

| 엔드포인트 | 설명 |
|---|---|
| `GET /health` | 상태, 토픽별 분류 모델(키·run 이름·오류), 생성 모델 설정 |
| `POST /chat` | 분류 후 생성 모델 **하나**(`provider`: `openai`(기본) / `ollama`)로 답변 생성 |
| `POST /compare` | 분류·프롬프트는 한 번만 만들고 같은 프롬프트를 두 생성 모델에 동시에 보내 결과를 나란히 반환. 한 쪽이 실패해도 나머지는 반환(`partial_success`) |
| `/admin/*` | 추적 조회(읽기 전용). 아래 참고 |

요청 형식:

```json
{ "text": "배송 현황을 확인하고 싶어요.", "dataset": "inquiries", "provider": "openai" }
```

| 필드 | 규칙 |
|---|---|
| `text` | 공백이 아닌 1~10,000자 (앞뒤 공백 제거) |
| `dataset` | `documents` 또는 `inquiries` (`topic`도 허용) |
| `provider` | `/chat`만. 생략하면 `openai` |

단계가 실패해도 예외 대신 단계별 `status`(`success` / `error` / `timeout` / `skipped`)를 담아 응답합니다. 응답에는 `classification`(라벨, 확률, 신뢰도, 추론 시간, 입력 토큰, 잘림 여부)과 `generation`이 들어갑니다.

## 처리 흐름

```text
classify      분류 모델로 라벨 예측 (전체 원문 사용, 별도 스레드)
apply_policy  라벨별 정책으로 프롬프트 생성 (분류 실패·정책 없음이면 생략)
generate      생성 모델 호출 (원문은 MAX_INPUT_CHARS까지만 전달)
respond       API 응답으로 정리
```

| 토픽 | 생성 결과 |
|---|---|
| `inquiries` | `answer` + `next_steps` (다음 행동 안내) |
| `documents` | `answer`(개요) + `summary_items`(라벨별 항목형 요약: 공지=대상·일정·해야 할 일, 매뉴얼=전제 조건·순서·주의 사항, 회의록=논의·결정·담당자·후속 조치). 항목 구성 준수 여부는 `metrics.summary_structure_ok`로 기록 |

정책(`policy.py`)은 교육용 가상 기준이며 실제 주문 조회·환불 승인 도구는 연결되어 있지 않습니다.

## 분류 모델 선택

서비스 모델은 `artifacts/<토픽>/advanced_merge/recommendation.json`의 `recommended_model`을 따르고, 같은 `advanced_merge` 아래 `best_model/`을 불러옵니다.
`.env`의 `CLASSIFIER_MODEL_DOCUMENTS` / `CLASSIFIER_MODEL_INQUIRIES`에 `laya`, `clm`, `bert`, `roberta`, `koelectra` 중 하나를 적으면 바꿀 수 있습니다. `CLASSIFIER_DEVICE`는 비우면 CUDA(없으면 CPU)입니다.
`clm`은 외부 llama.cpp 인코더 서버가 필요합니다([jev/README.md](../jev/README.md)).

## 요청 추적과 /admin

- 모든 요청은 `X-Request-ID`를 받고, 단계별 호출 흐름(span)과 함께 `logs/tracking.db`(SQLite, WAL)에 저장됩니다. 쓰기는 큐 + 백그라운드 스레드로 처리해 요청을 막지 않고, 큐가 차면 기록을 버립니다.
- 요청 원문과 답변은 `requests` 테이블에만 저장됩니다. `TRACKING_STORE_TEXT=0`이면 글자 수와 해시만 남기고, `TRACKING_RETENTION_DAYS`로 보관 기간을 정할 수 있습니다. (DB는 Git에서 제외)
- `/admin` 엔드포인트: `auth-info`, `auth-check`, `requests`(목록), `requests/{id}`(호출 흐름), `stats`(집계), `export`(csv/json/jsonl).
- `TRACKING_ADMIN_PASSWORD`가 비어 있으면 인증 없이 열리고, 값이 있으면 모든 `/admin` 요청에 HMAC-SHA256 서명 헤더(`X-Admin-Timestamp`, `X-Admin-Nonce`, `X-Admin-Signature`)가 필요합니다. 암호 자체는 네트워크로 보내지 않으며, 클라이언트 서명 계산은 `chatbot/admin_client.py`와 같은 식입니다. 응답 내용까지 보호하려면 HTTPS가 필요합니다.

## 평가 도구 (`evaluation/`)

모두 저장소 루트에서 실행합니다.

**생성 모델 비교 (`compare.py` → 사람 채점 → `report.py`)**
```powershell
uv run python fastapi/evaluation/compare.py --per-kind 3 --dry-run           # 표본·호출 수만 확인 (호출 없음)
uv run python fastapi/evaluation/compare.py --per-kind 10 --repeats 1 --yes   # 실제 실행 (OpenAI 유료 호출 포함)
uv run python fastapi/evaluation/report.py artifacts/generation_compare/<이름>
```
- 입력은 `dataset/<토픽>/<종류>/test.csv`에서 종류(basic, basic_add, advanced, advanced_add)마다 `--per-kind`건을 시드로 뽑습니다.
- 서비스와 같은 분류·정책·생성 코드를 씁니다. 결과는 `artifacts/generation_compare/<이름>/`에 `results.jsonl`, `config.json`, `summary.json`, `human_scores.csv`(모델 숨김 채점표), `answer_key.csv`가 생기고, 채점 후 `report.py`가 `report.md/json`, `scores_long.csv`를 만듭니다.
- test 데이터를 쓰므로 결과를 보고 정책·프롬프트를 고친 뒤 같은 test로 다시 비교하면 독립 평가가 아닙니다. (`--allow-changed-prompt` 필요)

**서빙 경로 점검 (`serving_check.py`)**
```powershell
uv run python fastapi/evaluation/serving_check.py                              # validation 점검 (반복 가능)
uv run python fastapi/evaluation/serving_check.py --split test --confirm-test  # test는 모델마다 1회
```
서버가 쓰는 로딩·입력·예측 경로가 학습 때 저장된 예측과 같은 라벨·확률을 내는지 확인합니다.

## 부하 테스트
[locust/README.md](../locust/README.md)를 보세요.
