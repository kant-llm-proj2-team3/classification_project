# Chatbot (Streamlit UI)

FastAPI 서버([fastapi/README.md](../fastapi/README.md))를 호출하는 Streamlit 화면입니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `app.py` | 진입점. 페이지 라우팅(채팅 / 설정 / 트레이싱), 채팅 화면, API 호출 |
| `tracing_page.py` | 트레이싱 조회 페이지: 요청 목록, 호출 흐름(span), 통계, 파일 내보내기 |
| `admin_client.py` | `/admin` API 클라이언트. 암호 대신 HMAC-SHA256 서명 헤더를 보냄 |

## 실행

FastAPI 서버를 먼저 띄운 뒤, 저장소 루트에서 실행합니다.

```powershell
uv run uvicorn main:app --app-dir fastapi --host 127.0.0.1 --port 8000   # 터미널 1
uv run streamlit run chatbot/app.py                                       # 터미널 2
```

## 화면

- **채팅**: 데이터 종류(문서 / 고객 요청)와 생성 모델(OpenAI / 로컬 Ollama / 비교(둘 다))을 고르고 최대 10,000자까지 입력합니다. 비교는 `/compare`로 같은 입력을 두 모델에 보내 답변을 나란히 보여 줍니다. 단일 모델은 `/chat`을 씁니다.
- **설정**: API 사용 여부와 API 주소(기본 `http://127.0.0.1:8000`)를 지정합니다. API를 켜지 않으면 서버를 호출하지 않습니다.
- **트레이싱**: 기간·토픽·결과·라벨·생성 모델 등으로 요청을 필터링하고, 요청별 호출 흐름과 집계(지연 분포, 토큰, 오류, 생성 모델별 비교)를 보고 csv/json/jsonl로 내보냅니다.
  서버에 `TRACKING_ADMIN_PASSWORD`가 설정돼 있으면 이 페이지에서 암호를 입력해야 합니다. (암호는 전송하지 않고 서명만 전송)
