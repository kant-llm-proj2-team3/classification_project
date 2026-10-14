# 분류 챗봇 FastAPI 기본 구조

Streamlit에서 보내는 질문 한 건을 받아 처리하는 API 뼈대입니다. 모델 연결 전에도 서버 실행과 요청·응답 형식을 확인할 수 있도록 고정 안내 응답을 반환합니다.

**현재 실제 분류·답변 생성·학습 모델 로딩은 구현하지 않았습니다.** API 키나 모델 주소 설정도 필요하지 않습니다.

## 파일 구성

```text
fastapi/
├── main.py          # 기존 파일: 요청 검증, 서버 생명주기, Depends, API
├── classifier.py    # 주제별 모델 로딩·분류 인터페이스와 TODO
├── chain.py         # 분류 → 생성 순서 연결과 고정 안내 응답
└── README.md
```

세 파일은 같은 FastAPI 서버에서 실행됩니다. 기존 `main.py`는 변경하지 않고 `chain.py`, `classifier.py`를 같은 폴더에 놓습니다. `step06`, `step08`, 별도 설정 파일에는 의존하지 않습니다.

## 실행

저장소 루트의 터미널에서 실행합니다. 필요한 API 라이브러리는 기존 `pyproject.toml`에 정의되어 있습니다.

```powershell
uv sync
uv run uvicorn main:app --app-dir fastapi --host 127.0.0.1 --port 8000 --reload
```

- API 문서: `http://127.0.0.1:8000/docs`
- 서버 상태: `GET /health`
- 질문 처리: `POST /chat`

`main.py`에는 직접 서버를 시작하는 코드가 없으므로 `python fastapi/main.py` 대신 위의 uvicorn 명령으로 실행합니다.

## 요청 흐름

```text
서버 시작
  → main.lifespan()
  → load_classifier()로 주제별 모델 자리 준비
  → create_chain(classifiers)로 처리 순서 조립

질문 요청
  → POST /chat
  → ChatRequest 입력 검증
  → Depends(get_chain)로 준비된 체인 전달
  → await chain.ainvoke({text, dataset})
  → classify(): predict_text()의 미연결 결과
  → generate(): 고정 안내 응답
  → JSON 응답
```

주제별 모델 자리는 서버 시작 시 한 번 준비합니다. 질문마다 체인을 실행하고, 이전 질문이나 답변을 체인에 저장하지 않습니다. 동기 처리 함수도 `ainvoke()`에서 실행 스레드로 처리하며, 한 질문의 단계 순서는 유지합니다.

## 요청 형식

```json
{
  "text": "배송 현황을 확인하고 싶어요.",
  "dataset": "inquiries"
}
```

| 필드 | 규칙 |
|---|---|
| text | 공백이 아닌 1~10,000자 문자열; 앞뒤 공백 제거 |
| dataset | documents 또는 inquiries |