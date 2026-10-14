"""main.py에서 사용할 분류기 연결 인터페이스만 정의한다."""


def load_classifier() -> dict:
    # 서버 시작 시 사용할 주제별 모델 자리를 준비한다.
    # TODO: 선정 모델의 저장 형식이 확정되면 None을 실제 모델 객체로 교체한다.
    return {"documents": None, "inquiries": None}


def predict_text(text: str, bundle) -> dict:
    # 모델 연결 전에는 분류를 수행하지 않았다는 공통 결과를 반환한다.
    # TODO: 실제 추론 후 status="success", label, confidence를 반환한다.
    print(f"predict_text text는 : {text}")
    return {
        "status": "skipped",
        "label": None,
        "confidence": None,
    }
