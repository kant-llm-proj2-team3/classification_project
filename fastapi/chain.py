"""main.py와 분류기, 향후 모델 호출을 연결하는 LangChain 뼈대."""

from langchain_core.runnables import Runnable, RunnableLambda

from classifier import predict_text


def create_chain(classifiers: dict) -> Runnable:
    # main.py가 전달한 주제별 모델을 사용하는 처리 순서를 조립한다.
    def classify(item: dict) -> dict:
        print(f"create_chain의 classify item은 : {item}")
        # 이번 요청의 주제·원문을 유지하면서 분류 결과를 다음 단계에 전달한다.
        classification = predict_text(item["text"], classifiers[item["dataset"]])
        return {**item, "classification": classification}

    def generate(item: dict) -> dict:
        # 현재는 프론트 연결 확인용 고정 안내문을 반환하고 실제 모델은 호출하지 않는다.
        # TODO: 선정 모델에 맞게 정책·프롬프트·호출을 연결해 generation을 채운다.
        topic_name = {"documents": "문서", "inquiries": "고객 요청"}[item["dataset"]]
        return {
            "status": "partial_success",
            "mode": "stub",
            "classification": item["classification"],
            "generation": {
                # 기존 Streamlit이 안내문을 표시하도록 success를 사용한다.
                # 실제 생성 성공 여부는 모델 연결 단계에서 판정하도록 교체한다.
                "status": "success",
                "answer": (
                    f"{topic_name} 입력을 받았습니다. "
                    "현재는 API 연결 확인용 기본 응답입니다. "
                    "실제 분류와 답변 생성은 아직 연결되지 않았습니다."
                ),
                "next_steps": [],
            },
        }

    return RunnableLambda(classify) | RunnableLambda(generate)
