"""main.py의 /chat 요청을 처리하는 LCEL 체인: 분류 -> 정책 적용 -> 답변 생성 -> 응답 조립.

각 단계가 이전 단계의 결과를 들고 다니는 딕셔너리(item)를 받아 키를 더해 넘긴다.
  입력  : {"text", "dataset"}
  classify     : + classification   (분류 모델, 동기 함수라서 ainvoke에서는 별도 스레드에서 실행된다)
  apply_policy : + prompt            (분류에 실패했거나 정책이 없으면 None)
  generate     : + generation        (OpenAI, 비동기. 프롬프트가 없으면 skipped)
  respond      : API 응답(status, classification, generation)으로 정리
Streamlit이 검사하는 응답 계약 때문에 단계가 실패해도 예외 대신 단계별 status(error/timeout/skipped)를 담아 돌려준다.
"""

import os

from classifier import predict_text
from generator import OpenAIGenerator
from langchain_core.runnables import Runnable, RunnableLambda
from policy import build_prompt

# 생성 모델에 보낼 원문의 최대 글자 수. 분류에는 전체 원문을 쓴다.
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS") or 4000)


def create_chain(classifiers: dict, generator: OpenAIGenerator | None = None) -> Runnable:
    generator = generator or OpenAIGenerator()

    def classify(item: dict) -> dict:
        text = item["text"].strip()
        classification = predict_text(text, classifiers[item["dataset"]])
        return {**item, "text": text, "classification": classification}

    def apply_policy(item: dict) -> dict:
        classification = item["classification"]
        if classification["status"] != "success":
            return {**item, "prompt": None, "skip_reason": "분류에 성공하지 못해 생성을 건너뜁니다."}
        text = item["text"][:MAX_INPUT_CHARS]
        prompt = build_prompt(classification["label"], text)
        if prompt is None:
            return {**item, "prompt": None, "skip_reason": f"'{classification['label']}' 라벨의 정책이 없습니다."}
        return {**item, "prompt": prompt, "text_truncated": len(item["text"]) > MAX_INPUT_CHARS}

    async def generate(item: dict) -> dict:
        if item["prompt"] is None:
            return {**item, "generation": {"status": "skipped", "error": item["skip_reason"]}}
        generation = await generator.generate(item["prompt"])
        generation["metrics"]["text_truncated"] = item["text_truncated"]
        return {**item, "generation": generation}

    def respond(item: dict) -> dict:
        classification, generation = item["classification"], item["generation"]
        done = classification["status"] == "success" and generation["status"] == "success"
        return {"status": "success" if done else "partial_success",
                "classification": classification, "generation": generation}

    # run_name은 추적(tracking/callbacks.py)이 단계를 구분하는 이름이다. 바꾸면 그쪽의 STEP_ATTRS도 맞춰야 한다.
    return (RunnableLambda(classify).with_config(run_name="classify")
            | RunnableLambda(apply_policy).with_config(run_name="apply_policy")
            | RunnableLambda(generate).with_config(run_name="generate")
            | RunnableLambda(respond).with_config(run_name="respond")).with_config(run_name="chat")
