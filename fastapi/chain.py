"""main.py의 요청을 처리하는 LCEL 체인: 분류 -> 정책 적용 -> 답변 생성 -> 응답 조립.

각 단계가 이전 단계의 결과를 들고 다니는 딕셔너리(item)를 받아 키를 더해 넘긴다.
  입력  : {"text", "dataset"} (+ /chat이면 "provider")
  classify     : + classification   (분류 모델, 동기 함수라서 ainvoke에서는 별도 스레드에서 실행된다)
  apply_policy : + prompt            (분류에 실패했거나 정책이 없으면 None)
  generate     : + generation        (선택한 생성 모델 하나, 비동기. 프롬프트가 없으면 skipped)
  respond      : API 응답(status, classification, generation)으로 정리

두 가지 체인이 분류와 정책 단계를 공유한다.
  create_chain(...)         /chat    제공자 하나(openai 또는 ollama)로 답변을 만든다.
  create_compare_chain(...) /compare 분류와 프롬프트는 한 번만 만들고, 같은 프롬프트를 모든 제공자에게 동시에 보내 결과를 나란히 돌려준다.
Streamlit이 검사하는 응답 계약 때문에 단계가 실패해도 예외 대신 단계별 status(error/timeout/skipped)를 담아 돌려준다.
추적(tracking/callbacks.py)이 run_name으로 단계를 구분한다. 이름을 바꾸면 그쪽의 STEP_ATTRS도 맞춰야 한다.
"""

import asyncio
import os

from classifier import predict_text
from generator import PROVIDERS, create_generators
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from policy import build_prompt

# 생성 모델에 보낼 원문의 최대 글자 수. 분류에는 전체 원문을 쓴다.
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS") or 4000)
DEFAULT_PROVIDER = "openai"


def as_generators(generators) -> dict:
    """제공자 이름 -> 생성기. 예전 방식(생성기 하나)으로 부르면 openai로 본다. 비우면 .env 설정으로 만든다."""
    if generators is None:
        return create_generators()
    if isinstance(generators, dict):
        return generators
    return {DEFAULT_PROVIDER: generators}


def failed_generation(provider: str, status: str, message: str) -> dict:
    return {"status": status, "error": message, "metrics": {"provider": provider}}


def common_steps(classifiers: dict) -> tuple[RunnableLambda, RunnableLambda]:
    """/chat과 /compare가 같이 쓰는 분류, 정책 적용 단계."""

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

    return (RunnableLambda(classify).with_config(run_name="classify"),
            RunnableLambda(apply_policy).with_config(run_name="apply_policy"))


# ═════════════════════════ /chat: 제공자 하나 ═════════════════════════
def create_chain(classifiers: dict, generators=None) -> Runnable:
    generators = as_generators(generators)
    classify, apply_policy = common_steps(classifiers)

    async def generate(item: dict) -> dict:
        if item["prompt"] is None:
            return {**item, "generation": {"status": "skipped", "error": item["skip_reason"]}}
        provider = item.get("provider") or DEFAULT_PROVIDER
        generator = generators.get(provider)
        if generator is None:
            generation = failed_generation(provider, "error", f"사용할 수 없는 생성 모델입니다: {provider}")
        else:
            generation = await generator.generate(item["prompt"])
        generation["metrics"]["text_truncated"] = item["text_truncated"]
        return {**item, "generation": generation}

    def respond(item: dict) -> dict:
        classification, generation = item["classification"], item["generation"]
        done = classification["status"] == "success" and generation["status"] == "success"
        return {"status": "success" if done else "partial_success",
                "classification": classification, "generation": generation}

    return (classify | apply_policy
            | RunnableLambda(generate).with_config(run_name="generate")
            | RunnableLambda(respond).with_config(run_name="respond")).with_config(run_name="chat")


# ═════════════════════════ /compare: 모든 제공자 동시에 ═════════════════════════
def create_compare_chain(classifiers: dict, generators=None, providers: tuple = PROVIDERS) -> Runnable:
    generators = as_generators(generators)
    providers = tuple(name for name in providers if name in generators)
    classify, apply_policy = common_steps(classifiers)

    def make_call(provider: str) -> Runnable:
        async def call(prompt: str) -> dict:
            return await generators[provider].generate(prompt)
        # 제공자마다 span이 하나씩 생긴다. (generate_openai, generate_ollama)
        return RunnableLambda(call).with_config(run_name=f"generate_{provider}")

    calls = {provider: make_call(provider) for provider in providers}

    async def generate(item: dict, config: RunnableConfig) -> dict:
        # 이 span(generate)이 부모가 되도록 config를 넘겨 호출한다. 두 호출은 동시에 시작하므로 걸린 시간은 느린 쪽에 맞춰진다.
        if item["prompt"] is None:
            results = [{"provider": name, **failed_generation(name, "skipped", item["skip_reason"])} for name in providers]
            return {**item, "generations": results}
        outcomes = await asyncio.gather(*(calls[name].ainvoke(item["prompt"], config=config) for name in providers),
                                        return_exceptions=True)
        results = []
        for name, outcome in zip(providers, outcomes, strict=True):
            if isinstance(outcome, BaseException):       # 생성기는 예외를 던지지 않게 만들었지만, 예상 밖의 오류도 한 제공자의 실패로 가둔다.
                outcome = failed_generation(name, "error", f"{type(outcome).__name__}: {str(outcome)[:200]}")
            outcome["metrics"]["text_truncated"] = item["text_truncated"]
            results.append({"provider": name, **outcome})
        return {**item, "generations": results}

    def respond(item: dict) -> dict:
        results = item["generations"]
        done = item["classification"]["status"] == "success" and results and all(r["status"] == "success" for r in results)
        return {"status": "success" if done else "partial_success",
                "classification": item["classification"], "results": results}

    return (classify | apply_policy
            | RunnableLambda(generate).with_config(run_name="generate")
            | RunnableLambda(respond).with_config(run_name="respond")).with_config(run_name="compare")
