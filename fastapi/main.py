import asyncio
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from chain import create_chain
from classifier import close_classifiers, load_classifier
from generator import OpenAIGenerator
from fastapi import Depends, FastAPI, HTTPException, Request
from langchain_core.runnables import Runnable
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=10_000)
    dataset: Literal["documents", "inquiries"] = Field(
        validation_alias=AliasChoices("dataset", "topic")
    )

    @field_validator("text")
    @classmethod
    def nonblank(cls, text: str) -> str:
        text = text.strip()
        if not text:
            raise ValueError("공백만 입력할 수 없습니다.")
        return text


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifiers = await asyncio.to_thread(load_classifier)
    generator = OpenAIGenerator()
    app.state.classifiers = classifiers
    app.state.chain = create_chain(classifiers, generator)
    try:
        yield
    finally:
        app.state.chain = None
        await generator.aclose()
        await asyncio.to_thread(close_classifiers, classifiers)


app = FastAPI(title="분류 챗봇 API", lifespan=lifespan)


def get_chain(request: Request) -> Runnable:
    chain = getattr(request.app.state, "chain", None)
    if chain is None:
        raise HTTPException(status_code=503, detail="체인이 준비되지 않았습니다.")
    return chain


@app.get("/health")
async def health(request: Request) -> dict:
    # 서버 연결 확인용. 주제별 분류 모델이 준비됐는지도 함께 알려 준다.
    classifiers = getattr(request.app.state, "classifiers", {})
    return {
        "status": "ok",
        "mode": "live",
        "classifiers": {
            topic: {"model": bundle.model_key, "run_name": bundle.run_name, "error": bundle.error}
            for topic, bundle in classifiers.items()
        },
    }


@app.post("/chat")
async def chat(
    body: ChatRequest,
    chain: Annotated[Runnable, Depends(get_chain)],
) -> dict:
    return await chain.ainvoke(body.model_dump())
