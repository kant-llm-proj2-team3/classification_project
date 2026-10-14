import asyncio
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from chain import create_chain
from classifier import load_classifier
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
    app.state.chain = create_chain(classifiers)
    try:
        yield
    finally:
        app.state.chain = None


app = FastAPI(title="분류 챗봇 API", lifespan=lifespan)


def get_chain(request: Request) -> Runnable:
    chain = getattr(request.app.state, "chain", None)
    if chain is None:
        raise HTTPException(status_code=503, detail="체인이 준비되지 않았습니다.")
    return chain


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "mode": "stub"}


@app.post("/chat")
async def chat(body: ChatRequest) -> dict:
    # 프론트에 테스트 응답을 반환한다.
    return {
        "status": "success",
        "classification": {"status": "skipped"},
        "generation": {
            "status": "success",
            "answer": "요청을 정상적으로 받았습니다.",
            "next_steps": [],
        },
    }

# async def chat(
#     body: ChatRequest,
#     chain: Annotated[Runnable, Depends(get_chain)],
# ) -> dict:
#     return await chain.ainvoke(body.model_dump())
