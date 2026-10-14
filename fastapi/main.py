import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from admin import router as admin_router
from chain import create_chain
from classifier import close_classifiers, load_classifier
from generator import OpenAIGenerator
from fastapi import Depends, FastAPI, HTTPException, Request
from langchain_core.runnables import Runnable
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from tracking.callbacks import TrackingCallbackHandler, log_trace
from tracking.context import current_request
from tracking.middleware import RequestTrackingMiddleware, log_request
from tracking.store import open_store_from_env


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
    app.state.store = await asyncio.to_thread(open_store_from_env)
    app.state.tracker = TrackingCallbackHandler(sink=trace_sink)
    try:
        yield
    finally:
        app.state.chain = None
        await generator.aclose()
        await asyncio.to_thread(close_classifiers, classifiers)
        if app.state.store:
            await asyncio.to_thread(app.state.store.close)      # 큐에 남은 기록을 모두 저장하고 닫는다.


app = FastAPI(title="분류 챗봇 API", lifespan=lifespan)
app.state.store = None


def trace_sink(trace: dict) -> None:
    # 체인이 끝났을 때 호출 흐름(span)을 저장한다. 저장소를 열지 못했으면 로그 한 줄로 남긴다.
    if app.state.store:
        app.state.store.write_trace(trace)
    else:
        log_trace(trace)


def request_sink(record: dict) -> None:
    # 응답이 나간 뒤 요청 기록을 저장한다. 저장소가 있어도 요청마다 로그 한 줄은 남긴다.
    log_request(record)
    if app.state.store:
        app.state.store.write_request(record)


app.add_middleware(RequestTrackingMiddleware, sink=request_sink)
app.include_router(admin_router)


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
    request: Request,
    chain: Annotated[Runnable, Depends(get_chain)],
) -> dict:
    # 요청 ID와 요청 레코드는 미들웨어가 만든다. 검증을 통과한 입력과 결과를 레코드에 채우고, span은 같은 request_id로 연결한다.
    record = current_request()
    request_id = record.request_id if record else uuid.uuid4().hex
    if record:
        record.set_input(body.dataset, body.text)
    config = {"callbacks": [request.app.state.tracker], "metadata": {"request_id": request_id}}
    result = await chain.ainvoke(body.model_dump(), config=config)
    if record:
        record.set_output(result)
    return result
