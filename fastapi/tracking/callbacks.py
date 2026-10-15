"""LangChain 콜백으로 /chat 요청 하나의 호출 흐름(span 트리)과 단계별 지표를 모은다.

체인을 실행할 때 config={"callbacks": [handler], "metadata": {"request_id": ...}}로 넘기면, 체인과 각 단계(classify,
apply_policy, generate, respond)의 시작·종료마다 span을 하나씩 만든다. run_id/parent_run_id가 호출 트리가 되고,
metadata의 request_id가 모든 span에 붙어서 요청과 호출 흐름이 연결된다.
최상위 체인이 끝나면 그 요청의 span 전부를 sink(trace)로 한 번에 넘긴다. sink는 저장소(SQLite 등)나 로그 출력이다.

개인정보: 원문과 프롬프트는 기록하지 않는다. 입력은 글자 수와 해시 앞부분만 남긴다. 추적이 실패해도 요청은 영향을 받지 않는다.
"""

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler

logger = logging.getLogger("uvicorn.error")

MAX_PENDING_TRACES = 1000        # 끝나지 않은 요청이 쌓이는 것을 막는 상한. 넘으면 가장 오래된 것부터 버린다.
MAX_ERROR_CHARS = 300


@dataclass
class Span:
    run_id: str
    parent_run_id: str | None
    request_id: str | None
    name: str
    started_at: str                      # UTC ISO 8601
    duration_ms: float | None = None
    status: str = "running"              # success / error / timeout / skipped (단계가 돌려준 status를 따른다)
    error_type: str | None = None
    error_message: str | None = None
    attrs: dict = field(default_factory=dict)
    _t0: float = field(default_factory=time.perf_counter, repr=False)


# ═════════════════════════ 단계별로 남길 지표 ═════════════════════════
# 각 함수는 (입력, 출력) -> (status, attrs). 체인 단계(chain.py)의 입출력 형식을 알고 있어야 하는 유일한 곳이다.
def _text_attrs(text: str) -> dict:
    return {"text_len": len(text), "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]}


def _chat(inputs: dict, outputs: dict | None) -> tuple[str, dict]:
    attrs = {"dataset": inputs.get("dataset"), "provider": inputs.get("provider"), **_text_attrs(inputs.get("text", ""))}
    if outputs is None:
        return "running", attrs
    attrs["classification_status"] = outputs["classification"]["status"]
    attrs["generation_status"] = outputs["generation"]["status"]
    return outputs["status"], attrs


def _compare(inputs: dict, outputs: dict | None) -> tuple[str, dict]:
    """/compare의 최상위 체인. 제공자마다의 결과 상태를 함께 남긴다."""
    attrs = {"dataset": inputs.get("dataset"), **_text_attrs(inputs.get("text", ""))}
    if outputs is None:
        return "running", attrs
    attrs["classification_status"] = outputs["classification"]["status"]
    attrs["generation_statuses"] = {result["provider"]: result["status"] for result in outputs["results"]}
    return outputs["status"], attrs


def _classify(inputs: dict, outputs: dict | None) -> tuple[str, dict]:
    if outputs is None:
        return "running", {}
    result = outputs["classification"]
    attrs = {key: result[key] for key in ("label", "confidence", "probabilities", "error") if key in result}
    attrs.update(result.get("metrics", {}))
    return result["status"], attrs


def _apply_policy(inputs: dict, outputs: dict | None) -> tuple[str, dict]:
    if outputs is None:
        return "running", {}
    prompt = outputs["prompt"]
    attrs = {"has_prompt": prompt is not None, "prompt_chars": len(prompt) if prompt else 0,
             "text_truncated": outputs.get("text_truncated"), "skip_reason": outputs.get("skip_reason")}
    return ("success" if prompt else "skipped"), attrs


def _generation_result(result: dict) -> tuple[str, dict]:
    attrs = {"error": result["error"]} if "error" in result else {}
    attrs.update(result.get("metrics", {}))
    return result["status"], attrs


def _generate(inputs: dict, outputs: dict | None) -> tuple[str, dict]:
    """/chat이면 생성 호출 하나의 지표, /compare이면 제공자별 호출을 묶은 span이다.
    묶음 span의 시간은 가장 느린 호출에 맞춰지고, 호출별 지표는 아래의 generate_<제공자> span에 있다."""
    if outputs is None:
        return "running", {}
    if "generations" in outputs:
        statuses = {result["provider"]: result["status"] for result in outputs["generations"]}
        values = set(statuses.values())
        status = "success" if values == {"success"} else "skipped" if values == {"skipped"} else (
            "partial_success" if "success" in values else "error")
        return status, {"mode": "compare", "providers": list(statuses), "statuses": statuses}
    return _generation_result(outputs["generation"])


def _generate_call(inputs: dict, outputs: dict | None) -> tuple[str, dict]:
    """/compare에서 제공자 하나의 생성 호출 (span 이름: generate_openai, generate_ollama). 출력은 생성 결과 자체다."""
    if outputs is None:
        return "running", {}
    return _generation_result(outputs)


STEP_ATTRS: dict[str, Callable] = {"chat": _chat, "compare": _compare, "classify": _classify,
                                   "apply_policy": _apply_policy, "generate": _generate}


def step_extractor(name: str) -> Callable | None:
    if name.startswith("generate_"):
        return _generate_call
    return STEP_ATTRS.get(name)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ═════════════════════════ 콜백 핸들러 ═════════════════════════
class TrackingCallbackHandler(AsyncCallbackHandler):
    """핸들러 하나를 모든 요청이 같이 쓴다. 진행 중인 span은 run_id로 구분해서 요청마다 따로 모은다."""

    raise_error = False      # 추적 코드의 오류가 요청 처리를 깨뜨리지 않게 한다.

    def __init__(self, sink: Callable[[dict], None]):
        self.sink = sink
        self._spans: dict[str, Span] = {}          # run_id -> 진행 중인 span
        self._traces: dict[str, list[Span]] = {}   # 최상위 run_id -> 그 요청의 span들(시작 순서)
        self._owner: dict[str, str] = {}           # run_id -> 최상위 run_id

    # ── span 시작 ──
    async def on_chain_start(self, serialized, inputs, *, run_id: UUID, parent_run_id: UUID | None = None,
                             tags=None, metadata=None, **kwargs) -> None:
        try:
            self._start(str(run_id), str(parent_run_id) if parent_run_id else None, inputs, metadata or {}, kwargs, serialized)
        except Exception:
            logger.exception("[tracking] span 시작 기록에 실패했습니다.")

    def _start(self, run_id: str, parent_id: str | None, inputs, metadata: dict, kwargs: dict, serialized) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name") or "unknown"
        root_id = self._owner.get(parent_id, run_id) if parent_id else run_id
        parent = self._spans.get(parent_id) if parent_id else None
        request_id = metadata.get("request_id") or (parent.request_id if parent else None)
        span = Span(run_id=run_id, parent_run_id=parent_id, request_id=request_id, name=name, started_at=now_iso())
        extract = step_extractor(name)
        if extract and isinstance(inputs, dict):
            span.attrs.update(extract(inputs, None)[1])
        self._spans[run_id] = span
        self._owner[run_id] = root_id
        if root_id not in self._traces and len(self._traces) >= MAX_PENDING_TRACES:
            self._discard(next(iter(self._traces)))
        self._traces.setdefault(root_id, []).append(span)

    # ── span 종료 ──
    async def on_chain_end(self, outputs, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs) -> None:
        try:
            span = self._spans.get(str(run_id))
            if span is None:
                return
            extract = step_extractor(span.name)
            status, attrs = "success", {}
            if extract and isinstance(outputs, dict):
                status, attrs = extract({}, outputs)
            span.attrs.update(attrs)
            self._finish(span, status)
        except Exception:
            logger.exception("[tracking] span 종료 기록에 실패했습니다.")

    async def on_chain_error(self, error: BaseException, *, run_id: UUID, parent_run_id: UUID | None = None,
                             **kwargs) -> None:
        try:
            span = self._spans.get(str(run_id))
            if span is None:
                return
            span.error_type = type(error).__name__
            span.error_message = str(error)[:MAX_ERROR_CHARS]
            self._finish(span, "error")
        except Exception:
            logger.exception("[tracking] span 오류 기록에 실패했습니다.")

    def _finish(self, span: Span, status: str) -> None:
        span.duration_ms = (time.perf_counter() - span._t0) * 1000
        span.status = status
        if span.name == "classify" and "inference_ms" in span.attrs:
            # 모델 계산을 뺀 나머지: 모델을 쓰려고 기다린 시간(락·스레드 대기)과 토큰화 등
            span.attrs["non_inference_ms"] = span.duration_ms - span.attrs["inference_ms"]
        del self._spans[span.run_id]
        if self._owner.get(span.run_id) == span.run_id:      # 최상위 체인이 끝났다 = 요청이 끝났다
            self._emit(span.run_id)

    def _emit(self, root_id: str) -> None:
        spans = self._traces.get(root_id, [])
        root = spans[0]
        trace = {"request_id": root.request_id, "run_id": root_id, "started_at": root.started_at,
                 "duration_ms": root.duration_ms, "status": root.status,
                 "spans": [{k: v for k, v in asdict(span).items() if not k.startswith("_")} for span in spans]}
        self._discard(root_id)
        try:
            self.sink(trace)
        except Exception:
            logger.exception("[tracking] 추적 결과를 내보내지 못했습니다. request_id=%s", trace["request_id"])

    def _discard(self, root_id: str) -> None:
        for span in self._traces.pop(root_id, []):
            self._spans.pop(span.run_id, None)
            self._owner.pop(span.run_id, None)


# ═════════════════════════ 기본 sink ═════════════════════════
def log_trace(trace: dict) -> None:
    """요청 하나를 로그 한 줄로 요약한다. (저장소 sink가 붙기 전까지의 확인용)"""
    steps = " ".join(f"{s['name']}={s['duration_ms']:.0f}ms({s['status']})" for s in trace["spans"][1:])
    logger.info("[trace] request_id=%s status=%s total=%.0fms | %s", trace["request_id"], trace["status"],
                trace["duration_ms"], steps)
