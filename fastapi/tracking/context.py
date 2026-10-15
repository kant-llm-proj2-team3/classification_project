"""요청 하나의 추적 정보(RequestRecord)와, 그 요청을 처리하는 코드가 꺼내 쓰는 ContextVar.

미들웨어(tracking/middleware.py)가 요청이 들어올 때 RequestRecord를 만들어 ContextVar에 넣고, 요청 처리 중에 /chat이
입력·결과를 채우며, 응답이 끝나면 미들웨어가 완성된 레코드를 sink로 내보낸다.
span(tracking/callbacks.py)과는 request_id로 연결된다. 원문은 이 레코드에만 둔다. (span에는 글자 수와 해시만)
"""

import hashlib
import os
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone

# 요청 원문과 답변을 기록할지. 0/false/off/no면 글자 수와 해시만 남긴다.
STORE_TEXT = (os.getenv("TRACKING_STORE_TEXT") or "1").strip().lower() not in ("0", "false", "off", "no")
MAX_ERROR_CHARS = 300


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class RequestRecord:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)    # 서버가 발급. 항상 유일하다.
    client_request_id: str | None = None                                 # 클라이언트(Streamlit)가 보낸 X-Request-ID
    started_at: str = field(default_factory=now_iso)
    method: str = ""
    path: str = ""
    client_host: str | None = None
    user_agent: str | None = None
    status_code: int | None = None
    duration_ms: float | None = None
    outcome: str = "running"            # success / partial_success / error / cancelled (HTTP 오류는 error)
    error_type: str | None = None
    error_message: str | None = None
    error_detail: str | None = None     # 4xx/5xx 응답 본문 앞부분 (422 검증 오류의 원인 등)
    dataset: str | None = None
    input_len: int | None = None
    input_sha256: str | None = None
    input_text: str | None = None
    output: dict | None = None          # 사용자에게 돌려준 결과 요약 (분류 라벨, 답변 등)

    def set_input(self, dataset: str, text: str) -> None:
        self.dataset = dataset
        self.input_len = len(text)
        self.input_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        self.input_text = text if STORE_TEXT else None

    def set_output(self, result: dict) -> None:
        """/chat, /compare 응답을 요약해 둔다. (확률·지표는 span에 있으므로 넣지 않는다)
        /chat은 generation 하나, /compare는 제공자별 결과 목록(generations)을 담는다."""
        classification = result["classification"]
        output = {"status": result["status"],
                  "classification": {key: classification.get(key) for key in ("status", "label", "confidence", "error")}}

        def summarize(generation: dict) -> dict:
            summary = {"status": generation["status"], "error": generation.get("error")}
            if STORE_TEXT:
                summary.update(answer=generation.get("answer"), next_steps=generation.get("next_steps"))
                if generation.get("summary_items") is not None:      # 문서 요약(documents)
                    summary["summary_items"] = generation["summary_items"]
            return summary

        if "results" in result:
            output["generations"] = [{"provider": item["provider"], **summarize(item)} for item in result["results"]]
        else:
            output["generation"] = summarize(result["generation"])
            provider = result["generation"].get("metrics", {}).get("provider")
            if provider:
                output["generation"]["provider"] = provider
        self.output = output

    def set_error(self, error: BaseException) -> None:
        self.error_type = type(error).__name__
        self.error_message = str(error)[:MAX_ERROR_CHARS]

    def finish_outcome(self) -> None:
        if self.outcome == "cancelled":
            return
        if self.error_type or (self.status_code or 0) >= 400:
            self.outcome = "error"
        elif self.output:
            self.outcome = self.output["status"]
        else:
            self.outcome = "success"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


_current: ContextVar[RequestRecord | None] = ContextVar("tracking_request", default=None)


def current_request() -> RequestRecord | None:
    """지금 처리 중인 요청의 레코드. 미들웨어 밖(테스트 등)에서는 None."""
    return _current.get()


def bind_request(record: RequestRecord):
    """레코드를 지금 요청의 것으로 등록한다. 돌려받은 token을 unbind_request에 넘겨 해제한다."""
    return _current.set(record)


def unbind_request(token) -> None:
    _current.reset(token)
