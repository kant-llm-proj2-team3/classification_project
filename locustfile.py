"""Bounded /chat load test. Success requires successful classification AND generation.

Default request budget is 40 for one local Locust process. No response bodies or
API keys are written to test artifacts. Each request has a unique tracking ID.
"""
import json
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gevent
from gevent.event import Event
from locust import HttpUser, between, events, task
from locust.exception import StopUser
from locust.runners import MasterRunner, WorkerRunner

LABELS = {
    "inquiries": {"shipping", "refund", "account"},
    "documents": {"announcement", "manual", "minutes"},
}
CASES = [
    ("inquiries", "short", "배송 예정일이 비어 있습니다. 배송 상태를 확인하는 방법을 알려주세요."),
    ("documents", "short", "공지: 다음 주 월요일 오후 2시에 시스템 점검을 진행합니다. 작업을 미리 저장하세요."),
    ("inquiries", "long", "환불 접수 방법과 필요한 확인 절차를 알려주세요. " + "상품을 수령한 지 이틀 됐고 사용하지 않았습니다. 포장은 보관 중이며 주문 내역을 확인할 수 있습니다. " * 20),
    ("documents", "long", "회의록: 서비스 응답 지연 개선을 논의했습니다. " + "분류 시간과 외부 답변 생성 시간을 나눠 측정하기로 결정했습니다. 담당자와 완료일은 미정입니다. " * 20),
]
_ready = Event()
_state = {}


def error_kind(error):
    value = str(error or "")
    for token in ("insufficient_quota", "rate_limit_exceeded", "slow_down", "server_is_overloaded"):
        if token in value:
            return token
    for name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "RateLimitError", "APIConnectionError", "InternalServerError", "APITimeoutError", "RuntimeError", "JSONDecodeError", "ValueError"):
        if name in value:
            return name
    return "generation_error"


def validate_result(data, topic):
    if not isinstance(data, dict):
        return "invalid_json_object"
    classification = data.get("classification")
    generation = data.get("generation")
    if not isinstance(classification, dict) or not isinstance(generation, dict):
        return "missing_result_blocks"
    if classification.get("status") != "success":
        return "classification_" + str(classification.get("status", "missing"))
    if generation.get("status") != "success":
        return error_kind(generation.get("error"))
    if data.get("status") != "success":
        return "overall_status_not_success"
    if classification.get("label") not in LABELS[topic]:
        return "invalid_label"
    confidence = classification.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return "invalid_confidence"
    if not isinstance(generation.get("answer"), str) or not generation["answer"].strip():
        return "empty_answer"
    steps = generation.get("next_steps")
    if not isinstance(steps, list) or any(not isinstance(step, str) for step in steps):
        return "invalid_next_steps"
    return None


@events.init_command_line_parser.add_listener
def add_options(parser):
    parser.add_argument("--max-requests", type=int, default=40, help="Maximum requests in this local run")
    parser.add_argument("--run-id", default="", help="Tracking prefix, ASCII letters/digits/colon/hyphen only")
    parser.add_argument("--request-records", default="", help="Sanitized per-request JSONL destination")
    parser.add_argument("--think-min", type=float, default=5.0)
    parser.add_argument("--think-max", type=float, default=15.0)
    parser.add_argument("--wait-for-users", action="store_true", help="Wait for target users before first request (controlled small sample only)")


@events.test_start.add_listener
def start_test(environment, **kwargs):
    if isinstance(environment.runner, (MasterRunner, WorkerRunner)):
        raise RuntimeError("Request budget is local-process only; distributed execution is disabled")
    options = environment.parsed_options
    if options.max_requests <= 0 or not 0 <= options.think_min <= options.think_max:
        raise ValueError("Use a positive request budget and valid think-time range")
    prefix = options.run_id or "lt-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    if len(prefix) > 40 or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:-" for ch in prefix):
        raise ValueError("run-id must be at most 40 ASCII letters/digits/colon/hyphen")
    _ready.clear()
    _state.clear()
    _state.update(reserved=0, completed=0, quitting=False, limit=options.max_requests, run_id=prefix, record_path=options.request_records)
    if not options.wait_for_users:
        _ready.set()
    if options.request_records:
        path = Path(options.request_records)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


@events.spawning_complete.add_listener
def spawning_done(user_count, **kwargs):
    _ready.set()


class ChatUser(HttpUser):
    def wait_time(self):
        options = self.environment.parsed_options
        return between(options.think_min, options.think_max)(self)

    @task
    def chat(self):
        _ready.wait()
        if _state.get("reserved", 0) >= _state.get("limit", 0):
            raise StopUser()
        sequence = _state["reserved"]
        _state["reserved"] += 1  # No yield occurs between budget check and reservation.
        topic, case_name, text = CASES[sequence % len(CASES)]
        client_id = _state["run_id"] + ":" + uuid.uuid4().hex[:16]
        started = time.perf_counter()
        record = {"sequence": sequence + 1, "client_request_id": client_id, "topic": topic, "case_name": case_name,
                  "input_chars": len(text), "started_at": datetime.now(timezone.utc).isoformat()}
        try:
            with self.client.post("/chat", json={"text": text, "dataset": topic},
                                  headers={"X-Request-ID": client_id}, name=f"/chat/{topic}/{case_name}",
                                  timeout=(5, 90), catch_response=True) as response:
                record["http_status"] = response.status_code
                record["server_request_id"] = response.headers.get("X-Request-ID")
                data = {}
                if response.status_code != 200:
                    failure = f"http_{response.status_code}" if response.status_code else "connection_or_timeout"
                else:
                    try:
                        data = response.json()
                        failure = validate_result(data, topic)
                    except (ValueError, TypeError):
                        failure = "invalid_json"
                record["valid_success"] = failure is None
                record["error_kind"] = failure
                if failure:
                    response.failure(failure)
                if isinstance(data, dict):
                    c, g = data.get("classification") or {}, data.get("generation") or {}
                    if isinstance(c, dict) and isinstance(g, dict):
                        cm, gm = c.get("metrics") or {}, g.get("metrics") or {}
                        record.update(classification_status=c.get("status"), generation_status=g.get("status"),
                                      label=c.get("label"), confidence=c.get("confidence"),
                                      classifier_model=cm.get("model"), inference_ms=cm.get("inference_ms"),
                                      llm_model=gm.get("model"), generation_ms=gm.get("latency_ms"),
                                      input_tokens=gm.get("input_tokens"), output_tokens=gm.get("output_tokens"))
        finally:
            record["elapsed_ms"] = (time.perf_counter() - started) * 1000
            record["ended_at"] = datetime.now(timezone.utc).isoformat()
            if _state.get("record_path"):
                with Path(_state["record_path"]).open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            _state["completed"] += 1
            if _state["completed"] >= _state["limit"] and not _state["quitting"]:
                _state["quitting"] = True
                gevent.spawn_later(0.1, self.environment.runner.quit)
