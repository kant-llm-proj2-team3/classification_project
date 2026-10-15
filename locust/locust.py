"""Bounded /chat load test. Success requires successful classification AND generation.

Default request budget is 40 for one local Locust process. No response bodies or
API keys are written to test artifacts. Each request has a unique tracking ID.
"""
import csv
import json
import logging
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
from locust.stats import CSV_STATS_INTERVAL_SEC, StatsCSVFileWriter

LABELS = {
    "inquiries": {"shipping", "refund", "account"},
    "documents": {"announcement", "manual", "minutes"},
}
QUESTIONS_CSV = Path(__file__).resolve().parent / "advanced_merge_100.csv"


def load_questions(path):
    """Read dataset/text rows once per run; no model or API calls."""
    cases = []
    seen_ids = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"dataset", "text", "label", "id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("questions CSV requires dataset, text, label, id columns")
        for line, row in enumerate(reader, 2):
            topic, label = row["dataset"], row["label"]
            text, sample_id = row["text"], row["id"]
            if topic not in LABELS or label not in LABELS[topic]:
                raise ValueError(f"Invalid dataset/label in CSV record {line}")
            if not sample_id or sample_id in seen_ids or not text.strip() or len(text) > 10_000:
                raise ValueError(f"Invalid or duplicate sample in CSV record {line}")
            seen_ids.add(sample_id)
            cases.append((topic, label, text, sample_id))
    if not cases:
        raise ValueError("questions CSV is empty")
    return cases


_ready = Event()
_state = {}


def error_kind(error, status=None):
    value = str(error or "")
    if status == "timeout" or "APITimeoutError" in value:
        return "generation_timeout"
    if "max_output_tokens" in value:
        return "max_output_tokens"
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
        return error_kind(generation.get("error"), generation.get("status"))
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
    # The earlier API returns document next_steps; the summary API leaves them empty.
    if topic == "documents" and ("summary_items" in generation or not steps):
        items = generation.get("summary_items")
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            return "invalid_summary_items"
        if not any(item.strip() for item in items):
            return "empty_summary_items"
    return None


@events.init_command_line_parser.add_listener
def add_options(parser):
    parser.add_argument("--questions-csv", default=str(QUESTIONS_CSV), help="Input dataset/text CSV")
    parser.add_argument("--max-requests", type=int, default=40, help="Maximum requests in this local run")
    parser.add_argument("--run-id", default="", help="Tracking prefix, ASCII letters/digits/colon/hyphen only")
    parser.add_argument("--request-records", default="", help="Sanitized per-request JSONL destination")
    parser.add_argument("--think-min", type=float, default=5.0)
    parser.add_argument("--think-max", type=float, default=15.0)
    parser.add_argument("--wait-for-users", action="store_true", help="Wait for target users before first request (controlled small sample only)")


@events.test_start.add_listener
def start_test(environment, **kwargs):
    # Locust logs listener exceptions and continues spawning, so invalidate the old run first.
    if _state.get("start_error") and environment.process_exit_code == 2:
        environment.process_exit_code = None
    _ready.clear()
    _state.clear()
    _state.update(active=False, reserved=0, completed=0, quitting=False, limit=0)
    try:
        if isinstance(environment.runner, (MasterRunner, WorkerRunner)):
            raise RuntimeError("Request budget is local-process only; distributed execution is disabled")
        options = environment.parsed_options
        if options.max_requests <= 0 or not 0 <= options.think_min <= options.think_max:
            raise ValueError("Use a positive request budget and valid think-time range")
        prefix = options.run_id or "lt-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        if len(prefix) > 40 or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:-" for ch in prefix):
            raise ValueError("run-id must be at most 40 ASCII letters/digits/colon/hyphen")
        cases = load_questions(options.questions_csv)
        if options.request_records:
            path = Path(options.request_records)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
    except Exception as error:
        _state.update(start_error=True, quitting=True)
        _ready.set()
        environment.process_exit_code = 2
        logging.getLogger(__name__).error("Load test did not start: %s", error)
        gevent.spawn_later(0, environment.runner.quit)
        return
    _state.update(active=True, limit=options.max_requests, run_id=prefix, record_path=options.request_records)
    _state["cases"] = cases
    if not options.wait_for_users:
        _ready.set()


@events.test_stop.add_listener
def save_reports_on_test_stop(environment, **kwargs):
    if getattr(environment.parsed_options, "csv_prefix", None):
        writer = getattr(getattr(environment, "web_ui", None), "stats_csv_writer", None)
        if isinstance(writer, StatsCSVFileWriter):
            # Reuse Locust's handles without yielding, so a UI restart cannot clear these counts.
            for stream, csv_writer, save in (
                (writer.requests_csv_filehandle, writer.requests_csv_writer, writer.requests_csv),
                (writer.failures_csv_filehandle, writer.failures_csv_writer, writer.failures_csv),
                (writer.exceptions_csv_filehandle, writer.exceptions_csv_writer, writer.exceptions_csv),
            ):
                stream.seek(0)
                save(csv_writer)
                stream.truncate()
                stream.flush()
            writer.stats_history_flush()
        else:
            # Headless Locust closes/flushed its handles on exit; allow its final write first.
            gevent.sleep(CSV_STATS_INTERVAL_SEC + 0.1)
    # Web UI stays open after runner.quit(); save when the test stops as well.
    filename = getattr(environment.parsed_options, "html_file", None)
    if not filename:
        return
    from locust.html import get_html_report
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(get_html_report(environment, show_download_link=False), encoding="utf-8")


@events.spawning_complete.add_listener
def spawning_done(user_count, **kwargs):
    if _state.get("active"):
        _ready.set()


class ChatUser(HttpUser):
    def wait_time(self):
        options = self.environment.parsed_options
        return between(options.think_min, options.think_max)(self)

    @task
    def chat(self):
        _ready.wait()
        if not _state.get("active") or _state.get("reserved", 0) >= _state.get("limit", 0):
            raise StopUser()
        sequence = _state["reserved"]
        _state["reserved"] += 1  # No yield occurs between budget check and reservation.
        cases = _state["cases"]
        topic, case_name, text, sample_id = cases[sequence % len(cases)]
        client_id = _state["run_id"] + ":" + uuid.uuid4().hex[:16]
        started = time.perf_counter()
        record = {"sequence": sequence + 1, "client_request_id": client_id, "topic": topic, "case_name": case_name,
                  "sample_id": sample_id, "expected_label": case_name,
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
