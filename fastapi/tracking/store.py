"""추적 결과를 SQLite 파일에 저장한다.

  requests : 요청 한 건 (tracking/context.py의 RequestRecord). 원문과 답변도 여기에만 있다.
  spans    : 그 요청의 호출 흐름 (tracking/callbacks.py의 trace). request_id로 requests와 연결된다.
  v_request_trace   : 요청과 호출 흐름을 한 번에 보는 뷰 (요청 1건 = 행 여러 개, seq 순서)
  v_request_summary : 요청 1건 = 행 1개. 단계별 시간과 분류·생성 지표를 한 줄로 모은 뷰 (집계용)
  v_generation_runs : 생성 호출 1번 = 행 1개. /chat은 호출 하나, /compare는 제공자마다 한 행이라 생성 모델끼리 비교할 때 쓴다.

쓰기는 요청 처리를 막지 않도록 큐에 넣고 백그라운드 스레드 하나가 묶어서 저장한다. (SQLite는 쓰기 연결이 하나일 때 가장 안전하다)
큐가 가득 차면 요청을 막는 대신 기록을 버리고 dropped 개수에 센다. 읽기는 별도 연결이라 쓰는 중에도 조회할 수 있다. (WAL)
"""

import json
import logging
import os
import queue
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("uvicorn.error")

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = ROOT / "logs" / "tracking.db"
SCHEMA_VERSION = 1
BATCH_SIZE = 200
PURGE_INTERVAL_SECONDS = 6 * 3600

TABLES = """
CREATE TABLE IF NOT EXISTS requests (
    request_id        TEXT PRIMARY KEY,
    client_request_id TEXT,
    started_at        TEXT NOT NULL,
    method            TEXT,
    path              TEXT,
    client_host       TEXT,
    user_agent        TEXT,
    status_code       INTEGER,
    duration_ms       REAL,
    outcome           TEXT,
    error_type        TEXT,
    error_message     TEXT,
    error_detail      TEXT,
    dataset           TEXT,
    input_len         INTEGER,
    input_sha256      TEXT,
    input_text        TEXT,
    output            TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_started_at ON requests(started_at);
CREATE INDEX IF NOT EXISTS idx_requests_client_id  ON requests(client_request_id);
CREATE INDEX IF NOT EXISTS idx_requests_outcome    ON requests(outcome, started_at);

CREATE TABLE IF NOT EXISTS spans (
    run_id        TEXT PRIMARY KEY,
    request_id    TEXT NOT NULL,
    parent_run_id TEXT,
    seq           INTEGER NOT NULL,
    name          TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    duration_ms   REAL,
    status        TEXT,
    error_type    TEXT,
    error_message TEXT,
    attrs         TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_request ON spans(request_id, seq);
CREATE INDEX IF NOT EXISTS idx_spans_name    ON spans(name, started_at);
"""

# 뷰는 정의를 바꿔도 반영되도록 시작할 때마다 다시 만든다.
VIEWS = """
DROP VIEW IF EXISTS v_request_trace;
CREATE VIEW v_request_trace AS
SELECT r.request_id, r.client_request_id, r.started_at AS request_started_at, r.method, r.path, r.status_code,
       r.outcome, r.dataset, r.duration_ms AS request_ms,
       s.seq, s.name AS span, p.name AS parent_span, s.started_at AS span_started_at, s.duration_ms AS span_ms,
       s.status AS span_status, s.error_type AS span_error_type, s.error_message AS span_error_message, s.attrs
FROM requests r
LEFT JOIN spans s ON s.request_id = r.request_id
LEFT JOIN spans p ON p.run_id = s.parent_run_id;

DROP VIEW IF EXISTS v_request_summary;
CREATE VIEW v_request_summary AS
SELECT r.request_id, r.client_request_id, r.started_at, r.dataset, r.status_code, r.outcome,
       r.duration_ms AS total_ms,
       c.duration_ms AS classify_ms, g.duration_ms AS generate_ms,
       json_extract(c.attrs, '$.label')            AS label,
       json_extract(c.attrs, '$.confidence')       AS confidence,
       json_extract(c.attrs, '$.model')            AS classifier_model,
       json_extract(c.attrs, '$.inference_ms')     AS inference_ms,
       json_extract(c.attrs, '$.non_inference_ms') AS non_inference_ms,
       json_extract(c.attrs, '$.input_tokens')     AS classify_input_tokens,
       json_extract(c.attrs, '$.truncated')        AS truncated,
       c.status AS classification_status,
       g.status AS generation_status,
       json_extract(g.attrs, '$.model')            AS llm_model,
       json_extract(g.attrs, '$.input_tokens')     AS llm_input_tokens,
       json_extract(g.attrs, '$.output_tokens')    AS llm_output_tokens,
       json_extract(g.attrs, '$.reasoning_tokens') AS llm_reasoning_tokens,
       json_extract(g.attrs, '$.reasoning_effort') AS llm_reasoning_effort,
       COALESCE(r.error_type, c.error_type, g.error_type,
                json_extract(c.attrs, '$.error'), json_extract(g.attrs, '$.error')) AS error,
       r.path AS endpoint,
       json_extract(g.attrs, '$.provider')         AS llm_provider,
       json_extract(g.attrs, '$.providers')        AS llm_providers
FROM requests r
LEFT JOIN spans c ON c.request_id = r.request_id AND c.name = 'classify'
LEFT JOIN spans g ON g.request_id = r.request_id AND g.name = 'generate';

DROP VIEW IF EXISTS v_generation_runs;
CREATE VIEW v_generation_runs AS
SELECT s.run_id, s.request_id, r.started_at, r.path AS endpoint, r.dataset,
       json_extract(c.attrs, '$.label')            AS label,
       json_extract(s.attrs, '$.provider')         AS provider,
       json_extract(s.attrs, '$.model')            AS model,
       s.name AS span, s.status, s.duration_ms AS span_ms,
       json_extract(s.attrs, '$.latency_ms')       AS latency_ms,
       json_extract(s.attrs, '$.input_tokens')     AS input_tokens,
       json_extract(s.attrs, '$.output_tokens')    AS output_tokens,
       json_extract(s.attrs, '$.reasoning_tokens') AS reasoning_tokens,
       json_extract(s.attrs, '$.tokens_per_sec')   AS tokens_per_sec,
       json_extract(s.attrs, '$.load_ms')          AS load_ms,
       json_extract(s.attrs, '$.context_full')     AS context_full,
       json_extract(s.attrs, '$.reasoning_effort') AS reasoning_effort,
       json_extract(s.attrs, '$.temperature')      AS temperature,
       json_extract(s.attrs, '$.error')            AS error
FROM spans s
JOIN requests r ON r.request_id = s.request_id
LEFT JOIN spans c ON c.request_id = s.request_id AND c.name = 'classify'
WHERE s.name LIKE 'generate%' AND json_extract(s.attrs, '$.provider') IS NOT NULL;
"""


def _dumps(value) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, default=str)


def init_schema(path: Path) -> None:
    """파일과 테이블을 만든다. 서버 시작 때 호출해서 문제가 있으면 바로 드러나게 한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(TABLES)
        connection.executescript(VIEWS)
        connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        connection.commit()
    finally:
        connection.close()


class TrackingStore:
    def __init__(self, path: Path | str | None = None, retention_days: int = 0, queue_size: int = 10_000):
        self.path = Path(path or DEFAULT_DB_PATH)
        self.retention_days = retention_days
        self.dropped = 0
        init_schema(self.path)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._run, name="tracking-writer", daemon=True)
        self._thread.start()

    # ── 쓰기 (요청 처리 중에 불리므로 기다리지 않는다) ──
    def write_trace(self, trace: dict) -> None:
        self._put(("trace", trace))

    def write_request(self, record: dict) -> None:
        self._put(("request", record))

    def _put(self, item) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1
            logger.error("[tracking] 쓰기 큐가 가득 차서 기록을 버렸습니다. (누적 %d건)", self.dropped)

    def flush(self, timeout: float = 10.0) -> bool:
        """큐에 쌓인 기록이 모두 저장될 때까지 기다린다. 시간 안에 끝나면 True."""
        done = threading.Event()
        threading.Thread(target=lambda: (self._queue.join(), done.set()), daemon=True).start()
        return done.wait(timeout)

    def close(self, timeout: float = 10.0) -> None:
        self.flush(timeout)
        self._queue.put(None)         # 종료 신호
        self._thread.join(timeout)

    # ── 쓰기 스레드 ──
    def _run(self) -> None:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA synchronous=NORMAL")
        last_purge = 0.0
        try:
            while True:
                item = self._queue.get()
                batch, stop = [item], item is None
                while not stop and len(batch) < BATCH_SIZE:
                    try:
                        batch.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
                    stop = batch[-1] is None
                try:
                    self._write_batch(connection, [entry for entry in batch if entry is not None])
                except Exception:
                    logger.exception("[tracking] 기록을 저장하지 못했습니다. (%d건 버림)", len(batch))
                for _ in batch:
                    self._queue.task_done()
                if self.retention_days and time.monotonic() - last_purge > PURGE_INTERVAL_SECONDS:
                    last_purge = time.monotonic()
                    self._purge(connection, self.retention_days)
                if stop:
                    break
        finally:
            connection.close()

    def _write_batch(self, connection: sqlite3.Connection, batch: list) -> None:
        with connection:         # 한 번의 트랜잭션
            for kind, data in batch:
                if kind == "trace":
                    self._insert_trace(connection, data)
                else:
                    self._insert_request(connection, data)

    @staticmethod
    def _insert_trace(connection: sqlite3.Connection, trace: dict) -> None:
        connection.executemany(
            "INSERT OR REPLACE INTO spans (run_id, request_id, parent_run_id, seq, name, started_at, duration_ms, status,"
            " error_type, error_message, attrs) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [(span["run_id"], span["request_id"], span["parent_run_id"], seq, span["name"], span["started_at"],
              span["duration_ms"], span["status"], span["error_type"], span["error_message"], _dumps(span["attrs"]))
             for seq, span in enumerate(trace["spans"])])

    @staticmethod
    def _insert_request(connection: sqlite3.Connection, record: dict) -> None:
        connection.execute(
            "INSERT OR REPLACE INTO requests (request_id, client_request_id, started_at, method, path, client_host,"
            " user_agent, status_code, duration_ms, outcome, error_type, error_message, error_detail, dataset,"
            " input_len, input_sha256, input_text, output) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (record["request_id"], record["client_request_id"], record["started_at"], record["method"], record["path"],
             record["client_host"], record["user_agent"], record["status_code"], record["duration_ms"], record["outcome"],
             record["error_type"], record["error_message"], record["error_detail"], record["dataset"],
             record["input_len"], record["input_sha256"], record["input_text"], _dumps(record["output"])))

    # ── 보관 정리 ──
    def purge(self, days: int | None = None) -> int:
        """days일보다 오래된 요청과 그 span을 지운다. 지운 요청 수를 돌려준다."""
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            return self._purge(connection, days or self.retention_days)
        finally:
            connection.close()

    @staticmethod
    def _purge(connection: sqlite3.Connection, days: int | None = None) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="milliseconds")
        with connection:
            connection.execute("DELETE FROM spans WHERE request_id IN (SELECT request_id FROM requests WHERE started_at < ?)",
                               (cutoff,))
            connection.execute("DELETE FROM spans WHERE started_at < ?", (cutoff,))     # 요청 기록 없이 남은 span
            return connection.execute("DELETE FROM requests WHERE started_at < ?", (cutoff,)).rowcount

    # ── 읽기 (쓰기 연결과 별개라서 서버가 쓰는 중에도 조회할 수 있다) ──
    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def get_request_trace(self, request_id: str) -> dict | None:
        """요청 한 건과 그 호출 흐름. request_id나 client_request_id 어느 쪽으로도 찾는다."""
        return read_request_trace(self.path, request_id)


def read_request_trace(path: Path | str, request_id: str) -> dict | None:
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute("SELECT * FROM requests WHERE request_id = ? OR client_request_id = ? "
                                 "ORDER BY started_at DESC LIMIT 1", (request_id, request_id)).fetchone()
        if row is None:
            return None
        request = dict(row)
        request["output"] = json.loads(request["output"]) if request["output"] else None
        spans = []
        for span in connection.execute("SELECT * FROM spans WHERE request_id = ? ORDER BY seq", (request["request_id"],)):
            span = dict(span)
            span["attrs"] = json.loads(span["attrs"]) if span["attrs"] else {}
            spans.append(span)
        return {"request": request, "spans": spans}
    finally:
        connection.close()


def open_store_from_env() -> TrackingStore | None:
    """.env의 TRACKING_DB_PATH(기본 logs/tracking.db), TRACKING_RETENTION_DAYS(기본 0 = 지우지 않음)로 저장소를 연다.
    열지 못해도 서버는 떠야 하므로 실패하면 None을 돌려준다. (그때는 로그로만 남는다)"""
    try:
        return TrackingStore(os.getenv("TRACKING_DB_PATH") or None, int(os.getenv("TRACKING_RETENTION_DAYS") or 0))
    except Exception:
        logger.exception("[tracking] 저장소를 열지 못했습니다. 로그로만 기록합니다.")
        return None
