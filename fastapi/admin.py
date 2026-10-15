"""추적 결과를 조회·내보내는 읽기 전용 /admin API. (Streamlit 트레이싱 페이지가 사용한다)

  GET  /admin/auth-info          암호가 필요한지 (인증 없이 부를 수 있다)
  GET  /admin/auth-check         서명이 맞는지 확인
  GET  /admin/requests           요청 목록 (필터, 정렬, 페이지)
  GET  /admin/requests/{id}      요청 한 건과 호출 흐름(span) 전체. request_id나 client_request_id로 찾는다.
  GET  /admin/stats              필터에 맞는 요청들의 집계 (건수, 지연 분포, 라벨, 토큰, 오류, 시간대별 추이)
  POST /admin/export             csv / json / jsonl 파일로 내보내기 (요약, 요청, span, 상세 중 선택)
인증은 admin_auth.py. 이 경로의 요청은 추적 기록에 남기지 않는다.
"""

import csv
import io
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from admin_auth import auth_required, require_admin
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from tracking.store import TrackingStore, read_request_trace

router = APIRouter(prefix="/admin", tags=["admin"])

MAX_LIST_ROWS = 1000
MAX_STAT_ROWS = 100_000
MAX_EXPORT_ROWS = 100_000
ID_CHUNK = 500
SORTS = {"started_at", "total_ms", "classify_ms", "generate_ms", "confidence"}

SUMMARY_COLUMNS = ("request_id", "client_request_id", "started_at", "dataset", "status_code", "outcome", "total_ms",
                   "classify_ms", "generate_ms", "label", "confidence", "classifier_model", "inference_ms",
                   "non_inference_ms", "classify_input_tokens", "truncated", "classification_status",
                   "generation_status", "llm_model", "llm_input_tokens", "llm_output_tokens", "llm_reasoning_tokens",
                   "llm_reasoning_effort", "error")
FROM_SUMMARY = "FROM v_request_summary s JOIN requests r ON r.request_id = s.request_id"


# ═════════════════════════ 필터 ═════════════════════════
class Filters(BaseModel):
    since: str | None = Field(None, description="이 시각(포함) 이후. ISO 8601")
    until: str | None = Field(None, description="이 시각(미포함) 이전. ISO 8601")
    dataset: Literal["documents", "inquiries"] | None = None
    outcome: str | None = None                 # success / partial_success / error / cancelled
    label: str | None = None
    status_code: int | None = None
    classification_status: str | None = None
    generation_status: str | None = None
    min_total_ms: float | None = None
    q: str | None = Field(None, description="입력 원문에 포함된 글자")


def query_filters(
    since: str | None = Query(None, description="이 시각(포함) 이후. ISO 8601"),
    until: str | None = Query(None, description="이 시각(미포함) 이전. ISO 8601"),
    dataset: Literal["documents", "inquiries"] | None = None,
    outcome: str | None = None,
    label: str | None = None,
    status_code: int | None = None,
    classification_status: str | None = None,
    generation_status: str | None = None,
    min_total_ms: float | None = None,
    q: str | None = Query(None, description="입력 원문에 포함된 글자"),
) -> Filters:
    # GET 쿼리 문자열을 Filters로 모은다. (쿼리 모델은 다른 쿼리 파라미터와 함께 쓸 수 없어서 직접 받는다)
    return Filters(since=since, until=until, dataset=dataset, outcome=outcome, label=label, status_code=status_code,
                   classification_status=classification_status, generation_status=generation_status,
                   min_total_ms=min_total_ms, q=q)


QueryFilters = Annotated[Filters, Depends(query_filters)]


def normalize_time(value: str) -> str:
    """저장된 started_at(UTC ISO)과 문자열로 비교할 수 있는 형식으로 바꾼다."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=422, detail=f"시각 형식이 올바르지 않습니다: {value}") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def build_where(filters: Filters) -> tuple[str, list]:
    clauses, params = [], []

    def add(clause: str, value) -> None:
        clauses.append(clause)
        params.append(value)

    if filters.since:
        add("s.started_at >= ?", normalize_time(filters.since))
    if filters.until:
        add("s.started_at < ?", normalize_time(filters.until))
    for column, value in (("s.dataset", filters.dataset), ("s.outcome", filters.outcome), ("s.label", filters.label),
                          ("s.status_code", filters.status_code),
                          ("s.classification_status", filters.classification_status),
                          ("s.generation_status", filters.generation_status)):
        if value is not None:
            add(f"{column} = ?", value)
    if filters.min_total_ms is not None:
        add("s.total_ms >= ?", filters.min_total_ms)
    if filters.q:
        escaped = filters.q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        add("r.input_text LIKE ? ESCAPE '\\'", f"%{escaped}%")
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def get_store(request: Request) -> TrackingStore:
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="추적 저장소가 열려 있지 않습니다.")
    return store


Store = Annotated[TrackingStore, Depends(get_store)]
Admin = Depends(require_admin)


# ═════════════════════════ 인증 ═════════════════════════
@router.get("/auth-info")
def auth_info(request: Request) -> dict:
    return {"required": auth_required(), "store": getattr(request.app.state, "store", None) is not None}


@router.get("/auth-check", dependencies=[Admin])
def auth_check() -> dict:
    return {"ok": True}


# ═════════════════════════ 목록 · 상세 ═════════════════════════
LIST_SELECT = (", ".join(f"s.{column}" for column in SUMMARY_COLUMNS)
               + ", r.input_len, substr(r.input_text, 1, 150) AS input_preview")


@router.get("/requests", dependencies=[Admin])
def list_requests(store: Store, filters: QueryFilters, limit: int = Query(200, ge=1, le=MAX_LIST_ROWS),
                  offset: int = Query(0, ge=0), sort: str = "started_at",
                  direction: Literal["asc", "desc"] = "desc") -> dict:
    if sort not in SORTS:
        raise HTTPException(status_code=422, detail=f"정렬 기준은 {sorted(SORTS)} 중 하나여야 합니다.")
    where, params = build_where(filters)
    connection = store.connect()
    try:
        total = connection.execute(f"SELECT count(*) {FROM_SUMMARY}{where}", params).fetchone()[0]
        rows = connection.execute(
            f"SELECT {LIST_SELECT} {FROM_SUMMARY}{where} ORDER BY s.{sort} {direction} LIMIT ? OFFSET ?",
            [*params, limit, offset]).fetchall()
    finally:
        connection.close()
    return {"total": total, "items": [dict(row) for row in rows]}


@router.get("/requests/{request_id}", dependencies=[Admin])
def request_detail(request_id: str, store: Store) -> dict:
    trace = read_request_trace(store.path, request_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="해당 요청을 찾을 수 없습니다.")
    return trace


# ═════════════════════════ 통계 ═════════════════════════
def percentile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def describe(values: list) -> dict | None:
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    return {"count": len(values), "mean": sum(values) / len(values), "min": values[0], "p50": percentile(values, 0.5),
            "p90": percentile(values, 0.9), "p95": percentile(values, 0.95), "p99": percentile(values, 0.99),
            "max": values[-1]}


def count_by(rows: list, key: str) -> dict:
    counts: dict = {}
    for row in rows:
        name = "(없음)" if row[key] is None else str(row[key])
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


@router.get("/stats", dependencies=[Admin])
def stats(store: Store, filters: QueryFilters) -> dict:
    where, params = build_where(filters)
    connection = store.connect()
    try:
        rows = [dict(row) for row in connection.execute(
            "SELECT s.started_at, s.dataset, s.outcome, s.status_code, s.total_ms, s.classify_ms, s.generate_ms, s.label,"
            " s.confidence, s.inference_ms, s.non_inference_ms, s.truncated, s.llm_input_tokens, s.llm_output_tokens,"
            f" s.llm_reasoning_tokens, s.error {FROM_SUMMARY}{where} ORDER BY s.started_at LIMIT ?",
            [*params, MAX_STAT_ROWS])]
    finally:
        connection.close()

    labels: dict = {}
    for row in rows:
        if row["label"] is not None:
            by_dataset = labels.setdefault(row["dataset"], {})
            by_dataset[row["label"]] = by_dataset.get(row["label"], 0) + 1

    timeline, bucket_unit = [], "hour"
    if rows:
        first = datetime.fromisoformat(rows[0]["started_at"])
        last = datetime.fromisoformat(rows[-1]["started_at"])
        bucket_unit = "hour" if (last - first) <= timedelta(days=3) else "day"     # 기간이 3일 이하면 시간별, 넘으면 일별
        buckets: dict = {}
        for row in rows:
            key = (row["started_at"][:13] + ":00:00+00:00" if bucket_unit == "hour"
                   else row["started_at"][:10] + "T00:00:00+00:00")
            bucket = buckets.setdefault(key, {"bucket": key, "count": 0, "errors": 0, "total_ms": []})
            bucket["count"] += 1
            bucket["errors"] += row["outcome"] != "success"
            bucket["total_ms"].append(row["total_ms"])
        timeline = [{"bucket": b["bucket"], "count": b["count"], "errors": b["errors"],
                     "avg_total_ms": sum(v for v in b["total_ms"] if v is not None) / max(len(b["total_ms"]), 1)}
                    for b in buckets.values()]

    return {
        "total": len(rows), "truncated_rows": len(rows) >= MAX_STAT_ROWS,
        "bucket": bucket_unit,
        "outcomes": count_by(rows, "outcome"), "datasets": count_by(rows, "dataset"),
        "status_codes": count_by(rows, "status_code"), "labels": labels,
        "latency": {name: describe([row[name] for row in rows])
                    for name in ("total_ms", "classify_ms", "generate_ms", "inference_ms", "non_inference_ms")},
        "confidence": describe([row["confidence"] for row in rows]),
        "tokens": {"llm_input": sum(row["llm_input_tokens"] or 0 for row in rows),
                   "llm_output": sum(row["llm_output_tokens"] or 0 for row in rows),
                   "llm_reasoning": sum(row["llm_reasoning_tokens"] or 0 for row in rows)},
        "truncated_inputs": sum(1 for row in rows if row["truncated"]),
        "errors": count_by([row for row in rows if row["error"]], "error"),
        "timeline": timeline,
    }


# ═════════════════════════ 내보내기 ═════════════════════════
class ExportRequest(BaseModel):
    format: Literal["csv", "json", "jsonl"] = "csv"
    scope: Literal["summary", "requests", "spans", "trace"] = "summary"
    include_text: bool = Field(False, description="입력 원문과 답변을 포함할지")
    filters: Filters = Filters()
    request_ids: list[str] | None = Field(None, max_length=10_000, description="주면 필터 대신 이 요청들만 내보낸다")
    limit: int = Field(MAX_EXPORT_ROWS, ge=1, le=MAX_EXPORT_ROWS)


def flatten(value, prefix: str = "") -> dict:
    """중첩 dict를 'a.b.c' 키의 한 줄로 편다. 리스트는 JSON 문자열로 둔다. (CSV용)"""
    if isinstance(value, dict):
        flat: dict = {}
        for key, item in value.items():
            flat.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    if isinstance(value, (list, tuple)):
        value = json.dumps(value, ensure_ascii=False)
    return {prefix: value}


def render_csv(rows: list[dict]) -> bytes:
    flat_rows = [flatten(row) for row in rows]
    columns: list = []
    for row in flat_rows:
        columns.extend(key for key in row if key not in columns)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(flat_rows)
    return buffer.getvalue().encode("utf-8-sig")        # BOM이 있어야 Excel에서 한글이 깨지지 않는다.


def render(rows: list[dict], file_format: str) -> bytes:
    if file_format == "csv":
        return render_csv(rows)
    if file_format == "json":
        return json.dumps(rows, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    return "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows).encode("utf-8")


def chunks(items: list, size: int = ID_CHUNK):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def select_ids(connection, body: ExportRequest) -> list[str]:
    """내보낼 요청 ID를 시작 시각 순서로 고른다."""
    if body.request_ids:
        found = []
        for part in chunks(body.request_ids):
            found += [(row[0], row[1]) for row in connection.execute(
                f"SELECT request_id, started_at FROM requests WHERE request_id IN ({','.join('?' * len(part))})", part)]
        return [request_id for request_id, _ in sorted(set(found), key=lambda item: item[1])][:body.limit]
    where, params = build_where(body.filters)
    return [row[0] for row in connection.execute(
        f"SELECT s.request_id {FROM_SUMMARY}{where} ORDER BY s.started_at LIMIT ?", [*params, body.limit])]


def fetch_by_ids(connection, sql: str, ids: list[str]) -> list[dict]:
    rows: list[dict] = []
    for part in chunks(ids):
        rows += [dict(row) for row in connection.execute(sql.format(marks=",".join("?" * len(part))), part)]
    return rows


def request_row(row: dict, include_text: bool) -> dict:
    row["output"] = json.loads(row["output"]) if row["output"] else None
    if not include_text:
        row["input_text"] = None
        if row["output"]:
            row["output"].get("generation", {}).pop("answer", None)
            row["output"].get("generation", {}).pop("next_steps", None)
    return row


def collect_rows(connection, body: ExportRequest) -> list[dict]:
    ids = select_ids(connection, body)
    order = {request_id: index for index, request_id in enumerate(ids)}
    if body.scope == "summary":
        extra = (", r.input_text, json_extract(r.output, '$.generation.answer') AS answer,"
                 " json_extract(r.output, '$.generation.next_steps') AS next_steps") if body.include_text else ""
        sql = (f"SELECT {', '.join('s.' + column for column in SUMMARY_COLUMNS)}, r.input_len{extra} "
               f"{FROM_SUMMARY} WHERE s.request_id IN ({{marks}})")
        rows = fetch_by_ids(connection, sql, ids)
    elif body.scope == "requests":
        rows = [request_row(row, body.include_text)
                for row in fetch_by_ids(connection, "SELECT * FROM requests WHERE request_id IN ({marks})", ids)]
    else:
        spans = fetch_by_ids(connection, "SELECT * FROM spans WHERE request_id IN ({marks}) ORDER BY seq", ids)
        for span in spans:
            span["attrs"] = json.loads(span["attrs"]) if span["attrs"] else {}
        if body.scope == "spans":
            rows = spans
        else:     # trace: 요청 + 그 span들
            requests = {row["request_id"]: request_row(row, body.include_text) for row in fetch_by_ids(
                connection, "SELECT * FROM requests WHERE request_id IN ({marks})", ids)}
            by_request: dict = {}
            for span in spans:
                by_request.setdefault(span["request_id"], []).append(span)
            rows = [{"request": requests[request_id], "spans": by_request.get(request_id, [])}
                    for request_id in ids if request_id in requests]
    key = "request_id"
    if body.scope == "trace":
        rows.sort(key=lambda row: order[row["request"]["request_id"]])
    else:
        rows.sort(key=lambda row: (order.get(row[key], 0), row.get("seq", 0)))
    return rows


@router.post("/export", dependencies=[Admin])
def export(body: ExportRequest, store: Store) -> Response:
    connection = store.connect()
    try:
        rows = collect_rows(connection, body)
    finally:
        connection.close()
    if body.format == "csv" and body.scope == "trace":
        # CSV는 한 줄이 한 span이어야 하므로 요청 정보를 span마다 반복해서 펼친다.
        rows = [{"request": row["request"], "span": span} for row in rows for span in (row["spans"] or [None])]
    payload = render(rows, body.format)
    extension, media = {"csv": ("csv", "text/csv; charset=utf-8"), "json": ("json", "application/json"),
                        "jsonl": ("jsonl", "application/x-ndjson")}[body.format]
    filename = f"tracing_{body.scope}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extension}"
    return Response(payload, media_type=media, headers={"Content-Disposition": f'attachment; filename="{filename}"',
                                                       "X-Export-Rows": str(len(rows))})
