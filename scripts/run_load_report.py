"""Run a small capped Locust suite and join measurements to the server tracking DB.

Uses the repository's existing Python environment. Does not read API keys,
change model settings, or retry failed requests. New /chat requests <= 40.
"""
import argparse
import json
import math
import sqlite3
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def percentile(values, quantile):
    values = sorted(float(v) for v in values if v is not None)
    return values[max(0, math.ceil(len(values) * quantile) - 1)] if values else None


def mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.mean(values) if values else None


def check_tracking_db(db):
    """Validate the existing read-only DB before sending any load requests."""
    if not db.is_file():
        raise RuntimeError(
            f"Tracking DB file not found: {db}. "
            "Use --tracking-db to select the server's existing DB; no DB is created."
        )
    conn = None
    try:
        conn = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=5)
        conn.execute("PRAGMA query_only=ON")
        conn.execute(
            "SELECT r.client_request_id,v.* FROM requests r "
            "LEFT JOIN v_request_summary v ON r.request_id=v.request_id WHERE 0"
        ).fetchall()
    except (sqlite3.Error, OSError) as error:
        raise RuntimeError(f"Cannot read tracking DB {db}: {error}") from error
    finally:
        if conn is not None:
            conn.close()


def join_tracking(db, rows):
    ids = [r["client_request_id"] for r in rows]
    if not db.exists() or not ids:
        return []
    for _ in range(8):
        try:
            conn = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=5)
            conn.row_factory = sqlite3.Row
            placeholders = ",".join("?" * len(ids))
            found = conn.execute(f"SELECT r.client_request_id,v.* FROM requests r LEFT JOIN v_request_summary v ON r.request_id=v.request_id WHERE r.client_request_id IN ({placeholders})", ids).fetchall()
            conn.close()
            if len(found) >= len(rows):
                break
        except sqlite3.Error:
            found = []
        time.sleep(0.5)
    safe_fields = ("request_id", "client_request_id", "total_ms", "classify_ms", "generate_ms", "inference_ms", "non_inference_ms", "classification_status", "generation_status", "llm_input_tokens", "llm_output_tokens", "llm_model", "classifier_model", "outcome")
    return [{k: row[k] for k in safe_fields if k in row.keys()} for row in found]


def summarize(rows, tracked):
    latency = [r["elapsed_ms"] for r in rows]
    succeeded = sum(bool(r.get("valid_success")) for r in rows)
    http_fail = sum(r.get("http_status") != 200 for r in rows)
    start = min((datetime.fromisoformat(r["started_at"]).timestamp() for r in rows), default=0)
    end = max((datetime.fromisoformat(r["ended_at"]).timestamp() for r in rows), default=0)
    return {
        "requests": len(rows), "valid_success": succeeded, "http_failures": http_fail,
        "application_failures": len(rows) - succeeded - http_fail,
        "success_rate": succeeded / len(rows) if rows else None,
        "average_ms": mean(latency), "p50_ms": percentile(latency, .5), "p95_ms": percentile(latency, .95), "p99_ms": percentile(latency, .99),
        "min_ms": min(latency) if latency else None, "max_ms": max(latency) if latency else None,
        "throughput_rps": len(rows) / (end - start) if end > start else None,
        "classify_average_ms": mean([r.get("classify_ms") for r in tracked]),
        "inference_average_ms": mean([r.get("inference_ms") for r in tracked]),
        "non_inference_average_ms": mean([r.get("non_inference_ms") for r in tracked]),
        "generate_average_ms": mean([r.get("generate_ms") for r in tracked]),
        "input_tokens": sum((r.get("input_tokens") or 0) for r in rows),
        "output_tokens": sum((r.get("output_tokens") or 0) for r in rows),
        "db_matched": len(tracked), "over_90s": sum(v > 90000 for v in latency),
        "error_kinds": dict(Counter(r.get("error_kind") for r in rows if r.get("error_kind"))),
        "topic_counts": dict(Counter(r.get("topic") for r in rows)),
    }


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tracking-db", type=Path, default=repo / "logs" / "tracking.db",
        help="Existing server tracking DB, read-only (default: %(default)s). "
             "Relative paths use the caller's working directory.",
    )
    parser.add_argument("--historical-json", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.tracking_db = args.tracking_db.expanduser().resolve()
    try:
        check_tracking_db(args.tracking_db)
    except RuntimeError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.host + "/health", timeout=5) as response:
        health = json.load(response)
    if health.get("mode") != "live":
        raise RuntimeError("Expected the live classification API")
    bad_topics = [topic for topic, item in health.get("classifiers", {}).items() if item.get("error")]
    if bad_topics:
        raise RuntimeError("Classifier startup error; no load generated: " + ",".join(bad_topics))
    historical = json.loads(args.historical_json.read_text(encoding="utf-8")) if args.historical_json and args.historical_json.exists() else {}
    report = {"historical": historical, "runs": [], "max_requests": 40, "sources": [
        {"label": "Locust response validation", "url": "https://docs.locust.io/en/stable/writing-a-locustfile.html#validating-responses"},
        {"label": "OpenAI latency optimization", "url": "https://developers.openai.com/api/docs/guides/latency-optimization"},
        {"label": "OpenAI rate limits", "url": "https://developers.openai.com/api/docs/guides/rate-limits"}],
        "scope_notes": ["새 호출 최대 40건. 실패 요청 자동 재시도 없음.", "locust/advanced_merge_100.csv의 문서·문의 질문을 사용하며, 각 단계는 같은 파일의 처음 4~12개를 순서대로 사용합니다. 정답률을 측정하는 테스트는 아닙니다.", "새 단계당 4~12건으로 탐색 결과이며 P95/P99·수용 인원 확정에 불충분.", "프론트 UI·WebSocket과 1,000/2,000명 수용 능력은 검증하지 않음.", "테스트 종료 후 남은 서버 작업과 기록 수를 점검. 분류 비추론 시간은 토큰화/스레드/락 대기가 섞인 지표."]}
    scenarios = [("연결·응답 검증", "smoke", 1, 1, 4, 0, 0), ("1명 기준", "baseline", 1, 1, 12, 1, 2), ("5명 동시", "concurrent5", 5, 5, 12, 1, 2), ("10명 동시", "concurrent10", 10, 10, 12, 1, 2)]
    if args.smoke_only:
        scenarios = scenarios[:1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    for label, suffix, users, ramp, budget, wait_min, wait_max in scenarios:
        run_id = "lr-" + stamp + "-" + suffix
        folder = args.output_dir / suffix
        folder.mkdir(exist_ok=True)
        raw = folder / "requests.jsonl"
        command = [sys.executable, "-X", "utf8", "-m", "locust", "-f", str(repo / "locust" / "locust.py"), "--host", args.host,
                   "--headless", "--users", str(users), "--spawn-rate", str(ramp), "--run-time", "3m", "--stop-timeout", "95",
                   "--max-requests", str(budget), "--run-id", run_id, "--request-records", str(raw),
                   "--think-min", str(wait_min), "--think-max", str(wait_max), "--csv", str(folder / "locust"),
                   "--csv-full-history", "--html", str(folder / "locust.html"), "--only-summary", "--wait-for-users"]
        started = datetime.now(timezone.utc).isoformat()
        print(f"START {suffix}: users={users}, request cap={budget}", flush=True)
        with (folder / "locust.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, timeout=310)
        rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()] if raw.exists() else []
        if len(rows) > budget:
            raise RuntimeError("Request budget exceeded")
        tracked = join_tracking(args.tracking_db, rows)
        summary = summarize(rows, tracked)
        run = {"label": label, "run_id": run_id, "users": users, "spawn_rate": ramp, "request_limit": budget,
               "started_at": started, "ended_at": datetime.now(timezone.utc).isoformat(), "summary": summary,
               "requests": rows, "tracked": tracked, "locust_exit_code": process.returncode, "think_time_seconds": [wait_min, wait_max]}
        report["runs"].append(run)
        (args.output_dir / "load_test_data.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"DONE {suffix}: completed={len(rows)}, success={summary['valid_success']}, db={len(tracked)}, avg_ms={summary['average_ms']}", flush=True)
        if len(rows) != budget or summary["valid_success"] != len(rows) or len(tracked) != len(rows):
            report["scope_notes"].append("오류/기록 불일치가 발생해 더 높은 동시 사용자 단계는 자동 중단했습니다.")
            break
        if summary["p95_ms"] and summary["p95_ms"] > 30000:
            report["scope_notes"].append("P95가 탐색 중단 기준 30초를 넘어 더 높은 단계는 진행하지 않았습니다. 이 기준은 서비스 SLA가 아닙니다.")
            break
        time.sleep(2)
    report["new_requests_completed"] = sum(r["summary"]["requests"] for r in report["runs"])
    (args.output_dir / "load_test_data.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"COMPLETE: {report['new_requests_completed']} /chat requests (cap 40)", flush=True)


if __name__ == "__main__":
    main()
