"""생성 모델 비교·평가 도구가 함께 쓰는 경로, 표본 추출, 집계, 채점표 입출력.

입력은 dataset/<토픽>/<종류>/test.csv 뿐이다. (종류: basic, basic_add, advanced, advanced_add)
비교용 데이터 파일을 따로 만들지 않고, 실행할 때 --per-kind N으로 종류마다 N건을 그 자리에서 뽑는다.
뽑은 행은 실행 결과 폴더의 config.json에 남아서 같은 시드로 다시 뽑거나 확인할 수 있다.

참고: project2-kit step09_compare.py(비교 기록과 채점표), step11_evaluate_final.py(확정 모델 최종 평가).
"""

import csv
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

FASTAPI_DIR = Path(__file__).resolve().parent.parent
ROOT = FASTAPI_DIR.parent
DATASET_DIR = ROOT / "dataset"
ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_OUT_ROOT = ARTIFACTS_DIR / "generation_compare"

TOPICS = ("documents", "inquiries")
KINDS = ("basic", "basic_add", "advanced", "advanced_add")

# 사람 평가 항목 (0점: 중요한 오류나 미충족, 1점: 일부 충족하나 수정 필요, 2점: 충족). 채점하지 않은 칸은 0점이 아니라 빈칸이다.
# 문의(inquiries)는 답변과 다음 행동을, 문서(documents)는 요약 항목을 본다. (열 이름은 같고, 문서는 괄호 안의 뜻으로 읽는다.)
CRITERIA = {
    "policy_correct_0_2": "정책과 원문에 맞는가? (문서: 요약이 원문과 정책의 항목 구성에 맞는가)",
    "required_points_0_2": "필요한 안내를 담았는가? (문서: 원문의 핵심을 빠뜨리지 않았는가)",
    "no_unsupported_claims_0_2": "없는 날짜, 메뉴, 담당자, 완료 사실을 만들지 않았는가? (문서: 원문에 없는 내용을 더하지 않았는가)",
    "clarity_0_2": "설명과 다음 행동이 분명한가? (문서: 요약이 분명하고 행동을 지시하지 않았는가)",
}
SCORE_VALUES = (0, 1, 2)


def add_fastapi_to_path() -> None:
    """서비스 코드(classifier, generator, policy, chain)를 같은 방식으로 불러오기 위한 경로."""
    if str(FASTAPI_DIR) not in sys.path:
        sys.path.insert(0, str(FASTAPI_DIR))


# ═════════════════════════ 1. 입력: test.csv에서 표본 뽑기 ═════════════════════════
def read_labels(topic: str, kind: str) -> list[str]:
    path = DATASET_DIR / topic / kind / "labels.json"
    return json.loads(path.read_text(encoding="utf-8"))["labels"]


def read_test_rows(topic: str, kind: str) -> list[dict]:
    path = DATASET_DIR / topic / kind / "test.csv"
    if not path.exists():
        raise FileNotFoundError(f"test 파일이 없습니다: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    missing = {"id", "text", "label", "group_id", "source"} - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{path}: 필요한 열이 없습니다: {sorted(missing)}")
    return rows


def sample_kind(rows: list[dict], labels: list[str], n: int, seed_key: str) -> list[dict]:
    """한 종류의 test에서 n건을 뽑는다. 라벨을 가능한 한 균등하게 나누고, 라벨 안에서는 서로 다른 group_id를 먼저 고른다.
    (같은 그룹의 문장은 서로 비슷해서 한 그룹에서 여러 건을 뽑으면 표본이 한쪽으로 쏠린다)
    seed_key가 같으면 항상 같은 행이 뽑힌다."""
    rng = random.Random(seed_key)
    base, extra = divmod(n, len(labels))
    order = list(labels)
    rng.shuffle(order)                  # n이 라벨 수로 나누어떨어지지 않을 때 어느 라벨이 한 건 더 받을지
    quota = {label: base + (1 if index < extra else 0) for index, label in enumerate(order)}

    picked = []
    for label in labels:
        pool = [row for row in rows if row["label"] == label]
        if quota[label] > len(pool):
            raise ValueError(f"'{label}' 라벨의 test가 {len(pool)}건뿐이라 {quota[label]}건을 뽑을 수 없습니다.")
        rng.shuffle(pool)
        chosen, groups = [], set()
        for row in pool:
            if len(chosen) < quota[label] and row["group_id"] not in groups:
                chosen.append(row)
                groups.add(row["group_id"])
        for row in pool:                # 그룹이 모자라면 남은 칸을 채운다.
            if len(chosen) < quota[label] and row not in chosen:
                chosen.append(row)
        picked += chosen
    return picked


def sample_cases(topic: str, per_kind: int, seed: int, kinds=KINDS, exclude=frozenset()) -> list[dict]:
    """토픽의 종류마다 per_kind건씩 뽑아 비교 입력(case) 목록을 만든다. 종류별 건수가 같아서 어느 데이터도 더 반영되지 않는다.
    exclude에 든 case_id('종류/행 id')는 뽑지 않는다. (이전 비교에 쓴 입력을 빼고 새 입력만 뽑을 때)"""
    cases = []
    for kind in kinds:
        labels = read_labels(topic, kind)
        rows = [row for row in read_test_rows(topic, kind) if f"{kind}/{row['id']}" not in exclude]
        if per_kind > len(rows):
            raise ValueError(f"{topic}/{kind}에서 쓸 수 있는 test가 {len(rows)}건이라 {per_kind}건을 뽑을 수 없습니다."
                             + (" (이전 비교에 쓴 입력을 제외했습니다)" if exclude else ""))
        for row in sample_kind(rows, labels, per_kind, f"{seed}:{topic}:{kind}"):
            cases.append({
                "case_id": f"{kind}/{row['id']}", "topic": topic, "kind": kind, "id": row["id"], "text": row["text"],
                "expected_label": row["label"], "group_id": row["group_id"], "source": row["source"],
                # test.csv에는 없는 항목이다. 비교 입력에 필요 안내와 금지 주장을 따로 붙였다면 여기에 채운다.
                "required_points": [], "forbidden_claims": [],
            })
    return cases


def validate_cases(cases: list[dict], max_chars: int) -> None:
    """빈 입력, 중복, 너무 긴 입력을 유료 호출 전에 걸러낸다."""
    if not cases:
        raise ValueError("비교 입력이 비어 있습니다.")
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("비교 입력의 case_id가 겹칩니다.")
    for case in cases:
        if not case["text"].strip():
            raise ValueError(f"{case['case_id']}: 빈 입력입니다.")
        if len(case["text"]) > max_chars:
            raise ValueError(f"{case['case_id']}: 입력이 {max_chars}자를 넘습니다. 생성 모델에는 잘린 원문이 가므로 비교에서 제외하세요.")


# ═════════════════════════ 2. 파일 도우미 ═════════════════════════
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def policy_fingerprint() -> str:
    """정책과 프롬프트 문구의 지문. 비교 결과를 보고 프롬프트를 고쳤다면 이 값이 바뀐다.
    줄바꿈 방식(CRLF/LF)에 따라 달라지지 않게 정리한 뒤 계산한다."""
    text = (FASTAPI_DIR / "policy.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    return sha256_text(text)[:16]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:     # BOM이 있어야 Excel에서 한글이 깨지지 않는다.
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


# ═════════════════════════ 3. 객관 지표 (분모를 분명히) ═════════════════════════
def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_provider(records: list[dict], provider: str) -> dict:
    """제공자 한 곳의 호출 통계.
      오류율, 형식 유효율 : 그 제공자의 전체 호출이 분모
      응답 시간            : 예외 없이 돌아온(error 없는) 호출만
    한 입력을 여러 번 보내면 기록은 늘지만 입력 수가 늘어난 것은 아니므로 고유 case_id를 따로 센다."""
    subset = [record for record in records if record["provider"] == provider]
    success = [record for record in subset if not record["error"]]
    calls = len(subset)
    latencies = [record["latency_seconds"] for record in success if record["latency_seconds"] is not None]
    speeds = [record["tokens_per_sec"] for record in success if record.get("tokens_per_sec")]
    outputs = [record["output_tokens"] for record in success if record["output_tokens"] is not None]
    structure = [record["metrics"]["summary_structure_ok"] for record in subset
                 if record.get("metrics", {}).get("summary_structure_ok") is not None]
    return {
        "summary_structure_ok_rate": sum(structure) / len(structure) if structure else None,   # documents 요약만 해당
        "cases": len({record["case_id"] for record in subset}),
        "calls": calls,
        "errors": calls - len(success),
        "error_rate_all_calls": (calls - len(success)) / calls if calls else None,
        "format_valid_rate_all_calls": sum(bool(record["output_format_valid"]) for record in subset) / calls if calls else None,
        "mean_latency_seconds_success_only": statistics.mean(latencies) if latencies else None,
        "median_latency_seconds_success_only": statistics.median(latencies) if latencies else None,
        "p95_latency_seconds_success_only": percentile(latencies, 0.95) if latencies else None,
        "mean_output_tokens_success_only": statistics.mean(outputs) if outputs else None,
        "median_tokens_per_sec_success_only": statistics.median(speeds) if speeds else None,
    }


# ═════════════════════════ 4. 채점표 ═════════════════════════
def format_response(record: dict) -> str:
    """사람이 읽을 답변. 성공이면 answer와 next_steps를 보기 좋게, 형식이 깨졌으면 모델이 낸 원문을, 호출 실패면 표시만 한다."""
    if record["status"] == "success":
        if record.get("summary_items") is not None:       # 문서 요약
            return json.dumps({"answer": record["answer"], "summary_items": record["summary_items"]}, ensure_ascii=False, indent=2)
        return json.dumps({"answer": record["answer"], "next_steps": record["next_steps"]}, ensure_ascii=False, indent=2)
    if record.get("response_text"):
        return record["response_text"]
    return "(호출 실패: 응답 없음. 채점하지 않습니다)"


SHEET_BLIND = ["item_id", "topic", "case_id", "origin", "expected_label", "predicted_label", "text", "response",
               "required_points", "forbidden_claims"]
SHEET_SCORES = ["reviewer", *CRITERIA, "notes"]
SHEET_UNBLIND_EXTRA = ["provider", "model", "repeat"]
KEY_FIELDS = ["item_id", "topic", "case_id", "origin", "repeat", "provider", "model", "status", "error",
              "latency_seconds", "output_tokens"]


def write_sheets(run_dir: Path, records: list[dict], seed_key: str, unblind: bool = False) -> None:
    """채점표(human_scores.csv)와 정답 키(answer_key.csv)를 쓴다.
    채점표에는 어느 모델의 답변인지 드러내지 않고(지연·토큰 수도 뺀다) 순서도 섞어서, 모델 이름이 점수에 영향을 주지 않게 한다.
    item_id가 채점표와 키를 잇는다. 점수 칸은 비워 둔다. 사람이 원문과 답변을 비교한 뒤 채운다."""
    rng = random.Random(seed_key)
    order = list(range(len(records)))
    rng.shuffle(order)
    sheet_rows, key_rows = [], []
    for number, index in enumerate(order, start=1):
        record = records[index]
        item_id = f"item-{number:04d}"
        row = {
            "item_id": item_id, "topic": record["topic"], "case_id": record["case_id"], "origin": record["kind"],
            "expected_label": record["expected_label"], "predicted_label": record["classification"].get("label"),
            "text": record["input_text"], "response": format_response(record),
            "required_points": json.dumps(record["required_points"], ensure_ascii=False),
            "forbidden_claims": json.dumps(record["forbidden_claims"], ensure_ascii=False),
            "provider": record["provider"], "model": record["model"], "repeat": record["repeat"],
        }
        sheet_rows.append(row)
        key_rows.append({**{key: record.get(key) for key in KEY_FIELDS}, "item_id": item_id, "origin": record["kind"]})
    fields = SHEET_BLIND + (SHEET_UNBLIND_EXTRA if unblind else []) + SHEET_SCORES
    write_csv(run_dir / "human_scores.csv", fields, sheet_rows)
    write_csv(run_dir / "answer_key.csv", KEY_FIELDS, key_rows)


def parse_score(value) -> int | None:
    """칸의 값을 점수로. 빈칸은 None(채점 안 함). 0, 1, 2가 아니면 ValueError."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"점수가 숫자가 아닙니다: {text!r}") from None
    if number not in SCORE_VALUES:
        raise ValueError(f"점수는 0, 1, 2 중 하나여야 합니다: {text!r}")
    return int(number)
