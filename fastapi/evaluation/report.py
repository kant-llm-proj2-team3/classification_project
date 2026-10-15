"""채점이 끝난 비교 결과를 검증하고 모아서 본다.

compare.py가 만든 폴더를 읽는다. 모델을 호출하지 않는다.
  uv run python fastapi/evaluation/report.py artifacts/generation_compare/<이름>

읽는 파일: results.jsonl(객관 지표), answer_key.csv(채점표의 item_id가 어느 모델인지), human_scores*.csv(사람 채점)
채점자가 여러 명이면 human_scores.csv를 복사해 human_scores_<이름>.csv로 저장하고 각자 채우면 된다. (reviewer 칸에도 이름을 적는다)
만드는 파일: report.json, report.md, scores_long.csv(채점 1건 = 1행, 모델 이름이 붙은 분석용 표)

채점표를 검사해서 문제를 알려 준다: 0·1·2가 아닌 점수, 채점자 이름 없음, 같은 항목 중복 채점, 일부 항목만 채점, 모르는 item_id,
답변 원문이 수정된 행, 호출이 실패한 항목에 준 점수. 채점하지 않은 빈칸은 0점으로 바꾸지 않고 '채점 안 함'으로 센다.
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import eval_common as common  # noqa: E402

TIE_TOLERANCE = 1e-9


# ═════════════════════════ 1. 읽기와 검사 ═════════════════════════
def reviewer_from_filename(path: Path) -> str:
    stem = path.stem                                   # human_scores / human_scores_alice
    return stem.removeprefix("human_scores").lstrip("_-") or ""


def load_scores(run_dir: Path, records: list[dict], key_rows: list[dict]) -> tuple[list[dict], dict]:
    """채점표들을 읽어 채점 건(item_id, reviewer, 점수)으로 만든다. 검사에서 나온 문제는 issues에 모은다."""
    key = {row["item_id"]: row for row in key_rows}
    by_call = {(r["case_id"], str(r["repeat"]), r["provider"]): r for r in records}
    issues: dict = {name: [] for name in ("invalid_score", "missing_reviewer", "duplicate", "partial", "unknown_item",
                                           "response_edited", "scored_failed_call")}
    scored: dict = {}
    sheets = sorted(Path(run_dir).glob("human_scores*.csv"))
    for sheet in sheets:
        for number, row in enumerate(common.read_csv(sheet), start=2):
            where = f"{sheet.name}:{number}"
            values, bad = {}, False
            for column in common.CRITERIA:
                try:
                    values[column] = common.parse_score(row.get(column))
                except ValueError as error:
                    issues["invalid_score"].append(f"{where} {column}: {error}")
                    values[column] = None
                    bad = True
            if all(value is None for value in values.values()) and not bad:
                continue                                  # 아직 채점하지 않은 행
            item_id = (row.get("item_id") or "").strip()
            if item_id not in key:
                issues["unknown_item"].append(f"{where} item_id={item_id!r}")
                continue
            reviewer = (row.get("reviewer") or "").strip() or reviewer_from_filename(sheet)
            if not reviewer:
                issues["missing_reviewer"].append(f"{where} {item_id}")
                continue
            if (item_id, reviewer) in scored:
                issues["duplicate"].append(f"{where} {item_id} / {reviewer} (먼저 읽은 채점만 씁니다)")
                continue
            info = key[item_id]
            record = by_call.get((info["case_id"], str(info["repeat"]), info["provider"]))
            if record is not None and (row.get("response") or "").strip() != common.format_response(record).strip():
                issues["response_edited"].append(f"{where} {item_id}")
            if any(value is None for value in values.values()):
                issues["partial"].append(f"{where} {item_id}: 빈 항목 {[c for c, v in values.items() if v is None]}")
            if info["status"] != "success":
                issues["scored_failed_call"].append(f"{where} {item_id} ({info['provider']} {info['status']})")
            scored[(item_id, reviewer)] = {
                "item_id": item_id, "reviewer": reviewer, "scores": values, "notes": (row.get("notes") or "").strip(),
                "topic": info["topic"], "case_id": info["case_id"], "origin": info["origin"], "repeat": info["repeat"],
                "provider": info["provider"], "model": info["model"], "status": info["status"],
            }
    return list(scored.values()), {name: items for name, items in issues.items() if items}


def total_score(scores: dict) -> int | None:
    values = list(scores.values())
    return None if any(value is None for value in values) else sum(values)


# ═════════════════════════ 2. 집계 ═════════════════════════
def mean_or_none(values: list) -> float | None:
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def human_by_provider(items: list[dict], providers: list[str]) -> dict:
    result = {}
    for provider in providers:
        subset = [item for item in items if item["provider"] == provider]
        criteria = {}
        for column in common.CRITERIA:
            values = [item["scores"][column] for item in subset if item["scores"][column] is not None]
            criteria[column] = {"n": len(values), "mean": mean_or_none(values),
                                "distribution": {str(score): values.count(score) for score in common.SCORE_VALUES}}
        totals = [total_score(item["scores"]) for item in subset]
        zero_claims = [item["item_id"] for item in subset if item["scores"]["no_unsupported_claims_0_2"] == 0]
        result[provider] = {
            "scored_items": len(subset), "criteria": criteria, "mean_total_0_8": mean_or_none(totals),
            "unsupported_claims_zero": {"count": len(zero_claims), "item_ids": sorted(set(zero_claims))},
        }
    return result


def human_breakdown(items: list[dict], providers: list[str], field: str) -> dict:
    """provider × (topic 또는 origin)별 평균 총점. 어느 데이터에서 모델 차이가 큰지 본다."""
    groups = sorted({item[field] for item in items})
    return {group: {provider: {"n": len([i for i in items if i[field] == group and i["provider"] == provider
                                         and total_score(i["scores"]) is not None]),
                               "mean_total_0_8": mean_or_none([total_score(i["scores"]) for i in items
                                                                if i[field] == group and i["provider"] == provider])}
                    for provider in providers} for group in groups}


def case_means(items: list[dict], provider: str) -> dict:
    """case별 총점 평균. 같은 입력을 여러 번 호출하고 여러 명이 채점했다면 그 값들의 평균이다."""
    values: dict = {}
    for item in items:
        total = total_score(item["scores"])
        if item["provider"] == provider and total is not None:
            values.setdefault(item["case_id"], []).append(total)
    return {case: statistics.mean(totals) for case, totals in values.items()}


def pairwise(items: list[dict], providers: list[str]) -> list[dict]:
    """모델 두 개를 같은 입력끼리 비교한다. 입력마다 총점 평균을 비교해 이긴 입력 수, 비긴 수, 진 입력 수를 센다."""
    rows = []
    for first, second in combinations(providers, 2):
        a, b = case_means(items, first), case_means(items, second)
        shared = sorted(set(a) & set(b))
        diffs = [a[case] - b[case] for case in shared]
        rows.append({
            "a": first, "b": second, "cases_compared": len(shared),
            "a_wins": sum(diff > TIE_TOLERANCE for diff in diffs), "ties": sum(abs(diff) <= TIE_TOLERANCE for diff in diffs),
            "b_wins": sum(diff < -TIE_TOLERANCE for diff in diffs), "mean_difference_a_minus_b": mean_or_none(diffs),
        })
    return rows


def repeat_consistency(items: list[dict], providers: list[str]) -> dict:
    """같은 입력을 여러 번 호출했을 때 점수가 얼마나 흔들리는가. (반복별 총점의 표준편차를 입력별로 구해 평균)"""
    result = {}
    for provider in providers:
        per_case: dict = {}
        for item in items:
            total = total_score(item["scores"])
            if item["provider"] == provider and total is not None:
                per_case.setdefault(item["case_id"], {}).setdefault(item["repeat"], []).append(total)
        spreads = [statistics.pstdev([statistics.mean(v) for v in repeats.values()])
                   for repeats in per_case.values() if len(repeats) >= 2]
        result[provider] = {"cases_with_repeats": len(spreads), "mean_std_of_total": mean_or_none(spreads)}
    return result


def quadratic_weighted_kappa(first: list[int], second: list[int]) -> float | None:
    """0·1·2 점수의 채점자 간 일치도. 1이면 완전히 일치, 0이면 우연 수준. 한쪽이 모든 항목에 같은 점수를 주면 계산할 수 없다(None)."""
    categories = common.SCORE_VALUES
    size, count = len(categories), len(first)
    if not count:
        return None
    observed = [[0.0] * size for _ in range(size)]
    for a, b in zip(first, second, strict=True):
        observed[a][b] += 1
    hist_a = [sum(row) for row in observed]
    hist_b = [sum(observed[i][j] for i in range(size)) for j in range(size)]
    numerator = denominator = 0.0
    for i in range(size):
        for j in range(size):
            weight = (i - j) ** 2 / (size - 1) ** 2
            numerator += weight * observed[i][j]
            denominator += weight * hist_a[i] * hist_b[j] / count
    return None if denominator == 0 else 1 - numerator / denominator


def reviewer_agreement(items: list[dict]) -> list[dict]:
    reviewers = sorted({item["reviewer"] for item in items})
    by_reviewer = {name: {item["item_id"]: item for item in items if item["reviewer"] == name} for name in reviewers}
    rows = []
    for first, second in combinations(reviewers, 2):
        shared = sorted(set(by_reviewer[first]) & set(by_reviewer[second]))
        for column in common.CRITERIA:
            pairs = [(by_reviewer[first][i]["scores"][column], by_reviewer[second][i]["scores"][column]) for i in shared]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if not pairs:
                continue
            rows.append({
                "reviewers": [first, second], "criterion": column, "n": len(pairs),
                "exact_agreement": sum(a == b for a, b in pairs) / len(pairs),
                "mean_abs_diff": statistics.mean(abs(a - b) for a, b in pairs),
                "quadratic_weighted_kappa": quadratic_weighted_kappa([a for a, _ in pairs], [b for _, b in pairs]),
            })
    return rows


# ═════════════════════════ 3. 보고서 ═════════════════════════
def build_report(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    records = common.read_jsonl(run_dir / "results.jsonl")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    key_rows = common.read_csv(run_dir / "answer_key.csv")
    providers = config["args"]["providers"]
    items, issues = load_scores(run_dir, records, key_rows)
    scored_ids = {item["item_id"] for item in items}
    reviewers = Counter(item["reviewer"] for item in items)

    case_labels = {}
    for record in records:
        case_labels[record["case_id"]] = record["classification"].get("label") == record["expected_label"]

    return {
        "run": run_dir.name, "config": {k: config[k] for k in ("args", "policy_fingerprint", "prompt_changed_since", "classifiers", "generators", "git_commit", "started_at")},
        "objective": {
            "providers": {p: common.summarize_provider(records, p) for p in providers},
            "by_topic": {t: {p: common.summarize_provider([r for r in records if r["topic"] == t], p) for p in providers}
                         for t in sorted({r["topic"] for r in records})},
        },
        "coverage": {"calls": len(records), "scored_items": len(scored_ids), "unscored_items": len(records) - len(scored_ids),
                     "scored_ratio": len(scored_ids) / len(records) if records else None, "reviewers": dict(reviewers)},
        "issues": issues,
        "human": {
            "human_quality_scored": bool(items),
            "by_provider": human_by_provider(items, providers),
            "by_topic": human_breakdown(items, providers, "topic"),
            "by_origin": human_breakdown(items, providers, "origin"),
            "pairwise": pairwise(items, providers),
            "repeat_consistency": repeat_consistency(items, providers),
            "reviewer_agreement": reviewer_agreement(items),
        },
        # 분류 결과는 프롬프트를 고르는 입력일 뿐이다. 분류 성능은 별도 평가(serving_check.py)로 보고 여기서 섞어 해석하지 않는다.
        "classification_note": {"cases": len(case_labels), "predicted_equals_expected": sum(case_labels.values())},
        "_items": items,
    }


def fmt(value, digits=2) -> str:
    return "-" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def render_markdown(report: dict) -> str:
    out = [f"# 생성 모델 비교 보고서: {report['run']}", ""]
    config = report["config"]
    out += [f"- 입력: test만 사용, 종류마다 {config['args']['per_kind']}건 (시드 {config['args']['seed']}), 반복 {config['args']['repeats']}회",
            f"- 비교한 모델: {', '.join(config['args']['providers'])}", f"- 정책 지문: {config['policy_fingerprint']}"]
    if config["prompt_changed_since"]:
        out.append(f"- ⚠️ 이전 비교({', '.join(config['prompt_changed_since'])}) 이후 정책이 바뀐 뒤 같은 입력으로 다시 비교했습니다. 독립적인 평가가 아닙니다.")
    coverage = report["coverage"]
    out += ["", "## 채점 현황", f"- 호출 {coverage['calls']}건 중 채점된 항목 {coverage['scored_items']}건 ({fmt(coverage['scored_ratio'] and coverage['scored_ratio'] * 100, 1)}%), "
            f"채점자 {coverage['reviewers'] or '없음'}"]
    for name, items in report["issues"].items():
        out += [f"- ⚠️ **{name}** {len(items)}건"] + [f"  - {text}" for text in items[:10]] + ([f"  - … 외 {len(items) - 10}건"] if len(items) > 10 else [])

    out += ["", "## 객관 지표 (모델 호출 기록)", "오류율·형식 유효율은 전체 호출이 분모, 응답 시간은 오류 없이 돌아온 호출만 계산합니다.", "",
            "| 모델 | 입력 수 | 호출 | 오류율 | 형식 유효율 | 응답 시간 평균(초) | 중앙값 | p95 | 출력 토큰 평균 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for provider, s in report["objective"]["providers"].items():
        out.append(f"| {provider} | {s['cases']} | {s['calls']} | {fmt(s['error_rate_all_calls'], 3)} | {fmt(s['format_valid_rate_all_calls'], 3)} | "
                   f"{fmt(s['mean_latency_seconds_success_only'])} | {fmt(s['median_latency_seconds_success_only'])} | "
                   f"{fmt(s['p95_latency_seconds_success_only'])} | {fmt(s['mean_output_tokens_success_only'], 0)} |")

    structure = {p: s["summary_structure_ok_rate"] for p, s in report["objective"]["providers"].items() if s.get("summary_structure_ok_rate") is not None}
    if structure:
        out += ["", "문서(documents) 요약의 항목 구성 준수율(항목 수·순서·항목 이름): " + ", ".join(f"{p} {v:.3f}" for p, v in structure.items())]
    human = report["human"]
    if not human["human_quality_scored"]:
        out += ["", "## 사람 평가", "아직 채점된 항목이 없습니다. human_scores.csv를 채운 뒤 다시 실행하세요."]
        return "\n".join(out) + "\n"
    out += ["", "## 사람 평가 (0~2점, 빈칸은 채점 안 함)", "", "| 모델 | 채점 항목 | " + " | ".join(common.CRITERIA) + " | 총점 평균(0~8) | 없는 사실 0점 |",
            "|---|---:|" + "---:|" * (len(common.CRITERIA) + 2)]
    for provider, s in human["by_provider"].items():
        cells = " | ".join(f"{fmt(c['mean'])} (n={c['n']})" for c in s["criteria"].values())
        out.append(f"| {provider} | {s['scored_items']} | {cells} | {fmt(s['mean_total_0_8'])} | {s['unsupported_claims_zero']['count']} |")
    for title, field in (("토픽별 총점 평균", "by_topic"), ("데이터 종류별 총점 평균", "by_origin")):
        out += ["", f"### {title}", "", "| 구분 | " + " | ".join(report["config"]["args"]["providers"]) + " |", "|---|" + "---:|" * len(report["config"]["args"]["providers"])]
        for group, row in human[field].items():
            out.append(f"| {group} | " + " | ".join(f"{fmt(cell['mean_total_0_8'])} (n={cell['n']})" for cell in row.values()) + " |")
    out += ["", "### 같은 입력끼리 비교 (입력별 총점 평균)", "", "| 비교 | 비교한 입력 | A 우세 | 동점 | B 우세 | 평균 차이(A−B) |", "|---|---:|---:|---:|---:|---:|"]
    for row in human["pairwise"]:
        out.append(f"| {row['a']} vs {row['b']} | {row['cases_compared']} | {row['a_wins']} | {row['ties']} | {row['b_wins']} | {fmt(row['mean_difference_a_minus_b'])} |")
    out += ["", "### 반복 일관성 (같은 입력 반복 간 총점 표준편차 평균, 작을수록 안정)", ""]
    out += [f"- {provider}: {fmt(s['mean_std_of_total'])} (반복이 있는 입력 {s['cases_with_repeats']}건)" for provider, s in human["repeat_consistency"].items()]
    if human["reviewer_agreement"]:
        out += ["", "### 채점자 간 일치도", "", "| 채점자 | 항목 | n | 정확히 일치 | 평균 점수 차 | 가중 카파 |", "|---|---|---:|---:|---:|---:|"]
        for row in human["reviewer_agreement"]:
            out.append(f"| {' / '.join(row['reviewers'])} | {row['criterion']} | {row['n']} | {fmt(row['exact_agreement'])} | {fmt(row['mean_abs_diff'])} | {fmt(row['quadratic_weighted_kappa'])} |")
    out += ["", "## 해석할 때 주의", "- 입력 수가 적으면 평균 차이가 우연일 수 있습니다. 위 표의 n과 '같은 입력끼리 비교'의 우세·동점 수를 함께 보세요.",
            "- 형식이 맞는 답변도 정책에 없는 안내를 만들 수 있습니다. '없는 사실 0점' 항목은 notes의 근거와 함께 직접 확인하세요.",
            "- 분류 성능과 생성 품질은 다른 평가입니다. 여기서의 분류 결과는 프롬프트를 만든 입력일 뿐입니다."]
    return "\n".join(out) + "\n"


def write_outputs(run_dir: Path, report: dict) -> None:
    run_dir = Path(run_dir)
    items = report.pop("_items")
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    rows = [{"item_id": i["item_id"], "reviewer": i["reviewer"], "topic": i["topic"], "origin": i["origin"], "case_id": i["case_id"],
             "repeat": i["repeat"], "provider": i["provider"], "model": i["model"], **i["scores"],
             "total_0_8": total_score(i["scores"]), "notes": i["notes"]} for i in items]
    common.write_csv(run_dir / "scores_long.csv", ["item_id", "reviewer", "topic", "origin", "case_id", "repeat", "provider", "model",
                                                   *common.CRITERIA, "total_0_8", "notes"], rows)
    report["_items"] = items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="compare.py가 만든 결과 폴더")
    args = parser.parse_args()
    report = build_report(args.run_dir)
    write_outputs(args.run_dir, report)
    print(render_markdown(report))
    print("저장:", args.run_dir / "report.md", "/ report.json / scores_long.csv")
    if report["issues"].get("invalid_score") or report["issues"].get("missing_reviewer") or report["issues"].get("unknown_item"):
        raise SystemExit(1)       # 채점표에 고쳐야 할 오류가 있으면 실패로 끝낸다.


if __name__ == "__main__":
    main()
