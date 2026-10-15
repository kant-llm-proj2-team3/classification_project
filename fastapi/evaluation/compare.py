"""같은 입력과 프롬프트로 생성 모델들을 호출하고, 비교 기록과 사람용 채점표를 저장한다.

입력은 dataset/<토픽>/<종류>/test.csv뿐이다. 토픽마다 4종류(basic, basic_add, advanced, advanced_add)에서 --per-kind N건씩
그 자리에서 뽑는다. (종류마다 같은 건수를 뽑아서 어느 데이터도 더 반영되지 않는다. 같은 --seed면 같은 행이 뽑힌다.)

실행은 프로젝트 루트에서 한다.
  uv run python fastapi/evaluation/compare.py --per-kind 3 --dry-run        # 뽑힌 표본과 호출 수만 확인 (호출 없음)
  uv run python fastapi/evaluation/compare.py --per-kind 3 --repeats 2 --yes # 실제 실행 (OpenAI 유료 호출이 포함된다)

분류와 정책 적용은 서비스(/chat, /compare)와 같은 코드(chain.common_steps)를 쓰고, 생성도 서비스와 같은 생성기를 쓴다.
비교를 위해 다른 생성 경로를 따로 만들지 않는다. 결과는 artifacts/generation_compare/<이름>/에 저장된다.
  results.jsonl   호출 1번 = 1행. 중단돼도 끝난 호출은 남도록 호출마다 바로 쓴다.
  config.json     실행 설정, 뽑은 표본(case 목록), 정책 지문, 분류 모델 정보
  summary.json    제공자별 객관 지표 (오류율, 형식 유효율, 응답 시간 ...)
  human_scores.csv  사람이 채점할 표. 어느 모델의 답변인지 숨겨져 있다. (--unblind로 공개)
  answer_key.csv  채점표의 item_id가 어느 모델·반복인지 알려 주는 키. 채점이 끝날 때까지 보지 않는다.
채점이 끝나면 report.py로 점수를 모아 본다.

주의: test 데이터를 쓰므로 이 결과를 보고 정책·프롬프트를 고치면 같은 test로 다시 비교한 결과는 독립적인 평가가 아니다.
정책이 바뀐 뒤 같은 입력으로 다시 비교하려면 --allow-changed-prompt가 필요하고, 그 사실이 config.json에 남는다.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # 이 파일을 직접 실행해도 evaluation 패키지를 찾게 한다.

from evaluation import eval_common as common  # noqa: E402

common.add_fastapi_to_path()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-kind", type=int, required=True, help="토픽의 종류(basic, basic_add, advanced, advanced_add)마다 뽑을 건수 N")
    parser.add_argument("--topic", choices=[*common.TOPICS, "both"], default="both")
    parser.add_argument("--kinds", default=",".join(common.KINDS), help="뽑을 종류. 기본은 네 가지 모두 (쉼표로 구분)")
    parser.add_argument("--repeats", type=int, default=2, help="같은 입력을 제공자마다 몇 번 호출할지 (기본 2)")
    parser.add_argument("--providers", default="openai,ollama", help="비교할 생성 모델 (쉼표로 구분)")
    parser.add_argument("--seed", type=int, default=42, help="표본을 뽑는 시드. 같으면 같은 행이 뽑힌다.")
    parser.add_argument("--name", help="결과 폴더 이름 (기본: compare_날짜시각). 이미 있으면 멈춘다.")
    parser.add_argument("--out-root", type=Path, default=common.DEFAULT_OUT_ROOT, help="결과를 보관할 폴더")
    parser.add_argument("--unblind", action="store_true", help="채점표에 모델 이름을 공개한다. (기본은 숨김)")
    parser.add_argument("--dry-run", action="store_true", help="표본과 호출 수만 보여주고 아무것도 호출하거나 저장하지 않는다.")
    parser.add_argument("--yes", action="store_true", help="유료 호출이 포함된 실제 실행을 확인했다는 표시")
    parser.add_argument("--fresh", action="store_true", help="이전 비교(같은 --out-root)에 쓴 입력은 빼고 새 입력만 뽑는다. 정책을 고친 뒤 독립적으로 다시 비교할 때 쓴다.")
    parser.add_argument("--allow-changed-prompt", action="store_true", help="이전 비교 이후 정책이 바뀌었어도 같은 입력으로 다시 비교한다.")
    return parser


# ═════════════════════════ 계획 ═════════════════════════
def seen_case_ids(out_root: Path) -> set:
    return {case["case_id"] for config in previous_runs(out_root) for case in config.get("cases", [])}


def build_cases(args) -> dict[str, list[dict]]:
    topics = common.TOPICS if args.topic == "both" else (args.topic,)
    kinds = tuple(kind.strip() for kind in args.kinds.split(",") if kind.strip())
    unknown = set(kinds) - set(common.KINDS)
    if unknown or not kinds:
        raise ValueError(f"종류는 {common.KINDS} 중에서 골라야 합니다: {sorted(unknown)}")
    if args.per_kind < 1 or args.repeats < 1:
        raise ValueError("--per-kind와 --repeats는 1 이상이어야 합니다.")
    exclude = seen_case_ids(args.out_root) if args.fresh else frozenset()
    if exclude:
        print(f"--fresh: 이전 비교에 쓴 입력 {len(exclude)}건을 제외하고 뽑습니다.")
    return {topic: common.sample_cases(topic, args.per_kind, args.seed, kinds, exclude) for topic in topics}


def print_plan(cases_by_topic: dict, providers: list[str], repeats: int) -> int:
    total_cases = sum(len(cases) for cases in cases_by_topic.values())
    print(f"뽑힌 입력: 토픽 {len(cases_by_topic)}개, 총 {total_cases}건 (test.csv에서만)")
    for topic, cases in cases_by_topic.items():
        by_kind: dict = {}
        for case in cases:
            by_kind.setdefault(case["kind"], Counter())[case["expected_label"]] += 1
        for kind, labels in by_kind.items():
            print(f"  {topic}/{kind}: {sum(labels.values())}건  라벨 {dict(labels)}")
    calls = total_cases * repeats * len(providers)
    print(f"호출: 입력 {total_cases} × 반복 {repeats} × 모델 {len(providers)} = {calls}회")
    for provider in providers:
        print(f"  - {provider}: {total_cases * repeats}회" + ("  (유료)" if provider == "openai" else ""))
    return calls


def previous_runs(out_root: Path) -> list[dict]:
    configs = []
    for path in sorted(Path(out_root).glob("*/config.json")) if Path(out_root).exists() else []:
        try:
            configs.append({"name": path.parent.name, **json.loads(path.read_text(encoding="utf-8"))})
        except ValueError:
            continue
    return configs


def check_prompt_reuse(out_root: Path, case_ids: set, fingerprint: str, allow: bool) -> list[str]:
    """이전 비교와 같은 입력을 쓰는데 정책(프롬프트)이 바뀌었는지 확인한다. 바뀌었다면 test가 개발에 쓰인 것이므로 막는다."""
    changed = []
    for config in previous_runs(out_root):
        earlier = {case["case_id"] for case in config.get("cases", [])}
        if earlier & case_ids and config.get("policy_fingerprint") != fingerprint:
            changed.append(config["name"])
    if changed and not allow:
        raise SystemExit(
            f"이전 비교({', '.join(changed)})와 같은 입력이 겹치는데 그 뒤 정책/프롬프트(policy.py)가 바뀌었습니다.\n"
            "  이전 답변을 보고 프롬프트를 고쳤다면 이 test 입력은 개발에 쓰인 것이라 독립적인 평가가 아닙니다.\n"
            "  그래도 비교하려면 --allow-changed-prompt를 붙이세요. (그 사실이 config.json에 기록됩니다)\n"
            "  독립적인 결과가 필요하면 --fresh를 붙여 이전에 쓴 입력을 뺀 새 입력만 뽑으세요.")
    return changed


def git_commit() -> str | None:
    try:
        done = subprocess.run(["git", "-C", str(common.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None


def generator_settings(generators: dict, providers: list[str]) -> dict:
    keys = ("model", "max_output_tokens", "timeout", "temperature", "reasoning_effort", "num_ctx", "keep_alive", "base_url")
    return {name: {key: getattr(generators[name], key) for key in keys if hasattr(generators[name], key)} for name in providers}


# ═════════════════════════ 실행 ═════════════════════════
def prepare(cases_by_topic: dict, bundles: dict) -> dict[str, dict]:
    """모든 입력을 서비스와 같은 코드로 분류하고 프롬프트를 만든다. 하나라도 실패하면 유료 호출 전에 멈춘다."""
    from chain import common_steps

    prepared = {}
    for topic, cases in cases_by_topic.items():
        classify, apply_policy = common_steps({topic: bundles[topic]})
        pipeline = classify | apply_policy
        for case in cases:
            item = pipeline.invoke({"text": case["text"], "dataset": topic})
            if item["classification"]["status"] != "success" or item["prompt"] is None:
                raise RuntimeError(f"{case['case_id']}: 분류 또는 프롬프트 준비에 실패했습니다: "
                                   f"{item['classification'].get('error') or item.get('skip_reason')}")
            prepared[case["case_id"]] = item
    return prepared


def make_record(case: dict, item: dict, repeat: int, provider: str, result: dict) -> dict:
    metrics = result.get("metrics", {})
    latency_ms = metrics.get("latency_ms")
    return {
        "topic": case["topic"], "kind": case["kind"], "case_id": case["case_id"], "repeat": repeat,
        "provider": provider, "model": metrics.get("model"),
        "expected_label": case["expected_label"], "classification": item["classification"],
        "input_text": item["text"], "prompt": item["prompt"],
        "required_points": case["required_points"], "forbidden_claims": case["forbidden_claims"],
        "status": result["status"], "error": result.get("error"),
        "latency_seconds": latency_ms / 1000 if latency_ms is not None else None,
        "input_tokens": metrics.get("input_tokens"), "output_tokens": metrics.get("output_tokens"),
        "tokens_per_sec": metrics.get("tokens_per_sec"),
        "generation_complete": metrics.get("response_status") == "completed",
        "finish_reason": metrics.get("done_reason") or metrics.get("response_status"),
        "output_format_valid": result["status"] == "success",
        "response_text": result.get("raw_text") or "", "answer": result.get("answer"), "next_steps": result.get("next_steps"),
        "summary_items": result.get("summary_items"),
        "metrics": metrics,
    }


async def run_comparison(args, generators: dict | None = None, bundles: dict | None = None) -> dict:
    cases_by_topic = build_cases(args)
    providers = [name.strip() for name in args.providers.split(",") if name.strip()]
    from chain import MAX_INPUT_CHARS, note_summary_structure
    from generator import PROVIDERS

    unknown = set(providers) - set(PROVIDERS)
    if unknown or not providers:
        raise ValueError(f"모델은 {PROVIDERS} 중에서 골라야 합니다: {sorted(unknown)}")
    for cases in cases_by_topic.values():
        common.validate_cases(cases, MAX_INPUT_CHARS)
    calls = print_plan(cases_by_topic, providers, args.repeats)
    if args.dry_run:
        print("\n--dry-run: 여기까지만 확인했습니다. 호출도 저장도 하지 않았습니다.")
        return {"dry_run": True, "calls": calls}
    if not args.yes:
        raise SystemExit("\n실제로 호출하려면 --yes를 붙이세요. (OpenAI 호출은 요금이 발생합니다.)")

    out_dir = Path(args.out_root) / (args.name or time.strftime("compare_%Y%m%d_%H%M%S"))
    if out_dir.exists():
        raise FileExistsError(f"결과 폴더가 이미 있습니다. 덮어쓰지 않습니다: {out_dir}  (--name을 바꾸세요)")
    fingerprint = common.policy_fingerprint()
    case_ids = {case["case_id"] for cases in cases_by_topic.values() for case in cases}
    changed_after = check_prompt_reuse(args.out_root, case_ids, fingerprint, args.allow_changed_prompt)

    from classifier import close_classifiers, load_topic, resolve_device
    from generator import create_generators

    generators = generators or create_generators()
    for name in providers:
        if not generators[name].describe()["configured"]:
            raise SystemExit(f"{name} 생성 모델 설정이 .env에 없습니다.")

    owns_bundles = bundles is None
    if owns_bundles:
        device = resolve_device(os.getenv("CLASSIFIER_DEVICE") or None)
        bundles = {topic: load_topic(topic, device) for topic in cases_by_topic}
    try:
        for topic, bundle in bundles.items():
            if topic in cases_by_topic and bundle.error:
                raise RuntimeError(f"{topic} 분류 모델을 불러오지 못했습니다: {bundle.error}")
        prepared = prepare(cases_by_topic, bundles)       # 여기까지 실패가 없어야 유료 호출을 시작한다.
        preloaded = await generators["ollama"].preload() if "ollama" in providers else None
        out_dir.mkdir(parents=True, exist_ok=False)
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        config = {
            "name": out_dir.name, "started_at": started, "finished_at": None, "args": {
                "per_kind": args.per_kind, "topic": args.topic, "kinds": args.kinds, "repeats": args.repeats,
                "providers": providers, "seed": args.seed, "unblind": args.unblind},
            "data": "dataset/<토픽>/<종류>/test.csv (test만 사용)", "policy_fingerprint": fingerprint,
            "prompt_changed_since": changed_after, "git_commit": git_commit(),
            "classifiers": {topic: {"model": bundles[topic].model_key, "run_name": bundles[topic].run_name} for topic in cases_by_topic},
            "generators": generator_settings(generators, providers), "ollama_preloaded": preloaded,
            "cases": [{key: case[key] for key in ("case_id", "topic", "kind", "id", "group_id", "expected_label", "source")}
                      for cases in cases_by_topic.values() for case in cases],
        }
        (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        records = []
        number = 0
        with (out_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
            for cases in cases_by_topic.values():
                for case in cases:
                    item = prepared[case["case_id"]]
                    for repeat in range(1, args.repeats + 1):
                        # 호출 순서가 응답 시간에 영향을 주지 않도록 입력과 반복마다 시작하는 모델을 돌려 가며 호출한다.
                        shift = (number + repeat) % len(providers)
                        for provider in providers[shift:] + providers[:shift]:
                            result = await generators[provider].generate(item["prompt"], keep_raw=True, topic=case["topic"])
                            note_summary_structure(item["classification"]["label"], result)      # 문서 요약이 항목 구성을 지켰는지 (서비스와 같은 검사)
                            record = make_record(case, item, repeat, provider, result)
                            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                            stream.flush()
                            records.append(record)
                    number += 1
                    print(f"  {number}/{len(case_ids)} {case['case_id']} 저장 ({args.repeats}회 × {len(providers)}개 모델)")

        summary = {
            "cases": len(case_ids), "repeats": args.repeats, "calls": len(records), "human_quality_scored": False,
            "providers": {provider: common.summarize_provider(records, provider) for provider in providers},
            "by_topic": {topic: {provider: common.summarize_provider([r for r in records if r["topic"] == topic], provider)
                                 for provider in providers} for topic in cases_by_topic},
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        common.write_sheets(out_dir, records, seed_key=f"{args.seed}:{out_dir.name}", unblind=args.unblind)
        config["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        if owns_bundles:
            close_classifiers(bundles)
        for generator in generators.values():
            await generator.aclose()
    summary["out_dir"] = str(out_dir)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = asyncio.run(run_comparison(args))
    if summary.get("dry_run"):
        return
    print("\n" + json.dumps({key: value for key, value in summary.items() if key != "by_topic"}, ensure_ascii=False, indent=2))
    print("\n저장 위치:", summary["out_dir"])
    print("human_scores.csv에 원문과 답변을 비교한 점수와 이유를 적은 뒤(answer_key.csv는 채점이 끝날 때까지 열지 않는다),")
    print(f"  uv run python fastapi/evaluation/report.py {summary['out_dir']}")


if __name__ == "__main__":
    main()
