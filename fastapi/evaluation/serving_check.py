"""서버가 실제로 쓰는 분류 모델 경로(classifier.py)로 한 split을 평가하고, 학습 때 저장된 결과와 같은지 확인한다.
(참고: project2-kit step11_evaluate_final.py. 확정한 분류기를 남겨 둔 test로 한 번 평가하고 덮어쓰지 않는다.)

모델 선택은 artifacts/<토픽>/advanced_merge/recommendation.json을 따르고, .env의 CLASSIFIER_MODEL_<토픽>으로 바꿀 수 있다. (서버와 같은 규칙)
  uv run python fastapi/evaluation/serving_check.py                          # validation으로 점검 (기본, 몇 번이든 가능)
  uv run python fastapi/evaluation/serving_check.py --split test --confirm-test   # test는 모델마다 한 번만

확인하는 것: 서빙 코드(로딩, 입력 처리, 예측)가 연구 단계의 저장된 예측과 같은 라벨·확률을 내는가. 다르면 서빙 경로에 문제가 있는 것이다.
test는 최종 보고에만 쓴다. 이 점수를 보고 모델이나 설정을 다시 고르지 않는다. 이미 같은 모델의 test 결과가 있어도,
이 도구의 test 결과 파일이 있으면 다시 실행하지 않고 멈춘다. (결과는 artifacts/serving_check/ 아래에 따로 저장되며 연구 산출물은 건드리지 않는다.)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import eval_common as common  # noqa: E402

common.add_fastapi_to_path()

DEFAULT_OUT_ROOT = common.ARTIFACTS_DIR / "serving_check"
PROB_TOLERANCE = 0.01          # 저장된 확률과 이 값 넘게 다르면 경고. CLM은 임베딩 캐시의 float16 때문에 1e-3 안팎으로 달라질 수 있다.


def classification_scores(actual: list[str], predicted: list[str], labels: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
    return {
        "n": len(actual), "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
                      for i, label in enumerate(labels)},
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(), "confusion_matrix_axes": "row=actual, column=predicted",
    }


def saved_macro_f1(best_dir: Path, split: str) -> float | None:
    """연구 단계에서 저장한 지표. validation은 최상위 macro_f1, test는 classifier 안에 있다."""
    path = best_dir / f"{split}_metrics.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("macro_f1") if "macro_f1" in data else (data.get("classifier") or {}).get("macro_f1")


def compare_with_saved(best_dir: Path, split: str, labels: list[str], served: dict) -> dict | None:
    """서빙 경로의 예측을 연구 단계에서 저장한 예측(<split>_predictions.csv)과 id별로 비교한다."""
    path = best_dir / f"{split}_predictions.csv"
    if not path.exists():
        return None
    saved = {row["id"]: row for row in common.read_csv(path)}
    shared = [row_id for row_id in served if row_id in saved]
    label_mismatch = [row_id for row_id in shared if served[row_id]["label"] != saved[row_id]["predicted"]]
    diffs = [max(abs(served[row_id]["probabilities"][label] - float(saved[row_id][f"prob_{label}"])) for label in labels) for row_id in shared]
    return {"saved_file": str(path.relative_to(common.ROOT)), "compared_rows": len(shared), "rows_only_in_one_side": len(set(served) ^ set(saved)),
            "label_mismatches": len(label_mismatch), "label_mismatch_ids": label_mismatch[:20],
            "max_abs_prob_diff": max(diffs) if diffs else None, "prob_diff_over_tolerance": sum(diff > PROB_TOLERANCE for diff in diffs)}


def check_topic(args, topic: str, device) -> dict:
    from classifier import choose_model, load_topic, predict_text

    key, _, best_dir = choose_model(topic)
    run_name = json.loads((best_dir / "best_info.json").read_text(encoding="utf-8"))["best"]["run_name"]
    out_dir = Path(args.out_root) / topic / args.data_dir
    stamp = "" if args.split == "test" else time.strftime("_%Y%m%d_%H%M%S")
    result_path = out_dir / f"{args.split}_{key}_{run_name}{stamp}.json"
    if args.split == "test":
        # test를 읽기 전에 막는다. 한 번 평가한 모델의 test를 다시 돌려 설정을 고르는 데 쓰지 않는다.
        if result_path.exists():
            raise FileExistsError(f"이 모델의 test 결과가 이미 있습니다. 덮어쓰지 않습니다: {result_path}")
        print(f"[{topic}] test 평가: {key} (run {run_name}). 이 점수는 최종 보고에만 씁니다.")

    bundle = load_topic(topic, device)
    if bundle.error:
        raise RuntimeError(f"{topic} 분류 모델을 불러오지 못했습니다: {bundle.error}")
    labels = common.read_labels(topic, args.data_dir)
    if labels != bundle.labels:
        raise ValueError(f"데이터의 라벨 순서 {labels}와 서빙 모델의 라벨 순서 {bundle.labels}가 다릅니다.")
    path = common.DATASET_DIR / topic / args.data_dir / f"{args.split}.csv"
    rows = common.read_csv(path)

    served, failures = {}, []
    for row in rows:
        result = predict_text(row["text"].strip(), bundle)
        if result["status"] != "success":
            failures.append({"id": row["id"], "error": result.get("error")})
            continue
        served[row["id"]] = result
    if failures:
        raise RuntimeError(f"{len(failures)}건의 예측에 실패했습니다: {failures[:3]}")

    actual = [row["label"] for row in rows]
    predicted = [served[row["id"]]["label"] for row in rows]
    report = {
        "topic": topic, "data_dir": args.data_dir, "split": args.split, "file": str(path.relative_to(common.ROOT)),
        "model": key, "run_name": run_name, "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "serving": classification_scores(actual, predicted, labels),
        "saved_macro_f1": saved_macro_f1(best_dir, args.split),
        "compared_with_saved_predictions": compare_with_saved(best_dir, args.split, labels, served),
        "mean_inference_ms": sum(served[row["id"]]["metrics"]["inference_ms"] for row in rows) / len(rows),
    }
    saved = report["saved_macro_f1"]
    report["macro_f1_diff_vs_saved"] = None if saved is None else report["serving"]["macro_f1"] - saved
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors = [{"id": row["id"], "actual": row["label"], "predicted": served[row["id"]]["label"],
               "confidence": served[row["id"]]["confidence"], "text": row["text"]}
              for row in rows if served[row["id"]]["label"] != row["label"]]
    common.write_csv(result_path.with_suffix(".errors.csv"), ["id", "actual", "predicted", "confidence", "text"], errors)
    report["result_path"] = str(result_path)
    return report


def verdict(report: dict) -> str:
    compared = report["compared_with_saved_predictions"]
    if compared is None:
        return "저장된 예측이 없어 비교하지 못했습니다."
    if compared["label_mismatches"] or compared["prob_diff_over_tolerance"] or compared["rows_only_in_one_side"]:
        return (f"⚠️ 저장된 결과와 다릅니다: 라벨 불일치 {compared['label_mismatches']}건, 확률 차이가 {PROB_TOLERANCE} 넘는 행 "
                f"{compared['prob_diff_over_tolerance']}건, 한쪽에만 있는 행 {compared['rows_only_in_one_side']}건. 서빙 코드(classifier.py)를 확인하세요.")
    return f"저장된 예측과 일치합니다. (라벨 모두 같음, 확률 최대 차이 {compared['max_abs_prob_diff']:.4f})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--topic", choices=[*common.TOPICS, "both"], default="both")
    parser.add_argument("--data-dir", default="advanced_merge", help="평가할 데이터 종류. 기본은 추천(recommendation.json)을 만든 advanced_merge")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--confirm-test", action="store_true", help="test는 모델마다 한 번만 평가한다는 것을 확인했다는 표시")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_test:
        raise SystemExit("test 평가는 모델마다 한 번만 합니다. 이 점수를 보고 모델이나 설정을 다시 고르지 않을 때만 --confirm-test를 붙이세요.")

    from classifier import close_classifiers, resolve_device

    device = resolve_device(os.getenv("CLASSIFIER_DEVICE") or None)
    topics = common.TOPICS if args.topic == "both" else (args.topic,)
    try:
        for topic in topics:
            report = check_topic(args, topic, device)
            s = report["serving"]
            print(f"\n[{topic}] {report['model']} (run {report['run_name']}) {args.data_dir}/{args.split} {s['n']}건")
            print(f"  서빙 경로: accuracy {s['accuracy']:.4f}, macro F1 {s['macro_f1']:.4f}, 평균 추론 {report['mean_inference_ms']:.0f}ms")
            if report["saved_macro_f1"] is not None:
                print(f"  연구 단계 저장값: macro F1 {report['saved_macro_f1']:.4f} (차이 {report['macro_f1_diff_vs_saved']:+.4f})")
            print("  " + verdict(report))
            print(f"  저장: {report['result_path']}")
    finally:
        close_classifiers({})


if __name__ == "__main__":
    main()
