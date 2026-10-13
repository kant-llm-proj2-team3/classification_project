"""결과 파일 만들기 (baseline/steps/result_files.py).

  main.json  : 시각화와 팀원 비교용 핵심 결과 (모든 run 이 같은 키)
  sub.json   : 그 밖의 부수 정보 (세부 조건, 환경, 상세 지표, 에폭 상세)
  validation_predictions.csv / test_predictions.csv : 문장별 예측 (id, 정답, 예측, 확신도)

학습이 끝날 때마다 run 폴더(날짜 폴더)에 만들고, best 이면 best_model 로 함께 복사된다.
이미 있는 결과에는 메뉴 6번으로 (다시 학습하지 않고) 만들 수 있고, 7번으로 한 곳(artifacts/viz)에 모은다.
main.json / sub.json 은 run 폴더에 있는 기존 파일(run_config.json, training_summary.json,
validation_metrics.json, epochs.jsonl)에서 만들기 때문에, 예전 run 에도 똑같이 적용된다.
"""
import json
import sys
import time
from pathlib import Path

# 설정 파일(baseline_bert_config.py)은 프로젝트 루트(project_2)에 있다. (이 파일에서 두 단계 위)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import baseline_bert_config as cfg  # noqa: E402
import common  # noqa: E402

SCHEMA = "project2-result/v1"      # 파일 형식의 버전. 항목을 바꾸면 올린다.
COST_FIELDS = ["train_seconds", "validation_inference_seconds", "test_inference_seconds",
               "peak_gpu_mb", "num_parameters", "model_size_mb"]


# ═════════════════════════ 1. 조각 함수 ═════════════════════════
def evaluation_block(scores, labels, loss=None):
    """validation 또는 test 한 번의 평가 결과를 '몇 문제 중 몇 개' 형태로 정리한다."""
    correct, total, per_label = common.correct_counts(scores, labels)
    return {
        "n": total, "correct": correct,
        "accuracy": scores["accuracy"], "macro_f1": scores["macro_f1"], "loss": loss,
        "per_label": {label: {"correct": ok, "n": n, "f1": scores["per_class"][label]["f1-score"]}
                      for label, (ok, n) in per_label.items()},
        "labels": labels,
        "confusion_matrix": scores["confusion_matrix"],     # 행 = 실제, 열 = 예측 (labels 순서)
    }


def write_predictions(path, rows, predictions, probabilities, labels):
    """문장별 예측을 저장한다. 팀원과 같은 문장에서 누가 맞고 틀렸는지 비교할 때 쓴다."""
    fields = ["id", "actual", "predicted", "correct", "confidence"] + [f"prob_{label}" for label in labels]
    out = []
    for row, prediction, probability in zip(rows, predictions, probabilities, strict=True):
        actual, predicted = row[cfg.LABEL_COLUMN], labels[prediction]
        item = {"id": row[cfg.ID_COLUMN], "actual": actual, "predicted": predicted,
                "correct": int(actual == predicted), "confidence": round(max(probability), 6)}
        item.update({f"prob_{label}": round(p, 6) for label, p in zip(labels, probability, strict=True)})
        out.append(item)
    common.write_csv(path, fields, out)


def safetensors_param_count(path):
    """safetensors 파일 머리말(JSON)만 읽어 파라미터 수를 센다. (torch 없이 가능)"""
    with open(path, "rb") as stream:
        header = json.loads(stream.read(int.from_bytes(stream.read(8), "little")))
    total = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        count = 1
        for dim in info["shape"]:
            count *= dim
        total += count
    return total


def weights_info(directory):
    """가중치 파일이 있는 폴더에서 (파라미터 수, 파일 크기 MB)를 읽는다. 없으면 (None, None)."""
    directory = Path(directory)
    safetensors, joblib_file = directory / "model.safetensors", directory / "baseline.joblib"
    if safetensors.exists():
        return safetensors_param_count(safetensors), round(safetensors.stat().st_size / 2**20, 1)
    if joblib_file.exists():
        size = round(joblib_file.stat().st_size / 2**20, 2)
        try:
            import joblib
            classifier = joblib.load(joblib_file)[-1]            # 파이프라인의 마지막 = 로지스틱 회귀
            return int(classifier.coef_.size + classifier.intercept_.size), size
        except Exception:
            return None, size
    return None, None


def _read_epochs(run_dir):
    path = Path(run_dir) / "epochs.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ═════════════════════════ 2. main.json / sub.json ═════════════════════════
def build_main_sub(run_dir, weights_dir=None):
    """run 폴더의 기존 파일로 main / sub 내용을 만든다. (예전 run 에도 그대로 적용됨)"""
    run_dir = Path(run_dir)
    config = common.load_json(run_dir / "run_config.json")
    summary = common.load_json(run_dir / "training_summary.json")
    metrics = common.load_json(run_dir / "validation_metrics.json")
    epochs = _read_epochs(run_dir)
    labels = config["labels"]
    key, rows = summary["model_key"], summary["validation_rows"]

    if key == common.BASELINE_KEY:
        tfidf, logistic = config["tfidf"], config["logistic_regression"]
        train = {"analyzer": tfidf["analyzer"], "ngram_range": tfidf["ngram_range"], "min_df": tfidf["min_df"],
                 "sublinear_tf": tfidf["sublinear_tf"], "max_iter": logistic["max_iter"],
                 "class_weight": logistic["class_weight"]}
        curve = None
    else:
        accumulation = config["gradient_accumulation_steps"]
        train = {"learning_rate": config["learning_rate"], "epochs_max": config.get("epochs_max"),
                 "epochs_run": summary.get("epochs_run", len(epochs) or None), "best_epoch": summary["best_epoch"],
                 "batch_size": config["batch_size"], "accumulation_steps": accumulation,
                 "effective_batch": config["batch_size"] * accumulation}
        curve = None if not epochs else {
            "epoch": [e["epoch"] for e in epochs],
            "train_loss": [round(e["train_loss"], 6) for e in epochs],
            "val_loss": [round(e["validation_loss"], 6) for e in epochs],
            "val_macro_f1": [round(e["validation_macro_f1"], 6) for e in epochs],
            "val_correct": [round(e["validation_accuracy"] * rows) for e in epochs]}

    parameters, size_mb = summary.get("num_parameters"), summary.get("model_size_mb")
    if parameters is None:                       # 예전 run 이면 가중치 파일이 있는 경우에만 계산
        parameters, size_mb = weights_info(weights_dir or run_dir)

    main = {
        "schema": SCHEMA,
        "model_name": key, "model_id": summary["model_id"],
        "topic": config["topic"], "dataset": config["dataset"],
        "dataset_id": f"{config['topic']}/{config['dataset']}",
        "run_id": summary["run_name"],
        "train": train,
        "input": {"train_rows": summary["train_rows"], "validation_rows": rows, "test_rows": None},
        "validation": evaluation_block(metrics, labels, summary.get("validation_loss")),
        "test": None,
        "test_meta": None,
        "cost": {"train_seconds": summary["train_seconds"],
                 "validation_inference_seconds": summary.get("validation_inference_seconds"),
                 "test_inference_seconds": None, "peak_gpu_mb": summary.get("peak_gpu_memory_mb"),
                 "num_parameters": parameters, "model_size_mb": size_mb},
        "curve": curve,
        "files": {"sub": "sub.json", "validation_predictions": "validation_predictions.csv",
                  "test_predictions": "test_predictions.csv", "validation_errors": "validation_errors.csv",
                  "log": "train.log"},
    }
    main["files"] = {k: (v if k == "sub" or (run_dir / v).exists() else None) for k, v in main["files"].items()}

    # 이미 test 평가(test_metrics.json)가 있는 폴더면 test 칸도 채운다.
    test_path = run_dir / "test_metrics.json"
    if test_path.exists():
        test_report = common.load_json(test_path)
        if test_report.get("classifier"):
            main["test"] = evaluation_block(test_report["classifier"], labels)
            main["test_meta"] = {"official": test_report.get("official", True), "evaluated_at": None}
            main["input"]["test_rows"] = test_report["classifier"]["n"]

    steps = None
    if epochs:
        steps = epochs[min(summary["best_epoch"], len(epochs)) - 1].get("optimizer_steps")
    sub = {
        "schema": SCHEMA, "run_id": summary["run_name"], "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": {
            "seed": config.get("seed"), "max_length": config.get("max_length"),
            "truncation": config.get("truncation"), "head_tokens": config.get("head_tokens"),
            "weight_decay": config.get("weight_decay"), "warmup_ratio": config.get("warmup_ratio"),
            "gradient_checkpointing": config.get("gradient_checkpointing"), "precision": config.get("precision"),
            "early_stopping_patience": config.get("early_stopping_patience"),
            "target_macro_f1": config.get("target_macro_f1"), "stop_reason": summary.get("stop_reason"),
            "optimizer_steps_at_checkpoint": steps, "selection_split": "validation"},
        "environment": {k: config.get(k) for k in ("device", "gpu", "python", "torch", "transformers", "sklearn")},
        "data": {"labels": labels, "train_csv": config.get("train_csv"),
                 "validation_csv": config.get("validation_csv"), "truncated_rows": config.get("truncated_rows")},
        "validation_detail": {"per_class": metrics["per_class"]},
        "epochs": epochs,
        "training_summary": summary,
        "run_config": config,
    }
    return main, sub


def write_result_files(run_dir, weights_dir=None):
    """main.json 과 sub.json 을 run 폴더에 쓴다. 이미 기록된 test 결과는 지우지 않고 이어받는다."""
    run_dir = Path(run_dir)
    main, sub = build_main_sub(run_dir, weights_dir)
    old_path = run_dir / "main.json"
    if old_path.exists():
        old = common.load_json(old_path)
        if old.get("test") and not main["test"]:
            main["test"] = old["test"]
            main["test_meta"] = old.get("test_meta")
            main["input"]["test_rows"] = old["input"].get("test_rows")
            main["cost"]["test_inference_seconds"] = old["cost"].get("test_inference_seconds")
        old_sub_path = run_dir / "sub.json"
        if old_sub_path.exists():                 # test 에서 잘린 행 수는 test 평가 때만 알 수 있어서 이어받는다
            old_truncated = (common.load_json(old_sub_path).get("data") or {}).get("truncated_rows") or {}
            if old_truncated.get("test") is not None and sub["data"]["truncated_rows"] is not None:
                sub["data"]["truncated_rows"]["test"] = old_truncated["test"]
    common.save_json(run_dir / "main.json", main)
    common.save_json(run_dir / "sub.json", sub)
    return main


def record_test_result(directory, scores, labels, rows, seconds, truncated=None, official=None, with_meta=False):
    """test 평가 결과를 main.json 의 test 칸과 sub.json 에 기록한다."""
    directory = Path(directory)
    if not (directory / "main.json").exists():
        write_result_files(directory)
    main = common.load_json(directory / "main.json")
    main["test"] = evaluation_block(scores, labels)
    main["input"]["test_rows"] = rows
    main["cost"]["test_inference_seconds"] = round(seconds, 3)
    if official is not None or with_meta:   # 공식(validation 으로 정한 모델) / 참고용 표시. with_meta 는 추천 제외 데이터셋의 official=None 기록용
        main["test_meta"] = {"official": official, "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    main["files"]["test_predictions"] = "test_predictions.csv" if (directory / "test_predictions.csv").exists() else None
    common.save_json(directory / "main.json", main)
    if truncated is not None and (directory / "sub.json").exists():
        sub = common.load_json(directory / "sub.json")
        data = sub.setdefault("data", {})
        data["truncated_rows"] = {**(data.get("truncated_rows") or {}), "test": truncated}
        common.save_json(directory / "sub.json", sub)


# ═════════════════════════ 3. 이미 있는 결과에 만들기 (다시 학습하지 않음) ═════════════════════════
def ensure_validation_predictions(model_key, topic, dataset):
    """best_model 의 가중치로 validation 문장별 예측을 다시 만든다. 만들었으면 True."""
    best = common.best_dir(model_key, topic, dataset)
    target = best / "validation_predictions.csv"
    if target.exists() or not (best / "training_summary.json").exists():
        return False
    data = common.load_data(topic, dataset)
    labels, rows = data["labels"], data["validation"]
    label2id = {label: index for index, label in enumerate(labels)}
    actual = [label2id[row[cfg.LABEL_COLUMN]] for row in rows]
    texts = [row[cfg.TEXT_COLUMN] for row in rows]
    if model_key == common.BASELINE_KEY:
        import joblib
        model = joblib.load(best / "baseline.joblib")
        predictions, probabilities = model.predict(texts).tolist(), model.predict_proba(texts).tolist()
    else:
        predictions, probabilities, _, _ = common.predict_with_model_dir(
            best, rows, labels, common.resolve_device())
    now_correct = common.correct_counts(common.classification_scores(actual, predictions, labels), labels)[0]
    recorded_correct = common.correct_counts(common.load_json(best / "validation_metrics.json"), labels)[0]
    if now_correct != recorded_correct:
        print(f"  [주의] {model_key}: 다시 예측한 정답 수({now_correct})가 기록된 값({recorded_correct})과 다릅니다.")
    write_predictions(target, rows, predictions, probabilities, labels)
    write_result_files(best)
    return True


def ensure_best_result_files(model_key, topic, dataset):
    """best_model 에 main.json / sub.json / 문장별 예측이 없으면 채운다.

    새로 학습한 run 이 best 를 못 이기면 이전 형식의 best 가 그대로 남는다. 그 best 자신의 기록(점수·설정)으로
    main/sub 를 정리하고, 문장별 예측은 저장된 가중치로 validation 을 한 번 더 예측해서 만든다. (다른 run 의 파일을 복사하지 않는다)
    실패해도 학습 결과에는 영향이 없도록 예외는 화면에만 알린다.
    """
    best = common.best_dir(model_key, topic, dataset)
    if not best.exists():
        return
    needed = ("run_config.json", "training_summary.json", "validation_metrics.json")
    try:
        if not (best / "main.json").exists() and all((best / name).exists() for name in needed):
            write_result_files(best)
            print(f"  [{model_key}] 이전 형식의 best_model 에 main.json / sub.json 을 만들었습니다.")
        if not (best / "validation_predictions.csv").exists() and ensure_validation_predictions(
                model_key, topic, dataset):
            print(f"  [{model_key}] 이전 형식의 best_model 의 문장별 예측을 만들었습니다.")
    except Exception as error:
        print(f"  [{model_key}] best_model 의 빠진 결과 파일을 채우지 못했습니다: {type(error).__name__}: {error}")
        print("         (메뉴 6번으로 다시 시도할 수 있습니다)")


def rebuild_result_files(topic, dataset, with_predictions=True):
    """main.json / sub.json 이 없는 run 폴더에 만들고, best_model 에는 문장별 예측도 만든다."""
    for key in common.ALL_MODEL_KEYS:
        folders = list(common.complete_run_dirs(key, topic, dataset))
        best = common.best_dir(key, topic, dataset)
        if best.exists():
            folders.append(best)
        made = 0
        for folder in folders:
            needed = ("run_config.json", "training_summary.json", "validation_metrics.json")
            if not (folder / "main.json").exists() and all((folder / name).exists() for name in needed):
                write_result_files(folder)
                made += 1
        note = ""
        if with_predictions and best.exists():
            try:
                note = ", validation 문장별 예측 생성" if ensure_validation_predictions(key, topic, dataset) else ""
            except Exception as error:
                note = f", 문장별 예측 실패({type(error).__name__}: {error})"
        if folders:
            print(f"  {key:<10} main/sub 새로 만든 폴더 {made}개 (전체 {len(folders)}개){note}")


# ═════════════════════════ 4. 시각화용으로 한 곳에 모으기 ═════════════════════════
FLAT_FIELDS = (["model_name", "model_id", "topic", "dataset", "run_id", "learning_rate", "epochs_run", "best_epoch",
                "validation_n", "validation_correct", "validation_accuracy", "validation_macro_f1", "validation_loss",
                "test_n", "test_correct", "test_accuracy", "test_macro_f1", "test_official"] + COST_FIELDS)


def flatten_main(main):
    validation, test, train = main["validation"], main.get("test") or {}, main["train"]
    row = {"model_name": main["model_name"], "model_id": main["model_id"], "topic": main["topic"],
           "dataset": main["dataset"], "run_id": main["run_id"], "learning_rate": train.get("learning_rate"),
           "epochs_run": train.get("epochs_run"), "best_epoch": train.get("best_epoch"),
           "validation_n": validation["n"], "validation_correct": validation["correct"],
           "validation_accuracy": validation["accuracy"], "validation_macro_f1": validation["macro_f1"],
           "validation_loss": validation.get("loss"), "test_n": test.get("n"), "test_correct": test.get("correct"),
           "test_accuracy": test.get("accuracy"), "test_macro_f1": test.get("macro_f1"),
           "test_official": (main.get("test_meta") or {}).get("official")}
    row.update({field: main["cost"].get(field) for field in COST_FIELDS})
    return row


def collect_all_main():
    """모든 데이터셋 x 모델의 best_model/main.json 을 artifacts/viz 에 모은다. (그래프용 표)"""
    results, missing = [], []
    for topic, dataset in common.all_pairs():
        for key in common.ALL_MODEL_KEYS:
            path = common.best_dir(key, topic, dataset) / "main.json"
            if path.exists():
                results.append(common.load_json(path))
            elif common.best_dir(key, topic, dataset).exists():
                missing.append(f"{topic}/{dataset}/{key}")
    out = cfg.ARTIFACTS_ROOT / "viz"
    common.save_json(out / "all_main.json", {"schema": SCHEMA, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                             "count": len(results), "results": results})
    common.write_csv(out / "all_main.csv", FLAT_FIELDS, [flatten_main(m) for m in results])

    epoch_rows, label_rows, confusion_rows = [], [], []
    for main in results:
        ident = {"model_name": main["model_name"], "topic": main["topic"], "dataset": main["dataset"]}
        curve = main.get("curve")
        if curve:
            n = main["validation"]["n"]
            for i, epoch in enumerate(curve["epoch"]):
                epoch_rows.append({**ident, "epoch": epoch, "train_loss": curve["train_loss"][i],
                                   "val_loss": curve["val_loss"][i], "val_macro_f1": curve["val_macro_f1"][i],
                                   "val_correct": curve["val_correct"][i], "val_n": n})
        for split in ("validation", "test"):
            block = main.get(split)
            if not block:
                continue
            for label, info in block["per_label"].items():
                label_rows.append({**ident, "split": split, "label": label, "correct": info["correct"],
                                   "n": info["n"], "f1": info["f1"]})
            for i, actual in enumerate(block["labels"]):
                for j, predicted in enumerate(block["labels"]):
                    confusion_rows.append({**ident, "split": split, "actual": actual, "predicted": predicted,
                                           "count": block["confusion_matrix"][i][j]})
    ident_fields = ["model_name", "topic", "dataset"]
    common.write_csv(out / "all_epochs.csv", ident_fields + ["epoch", "train_loss", "val_loss", "val_macro_f1",
                                                           "val_correct", "val_n"], epoch_rows)
    common.write_csv(out / "all_per_label.csv", ident_fields + ["split", "label", "correct", "n", "f1"], label_rows)
    common.write_csv(out / "all_confusion.csv", ident_fields + ["split", "actual", "predicted", "count"],
                     confusion_rows)
    print(f"  모은 결과: {len(results)}개 (best_model 의 main.json)")
    if missing:
        print(f"  [주의] main.json 이 없는 best_model {len(missing)}개는 빠졌습니다 -> 메뉴 6번으로 먼저 만드세요.")
    print(f"  저장: {out}  (all_main.json, all_main.csv, all_epochs.csv, all_per_label.csv, all_confusion.csv)")
    return results
