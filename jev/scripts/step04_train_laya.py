"""
laya(multilingual)의 zero-shot 평가와 파인튜닝을 baseline/BERT 계열과 같은 결과 형식으로 실행한다.

데이터셋마다 두 모델을 기록한다.
  laya_zeroshot : 학습 없이 원본 모델로 validation을 평가 (파인튜닝 전에 먼저 실행)
  laya          : 파인튜닝. 에폭마다 validation을 평가해 가장 좋은 에폭의 가중치만 남긴다.

결과는 동료 코드(baseline/steps)와 같은 구조로 저장한다.
  artifacts/<topic>/<dataset>/<laya_zeroshot | laya>/
      best_model/        validation이 가장 좋았던 결과 (laya는 가중치 포함)
      20261013_120000/   실행 1회 = 폴더 1개 (점수, 로그, main.json, sub.json)
에폭, 배치, 조기 종료, 데이터셋 ON/OFF는 baseline_bert_config.py, laya 전용 값은 settings.LAYA_CONFIG에서 바꾼다.

루트 디렉토리 기준 실행 코드 (Windows는 PYTHONUTF8=1 권장)
uv run python ./jev/scripts/step04_train_laya.py                                   # ON인 데이터셋 전부
uv run python ./jev/scripts/step04_train_laya.py --topic inquiries --dataset basic
uv run python ./jev/scripts/step04_train_laya.py --zero-shot-only                  # 파인튜닝 없이 zero-shot만
uv run python ./jev/scripts/step04_train_laya.py --test                            # best_model로 test 평가 (모델마다 1회)
"""

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from laya.common import state_room
from laya.train import (TrainConfig, calibration_records, items_from_rows, load_checkpoint,
                        optimizer_updates, save_checkpoint, train_model)

from settings import LAYA_CONFIG, LAYA_MODEL_ID, MODEL_PATH, ROOT
from step01_read_data import read_rows
from step02_check_data import check_splits, group_rule_for

# 동료 코드의 설정, 지표 계산, best_model 교체, 결과 파일 함수를 그대로 사용해 형식을 맞춘다.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baseline" / "steps"))
import baseline_bert_config as cfg  # noqa: E402
import common  # noqa: E402
import result_files  # noqa: E402

QID = "label"
ZERO_SHOT_KEY = "laya_zeroshot"
FINETUNE_KEY = "laya"
LAYA_KEYS = (ZERO_SHOT_KEY, FINETUNE_KEY)


class EarlyStop(Exception):
    """에폭 콜백에서 학습을 멈추기 위한 신호. (laya.train.train_model에는 중단 기능이 없다)"""


# CSV 행(id, text, label, group_id)과 laya용 JSONL 행을 같은 순서로 읽는다.
# CSV 행은 동료 코드의 지표/오답/예측 파일 함수에, JSONL 행은 laya 입력에 쓴다.
def load_split_pairs(topic: str, dataset: str, with_test: bool = False) -> dict:
    data = common.load_data(topic, dataset, with_test=with_test)
    folder = Path(cfg.DATA_ROOT) / topic / cfg.DATASET_FOLDERS[dataset]
    names = ("train", "validation", "test") if with_test else ("train", "validation")
    jsonl = {name: read_rows(folder / f"{name}.jsonl") for name in names}
    check_splits({name: jsonl[name] for name in ("train", "validation")}, data["labels"],
                 group_rule_for(folder.name))

    labels = data["labels"]
    for name in names:
        csv_ids = [row[cfg.ID_COLUMN] for row in data[name]]
        jsonl_ids = [common_tag(row, "id") for row in jsonl[name]]
        if csv_ids != jsonl_ids:
            raise ValueError(f"{topic}/{dataset}/{name}: CSV와 JSONL의 행 순서가 다릅니다. "
                             "jev/scripts/util/data_convert.py로 다시 변환하세요.")
        choices = list(jsonl[name][0]["questions"][QID]["criteria"])
        if choices != labels:
            raise ValueError(f"{name}: 선택지 순서 {choices}가 labels.json {labels}와 다릅니다.")
    return {"labels": labels, "csv": data, "jsonl": jsonl}


def common_tag(row: dict, key: str) -> str:
    return dict(tag.split(":", 1) for tag in row["tags"])[key]


def resolve_lengths(laya_cfg: dict) -> tuple[int, int]:
    max_len = LAYA_CONFIG["max_len"] or laya_cfg.get("max_len", 512)
    head_max_len = LAYA_CONFIG["head_max_len"] or laya_cfg.get("head_max_len", 192)
    return max_len, head_max_len


# JSONL 행을 laya 학습/평가 단위(item)로 바꾸고, max_len 때문에 문장이 잘린 행 수를 센다.
def build_items(tok, rows: list[dict], max_len: int, head_max_len: int) -> tuple[list[dict], int]:
    items, skipped = items_from_rows(tok, rows, max_len, head_max_len)
    if skipped or len(items) != len(rows):
        raise ValueError(f"laya 입력으로 바꾸지 못한 행이 있습니다: {skipped}")
    truncated = sum(1 for item in items
                    if len(item["state_ids"]) > state_room(tok, item["q"], max_len, head_max_len))
    return items, truncated


# 모델로 예측한다. 반환: (예측 번호, 라벨별 확률, 평균 loss, 추론 시간(초))
# 확률과 loss는 온도 보정 없이(T=1) 계산한다. BERT 계열과 같은 기준이다.
def predict_items(model, tok, items: list[dict], device, max_len: int, head_max_len: int):
    started = time.perf_counter()
    records = calibration_records(model, tok, items, device, max_len, head_max_len,
                                  batch_size=LAYA_CONFIG["eval_batch_size"])
    seconds = time.perf_counter() - started
    logits = torch.tensor(np.stack([record[1] for record in records]), dtype=torch.float32)
    targets = torch.tensor([int(np.argmax(record[2])) for record in records])
    probabilities = torch.softmax(logits, dim=-1)
    loss = torch.nn.functional.cross_entropy(logits, targets).item()
    return logits.argmax(-1).tolist(), probabilities.tolist(), loss, seconds


def label_ids(rows: list[dict], labels: list[str]) -> list[int]:
    label2id = {label: index for index, label in enumerate(labels)}
    return [label2id[row[cfg.LABEL_COLUMN]] for row in rows]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()


def environment(device) -> dict:
    use_cuda = device.type == "cuda"
    return {"device": str(device), "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__}


# main.json / sub.json을 동료 코드로 만든 뒤 laya에 맞게 train 칸을 보충한다.
def write_main(run_dir: Path, weights_dir: Path | None, train_extra: dict, zero_shot: bool = False) -> None:
    main, sub = result_files.build_main_sub(run_dir, weights_dir)
    if zero_shot:
        main["train"] = train_extra
        main["curve"] = None
    else:
        main["train"].update(train_extra)
    common.save_json(run_dir / "main.json", main)
    common.save_json(run_dir / "sub.json", sub)


def save_validation_files(run_dir: Path, rows: list[dict], labels: list[str], scores: dict,
                          predictions: list[int], probabilities: list[list[float]]) -> None:
    common.save_json(run_dir / "validation_metrics.json", scores)
    common.save_errors(run_dir / "validation_errors.csv", rows, predictions, labels)
    result_files.write_predictions(run_dir / "validation_predictions.csv", rows, predictions,
                                   probabilities, labels)


# ═════════════════════════ 1. zero-shot (학습 없음) ═════════════════════════
def run_zero_shot(topic: str, dataset: str, data: dict, device) -> dict:
    run_dir = common.begin_run(ZERO_SHOT_KEY, topic, dataset)
    try:
        return _zero_shot(run_dir, topic, dataset, data, device)
    except BaseException:
        common.safe_delete(run_dir)
        raise


def _zero_shot(run_dir: Path, topic: str, dataset: str, data: dict, device) -> dict:
    labels, rows = data["labels"], data["csv"]["validation"]
    set_seed(cfg.SEED)
    model, tok, laya_cfg = load_checkpoint(str(MODEL_PATH))
    model.to(device)
    max_len, head_max_len = resolve_lengths(laya_cfg)
    items, truncated = build_items(tok, data["jsonl"]["validation"], max_len, head_max_len)

    predictions, probabilities, loss, seconds = predict_items(model, tok, items, device,
                                                              max_len, head_max_len)
    scores = common.classification_scores(label_ids(rows, labels), predictions, labels)
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    num_parameters, model_size_mb = result_files.weights_info(MODEL_PATH)

    common.save_json(run_dir / "run_config.json", {
        "topic": topic, "dataset": dataset, "model_key": ZERO_SHOT_KEY, "model_id": LAYA_MODEL_ID,
        "run_name": run_dir.name, "learning_rate": None, "epochs_max": 0,
        "batch_size": LAYA_CONFIG["eval_batch_size"], "gradient_accumulation_steps": 1,
        "max_length": max_len, "head_max_len": head_max_len, "truncation": "right",
        "truncated_rows": {"validation": truncated}, "seed": cfg.SEED, "labels": labels,
        "validation_rows": len(rows), "weights": str(MODEL_PATH),
        **environment(device)})
    summary = common.make_summary(
        ZERO_SHOT_KEY, LAYA_MODEL_ID, run_dir.name, None, scores, loss, 0, 0.0, seconds, peak_mb,
        0, len(rows), extra={"epochs_run": 0, "epochs_max": 0, "stop_reason": "zero-shot (학습 없음)",
                             "num_parameters": num_parameters, "model_size_mb": model_size_mb})
    common.save_json(run_dir / "training_summary.json", summary)
    save_validation_files(run_dir, rows, labels, scores, predictions, probabilities)

    log = (f"  [{ZERO_SHOT_KEY}/{run_dir.name}] {topic}/{dataset} validation {len(rows)}행 | "
           f"잘린 행 {truncated} | val_loss={loss:.4f} val_macro_f1={scores['macro_f1']:.4f}\n"
           f"  [{ZERO_SHOT_KEY}/{run_dir.name}] validation 정답: {common.format_correct(scores, labels)}")
    print(log)
    (run_dir / "train.log").write_text(log + "\n", encoding="utf-8")
    write_main(run_dir, None, {"fine_tuned": False, "max_len": max_len}, zero_shot=True)

    common.finalize_run(ZERO_SHOT_KEY, topic, dataset, run_dir, summary, run_dir / common.WEIGHTS_TMP)
    common.mark_complete(run_dir)
    del model
    common.free_memory()
    return summary


# ═════════════════════════ 2. 파인튜닝 (에폭마다 validation) ═════════════════════════
def run_finetune(topic: str, dataset: str, data: dict, device, epochs: int | None = None) -> dict:
    run_dir = common.begin_run(FINETUNE_KEY, topic, dataset)
    try:
        return _finetune(run_dir, topic, dataset, data, device, epochs)
    except BaseException:
        common.safe_delete(run_dir)
        raise


def _finetune(run_dir: Path, topic: str, dataset: str, data: dict, device, epochs: int | None) -> dict:
    tc = cfg.TRAINING_CONFIG
    labels, rows = data["labels"], data["csv"]["validation"]
    run_name = run_dir.name
    weights_dir = run_dir / common.WEIGHTS_TMP
    lines = []

    def say(message):
        print(message)
        lines.append(message)

    set_seed(tc["seed"])
    model, tok, laya_cfg = load_checkpoint(str(MODEL_PATH))
    model.to(device)
    max_len, head_max_len = resolve_lengths(laya_cfg)
    train_items, train_cut = build_items(tok, data["jsonl"]["train"], max_len, head_max_len)
    validation_items, validation_cut = build_items(tok, data["jsonl"]["validation"], max_len, head_max_len)
    actual = label_ids(rows, labels)

    config = TrainConfig(
        epochs=epochs or tc["epochs"], micro_batch=tc["batch_size"],
        grad_accum=tc["gradient_accumulation_steps"], encoder_lr=LAYA_CONFIG["encoder_lr"],
        head_lr=LAYA_CONFIG["head_lr"], weight_decay=tc["weight_decay"], grad_clip=tc["max_grad_norm"],
        loss=LAYA_CONFIG["loss"], seed=tc["seed"], max_len=max_len, head_max_len=head_max_len,
        shuffle_options=("choice",) if LAYA_CONFIG["shuffle_options"] else (),
        gradient_checkpointing=bool(device.type == "cuda" and tc["gradient_checkpointing"]),
        log_every=0)
    config.validate()
    updates_per_epoch = optimizer_updates(len(train_items), config.micro_batch, config.grad_accum, 1)
    patience, target = tc["early_stopping_patience"], tc["target_macro_f1"]
    score_stop = common.score_stop_settings()

    say(f"  [{FINETUNE_KEY}/{run_name}] 설정: {topic}/{dataset} | train {len(train_items)}행 / "
        f"validation {len(validation_items)}행 | LR encoder {config.encoder_lr:g} head {config.head_lr:g} | "
        f"loss {config.loss} | 에폭 최대 {config.epochs} | 배치 {config.micro_batch}x{config.grad_accum} "
        f"(실질 {config.micro_batch * config.grad_accum}) | {device}")
    say(f"  [{FINETUNE_KEY}/{run_name}] 중단 조건: "
        f"{f'Macro-F1 {patience}에폭 연속 개선 없음' if patience else '조기 종료 끔'}"
        f"{f' / Macro-F1 {target} 이상' if target is not None else ''} | "
        f"점수 기준 종료: {common.describe_score_stop(score_stop)}")
    say(f"  [{FINETUNE_KEY}/{run_name}] 입력: max_len {max_len} (head {head_max_len}) | "
        f"잘린 행: train {train_cut}/{len(train_items)}, validation {validation_cut}/{len(validation_items)}")

    common.save_json(run_dir / "run_config.json", {
        "topic": topic, "dataset": dataset, "model_key": FINETUNE_KEY, "model_id": LAYA_MODEL_ID,
        "run_name": run_name, "learning_rate": config.encoder_lr, "head_learning_rate": config.head_lr,
        "loss": config.loss, "shuffle_options": LAYA_CONFIG["shuffle_options"],
        "epochs_max": config.epochs, "early_stopping_patience": patience, "target_macro_f1": target,
        "score_stop": score_stop, "batch_size": config.micro_batch,
        "gradient_accumulation_steps": config.grad_accum, "max_length": max_len,
        "head_max_len": head_max_len, "truncation": "right",
        "truncated_rows": {"train": train_cut, "validation": validation_cut},
        "weight_decay": config.weight_decay, "scheduler": "cosine (laya.train)", "seed": config.seed,
        "labels": labels, "train_rows": len(train_items), "validation_rows": len(validation_items),
        "train_csv": common.relative_paths(common.dataset_paths(topic, dataset)["train"]),
        "validation_csv": common.relative_paths(common.dataset_paths(topic, dataset)["validation"]),
        "precision": "float16 (amp)" if device.type == "cuda" else "float32",
        "gradient_checkpointing": config.gradient_checkpointing, **environment(device)})

    # 같은 질문 정의를 체크포인트와 함께 저장해 두면 서버에서 그대로 쓸 수 있다.
    questions = data["jsonl"]["train"][0]["questions"]
    out_cfg = dict(laya_cfg, max_len=max_len, head_max_len=head_max_len, fine_tuned=True)
    tracker = common.EpochTracker(patience, target, score_stop)
    state = {"best": None}
    started = time.perf_counter()

    def on_epoch_end(epoch_index, train_loss):
        epoch = epoch_index + 1
        # 학습 중 캐시된 GPU 메모리를 비우고 평가한다. (긴 문장 평가에서 8GB를 넘지 않도록)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        predictions, probabilities, val_loss, val_seconds = predict_items(
            model, tok, validation_items, device, max_len, head_max_len)
        model.train()   # calibration_records가 eval 모드로 바꾸므로 다음 에폭을 위해 되돌린다.
        memory_note = ""
        if device.type == "cuda":
            memory_note = (f"  GPU 사용 {torch.cuda.memory_allocated() / 2**20:.0f}MB / "
                           f"확보 {torch.cuda.memory_reserved() / 2**20:.0f}MB")
        scores = common.classification_scores(actual, predictions, labels)
        correct, total, _ = common.correct_counts(scores, labels)
        is_best = tracker.update(scores["macro_f1"], val_loss, correct, total)
        elapsed = time.perf_counter() - started
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                  "validation_accuracy": scores["accuracy"], "validation_macro_f1": scores["macro_f1"],
                  "optimizer_steps": updates_per_epoch * epoch, "elapsed_seconds": elapsed}
        with (run_dir / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        say(f"  [{FINETUNE_KEY}/{run_name}] epoch {epoch}/{config.epochs}  "
            f"train_loss({config.loss})={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_macro_f1={scores['macro_f1']:.4f}  정답 {correct}/{total} ({correct / total:.1%})"
            f"{'  <- best' if is_best else ''}  ({elapsed:.0f}s){memory_note}")
        if is_best:
            save_checkpoint(model, tok, out_cfg, str(weights_dir))
            common.save_json(weights_dir / "questions.json", questions)
            state["best"] = {"epoch": epoch, "scores": scores, "loss": val_loss, "predictions": predictions,
                             "probs": probabilities, "val_seconds": val_seconds}
        if tracker.should_stop:
            raise EarlyStop(tracker.stop_reason)

    epochs_run = config.epochs
    try:
        train_model(model, tok, train_items, config, device, max_len, head_max_len,
                    on_epoch_end=on_epoch_end)
    except EarlyStop as stop:
        epochs_run = tracker.epoch
        say(f"  [{FINETUNE_KEY}/{run_name}] 조기 종료: epoch {epochs_run}/{config.epochs} - {stop}")

    best = state["best"]
    train_seconds = time.perf_counter() - started
    stop_reason = tracker.stop_reason or f"최대 에폭({config.epochs}) 도달"
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    num_parameters, model_size_mb = result_files.weights_info(weights_dir)
    summary = common.make_summary(
        FINETUNE_KEY, LAYA_MODEL_ID, run_name, config.encoder_lr, best["scores"], best["loss"],
        best["epoch"], train_seconds, best["val_seconds"], peak_mb, len(train_items), len(validation_items),
        extra={"epochs_run": epochs_run, "epochs_max": config.epochs, "stop_reason": stop_reason,
               "num_parameters": num_parameters, "model_size_mb": model_size_mb})
    common.save_json(run_dir / "training_summary.json", summary)
    save_validation_files(run_dir, rows, labels, best["scores"], best["predictions"], best["probs"])

    gpu_note = f", GPU 최대 {peak_mb:.0f}MB" if peak_mb is not None else ""
    say(f"  [{FINETUNE_KEY}/{run_name}] 완료: best epoch {best['epoch']} (총 {epochs_run}에폭 학습), "
        f"val_macro_f1={best['scores']['macro_f1']:.4f}, {train_seconds:.0f}초{gpu_note}")
    say(f"  [{FINETUNE_KEY}/{run_name}] validation 정답(best epoch {best['epoch']}): "
        f"{common.format_correct(best['scores'], labels)}")
    (run_dir / "train.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_main(run_dir, weights_dir, {"head_learning_rate": config.head_lr, "loss": config.loss,
                                      "max_len": max_len, "fine_tuned": True})

    common.finalize_run(FINETUNE_KEY, topic, dataset, run_dir, summary, weights_dir)
    common.mark_complete(run_dir)
    del model
    common.free_memory()
    return summary


# ═════════════════════════ 3. test 평가 (best_model, 모델마다 1회) ═════════════════════════
def evaluate_test(topic: str, dataset: str, device) -> None:
    data = load_split_pairs(topic, dataset, with_test=True)
    labels, rows = data["labels"], data["csv"]["test"]
    actual = label_ids(rows, labels)
    # laya는 동료 코드의 모델 추천 대상이 아니므로 공식(official) 점수가 아니라 참고용으로 기록한다.
    official = None if common.is_recommend_excluded(dataset) else False
    for key in LAYA_KEYS:
        best = common.best_dir(key, topic, dataset)
        best_info = common.read_best_info(key, topic, dataset)
        if best_info is None:
            print(f"  [{key}] best_model이 없어 건너뜁니다.")
            continue
        if (best / "test_metrics.json").exists():
            print(f"  [{key}] 이미 test 평가를 했습니다. 덮어쓰지 않고 건너뜁니다.")
            continue
        weights = MODEL_PATH if key == ZERO_SHOT_KEY else best
        model, tok, laya_cfg = load_checkpoint(str(weights))
        model.to(device)
        run_config = common.load_json(best / "run_config.json")
        max_len, head_max_len = run_config["max_length"], run_config["head_max_len"]
        items, truncated = build_items(tok, data["jsonl"]["test"], max_len, head_max_len)
        predictions, probabilities, _, seconds = predict_items(model, tok, items, device,
                                                               max_len, head_max_len)
        scores = common.classification_scores(actual, predictions, labels)
        common.save_json(best / "test_metrics.json", {
            "topic": topic, "dataset": dataset, "split": "test", "selected_model": None,
            "evaluated_model": key, "official": official, "selected_run": best_info["best"]["run_name"],
            "model_id": LAYA_MODEL_ID, "classifier": scores, "baseline": None})
        common.save_errors(best / "test_errors.csv", rows, predictions, labels)
        result_files.write_predictions(best / "test_predictions.csv", rows, predictions, probabilities, labels)
        result_files.record_test_result(best, scores, labels, len(rows), seconds, truncated,
                                        official=official, with_meta=True)
        print(f"  [{key}] test macro F1 {scores['macro_f1']:.4f} (accuracy {scores['accuracy']:.4f}) [참고]")
        print(f"  [{key}] test 정답: {common.format_correct(scores, labels)}")
        del model
        common.free_memory()


def main():
    ap = argparse.ArgumentParser(description="laya zero-shot 평가 + 파인튜닝")
    ap.add_argument("--topic", choices=cfg.TOPICS, help="생략하면 ON인 주제 전부")
    ap.add_argument("--dataset", choices=cfg.DATASET_NAMES, help="생략하면 ON인 데이터셋 전부")
    ap.add_argument("--zero-shot-only", action="store_true", help="파인튜닝 없이 zero-shot 평가만")
    ap.add_argument("--epochs", type=int, default=None, help="최대 에폭 임시 변경 (기본: config 값)")
    ap.add_argument("--test", action="store_true", help="학습 대신 best_model로 test 평가")
    a = ap.parse_args()

    pairs = common.ready_pairs(a.topic, a.dataset)
    if not pairs:
        print("실행할 데이터셋이 없습니다. baseline_bert_config.py의 DATASET_SWITCH를 확인하세요.")
        return
    device = common.resolve_device()
    with common.session_log("test_laya" if a.test else "train_laya"):
        print(f"laya: {LAYA_MODEL_ID} | {MODEL_PATH} | {device}")
        print(f"laya 설정: {LAYA_CONFIG}")
        failures = []
        for topic, dataset in pairs:
            print(f"\n=== {topic}/{dataset} · laya ===")
            try:
                if a.test:
                    evaluate_test(topic, dataset, device)
                    continue
                data = load_split_pairs(topic, dataset)
                # laya는 학습 없이도 쓸 수 있으므로 파인튜닝 전에 zero-shot 지표를 먼저 기록한다.
                run_zero_shot(topic, dataset, data, device)
                if not a.zero_shot_only:
                    run_finetune(topic, dataset, data, device, a.epochs)
            except Exception as error:
                print(f"  [laya] {topic}/{dataset} 실패: {type(error).__name__}: {error}")
                failures.append(f"{topic}/{dataset}")
            common.free_memory()
        if failures:
            print(f"\n실패한 데이터셋: {failures}")
            sys.exit(1)


if __name__ == "__main__":
    main()
