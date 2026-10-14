"""
CLM(Contrastive Language Model)의 zero-shot 평가와 판단 헤드 파인튜닝을 baseline/BERT/laya와 같은 결과 형식으로 실행한다.

인코더는 Qwen3-8B Q4_K_M(4비트 고정, 학습하지 않음)이고 llama-server로 띄운다. 코드가 서버를 찾아서 쓰거나 직접 띄운다. (util/clm_encoder.py)
학습하는 것은 약 2천만 파라미터의 판단 헤드(state/action)뿐이라서, best_model에는 헤드(head.safetensors)와
그 헤드가 어떤 인코더로 학습됐는지 적은 기록(encoder.json)만 저장한다. 인코더 가중치는 프로젝트에 저장하지 않는다.

데이터셋마다 두 모델을 기록한다.
  clm_zeroshot : 공식 헤드(CLM_v0.1-8B.pt)를 학습 없이 validation에 평가 (파인튜닝 전에 먼저 실행)
  clm          : 같은 헤드에서 시작해 파인튜닝. 에폭마다 validation을 평가해 가장 좋은 에폭의 헤드만 남긴다.

결과는 step04(laya)와 같은 구조로 저장한다.
  artifacts/<topic>/<dataset>/<clm_zeroshot | clm>/
      best_model/        validation이 가장 좋았던 결과 (clm은 헤드 포함)
      20261014_120000/   실행 1회 = 폴더 1개 (점수, 로그, main.json, sub.json)
에폭, 조기 종료, 데이터셋 ON/OFF는 baseline_bert_config.py, CLM 전용 값은 clm_settings.py에서 바꾼다.

루트 디렉토리 기준 실행 코드 (Windows는 PYTHONUTF8=1 권장)
uv run python ./jev/scripts/step05_train_clm.py                                   # ON인 데이터셋 전부
uv run python ./jev/scripts/step05_train_clm.py --topic documents --dataset basic
uv run python ./jev/scripts/step05_train_clm.py --zero-shot-only                  # 파인튜닝 없이 zero-shot만
uv run python ./jev/scripts/step05_train_clm.py --finetune-only                   # zero-shot은 건너뛰고 파인튜닝만
uv run python ./jev/scripts/step05_train_clm.py --test                            # best_model로 test 평가 (모델마다 1회)
"""

import argparse
import functools
import hashlib
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import transformers

import clm_settings as S
from util.clm_encoder import ensure_encoder, load_clm
from util.clm_head import (ENCODER_FILE, QUESTIONS_FILE, ClmClassifier, HeadModel, base_head_path,
                           infonce_loss)
from util.data_convert import QID, load_questions

# 동료 코드의 설정, 지표 계산, best_model 교체, 결과 파일 함수를 그대로 사용해 형식을 맞춘다.
sys.path.insert(0, str(S.ROOT))
sys.path.insert(0, str(S.ROOT / "baseline" / "steps"))
import baseline_bert_config as cfg  # noqa: E402
import common  # noqa: E402
import result_files  # noqa: E402

ZERO_SHOT_KEY = "clm_zeroshot"
FINETUNE_KEY = "clm"
CLM_KEYS = (ZERO_SHOT_KEY, FINETUNE_KEY)


@dataclass
class Example:
    state_text: str          # 문장 + 질문 (헤드가 보는 state 텍스트)
    candidates: list[str]    # 선택지 설명 (labels.json 순서)
    label: int               # 정답 선택지 번호 = labels.json의 라벨 번호


# ═════════════════════════ 0. 데이터 · 공통 함수 ═════════════════════════
# CSV 행은 동료 코드의 지표/오답/예측 파일 함수에, 같은 행에서 만든 Example은 CLM 입력에 쓴다.
# 질문과 선택지 설명은 labels.json에서 data_convert.py(laya JSONL 변환)와 같은 방식으로 만든다. (JSONL 파일이 없어도 된다)
def load_split_rows(topic: str, dataset: str, with_test: bool = False) -> dict:
    data = common.load_data(topic, dataset, with_test=with_test)
    labels = data["labels"]
    questions, _ = load_questions(common.dataset_paths(topic, dataset)["labels"], topic)
    build_pairs = load_clm().schema.build_pairs
    examples = {}
    for name in ("train", "validation", "test"):
        if name not in data:
            continue
        items = []
        for row in data[name]:
            state_text, keys, candidates = build_pairs(row[cfg.TEXT_COLUMN], questions)[QID]
            if keys != labels:
                raise ValueError(f"{name}: 선택지 순서 {keys}가 labels.json {labels}와 다릅니다.")
            items.append(Example(state_text, candidates, keys.index(row[cfg.LABEL_COLUMN])))
        examples[name] = items
    return {"labels": labels, "csv": data, "questions": questions, "examples": examples}


def label_ids(rows: list[dict], labels: list[str]) -> list[int]:
    label2id = {label: index for index, label in enumerate(labels)}
    return [label2id[row[cfg.LABEL_COLUMN]] for row in rows]


def count_truncated(enc, examples: list[Example]) -> int:
    """인코더 한도(max_len) 때문에 앞부분이 잘리는 행 수. 질문은 끝에 있어서 남는다."""
    return sum(1 for length in enc.token_lengths([e.state_text for e in examples]) if length > enc.token_cap)


def shared_candidates(examples: list[Example]) -> list[str]:
    """모든 행이 같은 선택지를 가진다. (질문이 하나뿐이라서)"""
    first = examples[0].candidates
    if any(e.candidates != first for e in examples):
        raise ValueError("행마다 선택지가 다릅니다. CLM 학습 코드는 모든 행의 선택지가 같다고 가정합니다.")
    return first


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def environment(device, enc) -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"device": f"{device} (head) + llama.cpp (encoder)", "gpu": gpu, "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "llama_cpp_build": enc.identity.get("build"), "clm_commit": enc.clm_commit}


@functools.lru_cache(maxsize=None)
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cost_fields(enc, head_parameters: int, head_file: Path) -> dict:
    """비용 칸. 인코더(GGUF) 없이는 쓸 수 없으므로 num_parameters와 model_size_mb는 인코더와 헤드의 합이다.
    헤드만의 값은 head_*, 인코더만의 값은 encoder_*에 따로 남긴다. 서버가 파라미터 수를 알려 주지 않으면 합은 None이다."""
    encoder_parameters = enc.identity.get("n_params")
    encoder_mb = enc.gguf_size / 2**20 if enc.gguf_size else None
    head_mb = Path(head_file).stat().st_size / 2**20
    return {"num_parameters": head_parameters + int(encoder_parameters) if encoder_parameters else None,
            "model_size_mb": round(encoder_mb + head_mb, 1) if encoder_mb else None,
            "head_parameters": head_parameters, "head_size_mb": round(head_mb, 1),
            "encoder_parameters": int(encoder_parameters) if encoder_parameters else None,
            "encoder_size_mb": round(encoder_mb, 1) if encoder_mb else None}


def encoder_record(enc, extra: dict | None = None) -> dict:
    """헤드와 함께 저장하는 인코더 기록(encoder.json). 헤드는 이 인코더와 짝이 맞을 때만 쓸 수 있다."""
    base = base_head_path()
    return {**enc.describe(), "base_head": {"repo": S.BASE_HEAD["repo"], "file": S.BASE_HEAD["file"],
                                            "sha256": file_sha256(base)}, **(extra or {})}


# main.json / sub.json을 동료 코드로 만든 뒤 CLM에 맞게 train 칸을 보충한다.
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


def softmax_loss(logits: torch.Tensor, actual: list[int]) -> float:
    return torch.nn.functional.cross_entropy(logits, torch.tensor(actual)).item()


# ═════════════════════════ 1. zero-shot (학습 없음) ═════════════════════════
def run_zero_shot(topic: str, dataset: str, data: dict, enc, device, seed: int) -> dict:
    run_dir = common.begin_run(ZERO_SHOT_KEY, topic, dataset)
    try:
        return _zero_shot(run_dir, topic, dataset, data, enc, device, seed)
    except BaseException:
        common.safe_delete(run_dir)
        raise


def _zero_shot(run_dir: Path, topic: str, dataset: str, data: dict, enc, device, seed: int) -> dict:
    labels, rows = data["labels"], data["csv"]["validation"]
    set_seed(seed)
    classifier = ClmClassifier.from_base_head(data["questions"], enc, device)
    truncated = count_truncated(enc, data["examples"]["validation"])

    logits, seconds = classifier.predict_logits([row[cfg.TEXT_COLUMN] for row in rows])
    predictions, probabilities = logits.argmax(-1).tolist(), torch.softmax(logits, dim=-1).tolist()
    actual = label_ids(rows, labels)
    loss = softmax_loss(logits, actual)
    scores = common.classification_scores(actual, predictions, labels)
    peak_mb, peak_scope = enc.gpu_memory()
    base_path = base_head_path()
    cost = cost_fields(enc, classifier.head.num_parameters(), base_path)

    common.save_json(run_dir / "run_config.json", {
        "topic": topic, "dataset": dataset, "model_key": ZERO_SHOT_KEY, "model_id": S.CLM_MODEL_ID,
        "run_name": run_dir.name, "learning_rate": None, "epochs_max": 0,
        "batch_size": S.ENCODER["request_chunk"], "gradient_accumulation_steps": 1,
        "max_length": S.MAX_LEN, "truncation": "tail", "truncated_rows": {"validation": truncated},
        "seed": seed, "labels": labels, "validation_rows": len(rows), "weights": str(base_path),
        "encoder": enc.describe(), "gpu_memory_scope": peak_scope, **environment(device, enc)})
    summary = common.make_summary(
        ZERO_SHOT_KEY, S.CLM_MODEL_ID, run_dir.name, None, scores, loss, 0, 0.0, seconds, peak_mb,
        0, len(rows), extra={"epochs_run": 0, "epochs_max": 0, "stop_reason": "zero-shot (학습 없음)", **cost})
    common.save_json(run_dir / "training_summary.json", summary)
    save_validation_files(run_dir, rows, labels, scores, predictions, probabilities)

    log = (f"  [{ZERO_SHOT_KEY}/{run_dir.name}] {topic}/{dataset} validation {len(rows)}행 | "
           f"잘린 행 {truncated} | val_loss={loss:.4f} val_macro_f1={scores['macro_f1']:.4f}\n"
           f"  [{ZERO_SHOT_KEY}/{run_dir.name}] validation 정답: {common.format_correct(scores, labels)}")
    print(log)
    (run_dir / "train.log").write_text(log + "\n", encoding="utf-8")
    write_main(run_dir, None, {"fine_tuned": False, "max_len": S.MAX_LEN}, zero_shot=True)

    common.finalize_run(ZERO_SHOT_KEY, topic, dataset, run_dir, summary, run_dir / common.WEIGHTS_TMP)
    common.mark_complete(run_dir)
    common.free_memory()
    return summary


# ═════════════════════════ 2. 파인튜닝 (에폭마다 validation) ═════════════════════════
def run_finetune(topic: str, dataset: str, data: dict, enc, device, seed: int, epochs: int | None = None) -> dict:
    run_dir = common.begin_run(FINETUNE_KEY, topic, dataset)
    try:
        return _finetune(run_dir, topic, dataset, data, enc, device, seed, epochs)
    except BaseException:
        common.safe_delete(run_dir)
        raise


def _finetune(run_dir: Path, topic: str, dataset: str, data: dict, enc, device, seed: int,
              epochs: int | None) -> dict:
    tc, cc = cfg.TRAINING_CONFIG, S.CLM_CONFIG
    labels, rows = data["labels"], data["csv"]["validation"]
    train_examples, validation_examples = data["examples"]["train"], data["examples"]["validation"]
    run_name = run_dir.name
    weights_dir = run_dir / common.WEIGHTS_TMP
    max_epochs = epochs or tc["epochs"]
    lines = []

    def say(message):
        print(message)
        lines.append(message)

    set_seed(seed)
    base_path = base_head_path()
    head = HeadModel.from_checkpoint(base_path, device)
    train_cut, validation_cut = count_truncated(enc, train_examples), count_truncated(enc, validation_examples)

    # 인코더는 학습하지 않으므로 임베딩은 한 번만 계산해 캐시에 두고, 에폭마다 헤드만 학습한다.
    unique = list(dict.fromkeys(t for e in train_examples for t in (e.state_text, *e.candidates)))
    train_vectors, train_seconds_each = enc.embed(unique)
    text_id = {text: index for index, text in enumerate(unique)}
    emb = torch.from_numpy(train_vectors).to(device)
    kmax = max(len(e.candidates) for e in train_examples)
    state_idx = torch.tensor([text_id[e.state_text] for e in train_examples], device=device)
    option_idx = torch.tensor([[text_id[t] for t in e.candidates] + [-1] * (kmax - len(e.candidates))
                               for e in train_examples], device=device)
    option_target = torch.tensor([[float(i == e.label) for i in range(len(e.candidates))]
                                  + [0.0] * (kmax - len(e.candidates)) for e in train_examples], device=device)
    validation_state, validation_seconds_each = enc.embed([e.state_text for e in validation_examples])
    validation_candidates, _ = enc.embed(shared_candidates(validation_examples))
    actual = label_ids(rows, labels)
    embed_seconds = {"train": float(train_seconds_each[[text_id[e.state_text] for e in train_examples]].sum()),
                     "validation": float(validation_seconds_each.sum())}

    batch = cc["batch"]
    per_epoch = math.ceil(len(train_examples) / batch)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cc["lr"], weight_decay=cc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cc["lr"], total_steps=per_epoch * max_epochs, pct_start=cc["warmup_ratio"],
        anneal_strategy="cos")
    patience, target = tc["early_stopping_patience"], tc["target_macro_f1"]
    score_stop = common.score_stop_settings()
    tracker = common.EpochTracker(patience, target, score_stop)

    say(f"  [{FINETUNE_KEY}/{run_name}] 설정: {topic}/{dataset} | train {len(train_examples)}행 / "
        f"validation {len(validation_examples)}행 | LR {cc['lr']:g} | loss {cc['loss']} | "
        f"에폭 최대 {max_epochs} | 배치 {batch} (에폭당 {per_epoch}스텝) | 헤드 {device}")
    say(f"  [{FINETUNE_KEY}/{run_name}] 중단 조건: "
        f"{f'Macro-F1 {patience}에폭 연속 개선 없음' if patience else '조기 종료 끔'}"
        f"{f' / Macro-F1 {target} 이상' if target is not None else ''} | "
        f"점수 기준 종료: {common.describe_score_stop(score_stop)}")
    say(f"  [{FINETUNE_KEY}/{run_name}] 입력: max_len {S.MAX_LEN} | 잘린 행: train {train_cut}/{len(train_examples)}, "
        f"validation {validation_cut}/{len(validation_examples)} | 시작 헤드 {S.BASE_HEAD['file']} | "
        f"인코더 {S.ENCODER['gguf_file']}")

    peak_mb, peak_scope = enc.gpu_memory()
    common.save_json(run_dir / "run_config.json", {
        "topic": topic, "dataset": dataset, "model_key": FINETUNE_KEY, "model_id": S.CLM_MODEL_ID,
        "run_name": run_name, "learning_rate": cc["lr"], "loss": cc["loss"], "epochs_max": max_epochs,
        "early_stopping_patience": patience, "target_macro_f1": target, "score_stop": score_stop,
        "batch_size": batch, "gradient_accumulation_steps": 1, "max_length": S.MAX_LEN, "truncation": "tail",
        "truncated_rows": {"train": train_cut, "validation": validation_cut},
        "weight_decay": cc["weight_decay"], "scheduler": "OneCycleLR (cos)", "warmup_ratio": cc["warmup_ratio"],
        "grad_clip": cc["grad_clip"], "seed": seed, "labels": labels,
        "train_rows": len(train_examples), "validation_rows": len(validation_examples),
        "train_csv": common.relative_paths(common.dataset_paths(topic, dataset)["train"]),
        "validation_csv": common.relative_paths(common.dataset_paths(topic, dataset)["validation"]),
        "trainable": "state/action head only (encoder frozen)", "init_head": str(base_path),
        "encoder": enc.describe(), "gpu_memory_scope": peak_scope, "precision": "float32 (head)",
        **environment(device, enc)})

    generator = torch.Generator().manual_seed(seed)
    state = {"best": None}
    started = time.perf_counter()
    epochs_run = max_epochs
    for epoch in range(1, max_epochs + 1):
        head.train()
        total, count = 0.0, 0
        for idx in torch.randperm(len(train_examples), generator=generator).split(batch):
            loss = infonce_loss(head, emb, state_idx, option_idx, option_target, idx.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cc["grad_clip"])
            optimizer.step()
            scheduler.step()
            total += loss.item()
            count += 1
        train_loss = total / max(1, count)

        logits, head_seconds = head.score(validation_state, validation_candidates)
        predictions, probabilities = logits.argmax(-1).tolist(), torch.softmax(logits, dim=-1).tolist()
        val_loss = softmax_loss(logits, actual)
        scores = common.classification_scores(actual, predictions, labels)
        correct, total_rows, _ = common.correct_counts(scores, labels)
        is_best = tracker.update(scores["macro_f1"], val_loss, correct, total_rows)
        elapsed = time.perf_counter() - started
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                  "validation_accuracy": scores["accuracy"], "validation_macro_f1": scores["macro_f1"],
                  "optimizer_steps": per_epoch * epoch, "elapsed_seconds": elapsed}
        with (run_dir / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        say(f"  [{FINETUNE_KEY}/{run_name}] epoch {epoch}/{max_epochs}  "
            f"train_loss({cc['loss']})={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_macro_f1={scores['macro_f1']:.4f}  정답 {correct}/{total_rows} ({correct / total_rows:.1%})"
            f"{'  <- best' if is_best else ''}  ({elapsed:.0f}s)")
        if is_best:
            head.save(weights_dir)
            common.save_json(weights_dir / QUESTIONS_FILE, data["questions"])
            common.save_json(weights_dir / ENCODER_FILE, encoder_record(enc, {"trained_on": f"{topic}/{dataset}"}))
            state["best"] = {"epoch": epoch, "scores": scores, "loss": val_loss, "predictions": predictions,
                             "probs": probabilities, "head_seconds": head_seconds}
        if tracker.should_stop:
            epochs_run = epoch
            say(f"  [{FINETUNE_KEY}/{run_name}] 조기 종료: epoch {epochs_run}/{max_epochs} - {tracker.stop_reason}")
            break

    best = state["best"]
    train_seconds = time.perf_counter() - started
    stop_reason = tracker.stop_reason or f"최대 에폭({max_epochs}) 도달"
    peak_mb, peak_scope = enc.gpu_memory()
    cost = cost_fields(enc, head.num_parameters(), weights_dir / "head.safetensors")
    summary = common.make_summary(
        FINETUNE_KEY, S.CLM_MODEL_ID, run_name, cc["lr"], best["scores"], best["loss"], best["epoch"],
        train_seconds, embed_seconds["validation"] + best["head_seconds"], peak_mb, len(train_examples),
        len(validation_examples),
        extra={"epochs_run": epochs_run, "epochs_max": max_epochs, "stop_reason": stop_reason,
               "embed_seconds_train": embed_seconds["train"], "embed_seconds_validation": embed_seconds["validation"],
               **cost})
    common.save_json(run_dir / "training_summary.json", summary)
    save_validation_files(run_dir, rows, labels, best["scores"], best["predictions"], best["probs"])

    gpu_note = f", 인코더 GPU {peak_mb:.0f}MB" if peak_mb is not None else ""
    say(f"  [{FINETUNE_KEY}/{run_name}] 완료: best epoch {best['epoch']} (총 {epochs_run}에폭 학습), "
        f"val_macro_f1={best['scores']['macro_f1']:.4f}, 헤드 학습 {train_seconds:.0f}초{gpu_note}")
    say(f"  [{FINETUNE_KEY}/{run_name}] validation 정답(best epoch {best['epoch']}): "
        f"{common.format_correct(best['scores'], labels)}")
    (run_dir / "train.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_main(run_dir, weights_dir, {"loss": cc["loss"], "max_len": S.MAX_LEN, "fine_tuned": True})

    common.finalize_run(FINETUNE_KEY, topic, dataset, run_dir, summary, weights_dir)
    common.mark_complete(run_dir)
    common.free_memory()
    return summary


# ═════════════════════════ 3. test 평가 (best_model, 모델마다 1회) ═════════════════════════
def evaluate_test(topic: str, dataset: str, enc, device) -> None:
    data = load_split_rows(topic, dataset, with_test=True)
    labels, rows = data["labels"], data["csv"]["test"]
    actual = label_ids(rows, labels)
    # CLM은 동료 코드의 모델 추천 대상이 아니므로 공식(official) 점수가 아니라 참고용으로 기록한다.
    official = None if common.is_recommend_excluded(dataset) else False
    for key in CLM_KEYS:
        best = common.best_dir(key, topic, dataset)
        best_info = common.read_best_info(key, topic, dataset)
        if best_info is None:
            print(f"  [{key}] best_model이 없어 건너뜁니다.")
            continue
        if (best / "test_metrics.json").exists():
            print(f"  [{key}] 이미 test 평가를 했습니다. 덮어쓰지 않고 건너뜁니다.")
            continue
        if key == ZERO_SHOT_KEY:
            classifier = ClmClassifier.from_base_head(data["questions"], enc, device)
        else:
            classifier = ClmClassifier.from_best_dir(best, enc, device)
            if classifier.questions != data["questions"]:
                print(f"  [주의] {key}: 학습 때의 질문/선택지 설명이 지금 labels.json과 다릅니다. 학습 때 것으로 평가합니다.")
        truncated = count_truncated(enc, data["examples"]["test"])
        logits, seconds = classifier.predict_logits([row[cfg.TEXT_COLUMN] for row in rows])
        predictions, probabilities = logits.argmax(-1).tolist(), torch.softmax(logits, dim=-1).tolist()
        scores = common.classification_scores(actual, predictions, labels)
        common.save_json(best / "test_metrics.json", {
            "topic": topic, "dataset": dataset, "split": "test", "selected_model": None,
            "evaluated_model": key, "official": official, "selected_run": best_info["best"]["run_name"],
            "model_id": S.CLM_MODEL_ID, "classifier": scores, "baseline": None})
        common.save_errors(best / "test_errors.csv", rows, predictions, labels)
        result_files.write_predictions(best / "test_predictions.csv", rows, predictions, probabilities, labels)
        result_files.record_test_result(best, scores, labels, len(rows), seconds, truncated,
                                        official=official, with_meta=True)
        print(f"  [{key}] test macro F1 {scores['macro_f1']:.4f} (accuracy {scores['accuracy']:.4f}) [참고]")
        print(f"  [{key}] test 정답: {common.format_correct(scores, labels)}")
        common.free_memory()


def main():
    ap = argparse.ArgumentParser(description="CLM zero-shot 평가 + 판단 헤드 파인튜닝")
    ap.add_argument("--topic", choices=cfg.TOPICS, help="생략하면 ON인 주제 전부")
    ap.add_argument("--dataset", choices=cfg.DATASET_NAMES, help="생략하면 ON인 데이터셋 전부")
    ap.add_argument("--zero-shot-only", action="store_true", help="파인튜닝 없이 zero-shot 평가만")
    ap.add_argument("--finetune-only", action="store_true", help="zero-shot은 건너뛰고 파인튜닝만 (zero-shot은 같은 결과가 반복되므로 설정을 바꿔 다시 학습할 때 쓴다)")
    ap.add_argument("--epochs", type=int, default=None, help="최대 에폭 임시 변경 (기본: config 값)")
    ap.add_argument("--seed", type=int, default=None, help="seed 임시 변경 (기본: config의 seed)")
    ap.add_argument("--device", default=None, help="헤드 학습 장치 cpu/cuda (기본: clm_settings의 head_device)")
    ap.add_argument("--test", action="store_true", help="학습 대신 best_model로 test 평가")
    a = ap.parse_args()
    if a.zero_shot_only and a.finetune_only:
        ap.error("--zero-shot-only와 --finetune-only는 함께 쓸 수 없습니다.")

    pairs = common.ready_pairs(a.topic, a.dataset)
    if not pairs:
        print("실행할 데이터셋이 없습니다. baseline_bert_config.py의 DATASET_SWITCH를 확인하세요.")
        return
    device = common.resolve_device(a.device or S.CLM_CONFIG["head_device"])
    seed = a.seed if a.seed is not None else cfg.TRAINING_CONFIG["seed"]
    with common.session_log("test_clm" if a.test else "train_clm"):
        print(f"CLM: {S.CLM_MODEL_ID} | 헤드 장치 {device} | seed {seed}")
        print(f"CLM 설정: {S.CLM_CONFIG}")
        print(f"인코더 설정: {S.ENCODER}")
        failures = []
        try:
            with ensure_encoder() as enc:
                for topic, dataset in pairs:
                    print(f"\n=== {topic}/{dataset} · clm ===")
                    try:
                        if a.test:
                            evaluate_test(topic, dataset, enc, device)
                            continue
                        data = load_split_rows(topic, dataset)
                        # 파인튜닝 전에 zero-shot 지표를 먼저 기록한다.
                        if not a.finetune_only:
                            run_zero_shot(topic, dataset, data, enc, device, seed)
                        if not a.zero_shot_only:
                            run_finetune(topic, dataset, data, enc, device, seed, a.epochs)
                    except Exception as error:
                        print(f"  [clm] {topic}/{dataset} 실패: {type(error).__name__}: {error}")
                        failures.append(f"{topic}/{dataset}")
                    common.free_memory()
        except (FileNotFoundError, RuntimeError, TimeoutError) as error:
            print(f"\n[인코더 준비 실패] {error}")
            sys.exit(1)
        if failures:
            print(f"\n실패한 데이터셋: {failures}")
            sys.exit(1)


if __name__ == "__main__":
    main()
