"""공통 함수 모음 (baseline/steps/common.py).

  1. 경로 · 날짜 run 폴더 · ON/OFF      4. best_model 비교·교체 (run 이 끝날 때마다)
  2. 데이터 읽기 · 검사                  5. 사전학습 모델 학습 루프 (조기 종료 포함)
  3. 지표 · 오분류 저장                  6. 모델 추천 / test 최종 평가 / 초기화

결과 폴더 구조 (모든 모델이 한 곳에 모인다):
  artifacts/<topic>/<dataset>/<baseline | bert | roberta | koelectra>/
      best_model/        지금까지 validation 이 가장 좋았던 결과 + 가중치 + best_info.json
      20261012_195030/   학습 1회 = 폴더 1개 (점수·로그만. 가중치는 best 일 때 best_model 로 옮겨 감)

torch, transformers 는 학습·평가 함수 안에서만 불러옵니다.
(메뉴 시작과 TF-IDF 학습이 느려지지 않도록)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import platform
import random
import re
import shutil
import sys
import time
import traceback
import unicodedata
from pathlib import Path

# 설정 파일(baseline_bert_config.py)은 프로젝트 루트(project_2)에 있다. (이 파일에서 두 단계 위)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import baseline_bert_config as cfg  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

BASELINE_KEY = "baseline"                                # TF-IDF + 로지스틱 회귀
TRANSFORMER_KEYS = tuple(cfg.PRETRAINED_MODELS)          # ("bert", "roberta", "koelectra")
ALL_MODEL_KEYS = (BASELINE_KEY,) + TRANSFORMER_KEYS

RUN_NAME_RE = re.compile(r"^\d{8}_\d{6}$")               # 20261012_195030
WEIGHTS_TMP = "_weights_tmp"                              # 학습 중 가중치를 잠시 두는 폴더 (끝나면 옮기거나 지움)
BEST_INFO = "best_info.json"


# ═════════════════════════ 1. 경로 · 날짜 run 폴더 · ON/OFF ═════════════════════════
def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_enabled(topic, dataset):
    return bool(cfg.DATASET_SWITCH.get(topic, {}).get(dataset, False))


def is_recommend_excluded(dataset):
    """True 면 이 데이터셋은 모델 추천에서 빠진다. (서버용 total_merge)"""
    return dataset in cfg.RECOMMEND_EXCLUDED


# 예전 데이터셋 이름 -> 새 이름. 예전 이름으로 저장된 결과 폴더를 이어서 쓰기 위한 정보다.
LEGACY_DATASET_NAMES = {"basic_plus": "basic_merge", "advanced_plus": "advanced_merge"}
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".log", ".md", ".txt"}


def legacy_result_folders():
    """artifacts 에 남아 있는 예전 이름 폴더 목록: [(topic, 옛 이름, 새 이름)]"""
    return [(topic, old, new) for topic in cfg.TOPICS for old, new in LEGACY_DATASET_NAMES.items()
            if (cfg.ARTIFACTS_ROOT / topic / old).exists()]


def _replace_legacy_names_in_files(folder):
    """폴더 안 텍스트 파일(json/csv 등)에 적힌 옛 데이터셋 이름을 새 이름으로 바꾼다. 바이트 단위라 BOM 유무는 그대로다."""
    changed = 0
    for path in Path(folder).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        data = path.read_bytes()
        updated = data
        for old, new in LEGACY_DATASET_NAMES.items():
            updated = updated.replace(old.encode("utf-8"), new.encode("utf-8"))
        if updated != data:
            path.write_bytes(updated)
            changed += 1
    return changed


def migrate_legacy_dataset_names():
    """예전 이름(basic_plus, advanced_plus)으로 저장된 결과 폴더를 새 이름(basic_merge, advanced_merge)으로 옮긴다.

    폴더 이름을 바꾸고, 그 안의 json/csv 에 적힌 옛 이름도 함께 바꾼다. 새 이름 폴더에 이미 내용이 있으면 덮어쓰지 않고 건너뛴다.
    반환: (옮긴 목록, 건너뛴 목록, 실패 목록)
    """
    moved, skipped, failed = [], [], []
    for topic, old, new in legacy_result_folders():
        src, dst = cfg.ARTIFACTS_ROOT / topic / old, cfg.ARTIFACTS_ROOT / topic / new
        label = f"{topic}/{old} -> {topic}/{new}"
        if dst.exists() and any(dst.iterdir()):
            skipped.append(label + " (새 이름 폴더에 이미 내용이 있어 건너뜀)")
            continue
        try:
            if dst.exists():
                dst.rmdir()
            src.rename(dst)
            _replace_legacy_names_in_files(dst)
            moved.append(label)
        except OSError as error:
            failed.append(f"{label} ({type(error).__name__}: {error})")
    viz = cfg.ARTIFACTS_ROOT / "viz"
    if moved and viz.exists():
        _replace_legacy_names_in_files(viz)      # 모아 둔 시각화 표의 옛 이름도 바꾼다
    return moved, skipped, failed


def dataset_paths(topic, dataset):
    """한 데이터셋이 사용하는 파일 경로. train/validation/test 는 파일 목록(list)이다.

    폴더는 <DATA_ROOT>/<topic>/<DATASET_FOLDERS[dataset]>/ 이고, 폴더마다 같은 이름의 파일이 들어 있다.
    *_merge 폴더는 이미 합쳐진 파일이라 파일을 따로 이어 붙이지 않는다.
    """
    folder = Path(cfg.DATA_ROOT) / topic / cfg.DATASET_FOLDERS[dataset]
    paths = {"labels": folder / cfg.DATA_FILES["labels"]}
    for split in ("train", "validation", "test"):
        paths[split] = [folder / cfg.DATA_FILES[split]]   # 값은 파일 목록(list) 형태를 유지해 다른 코드는 그대로 쓴다
    return paths


def relative_paths(paths):
    """기록용: 프로젝트 루트 기준 상대 경로 문자열 목록."""
    return [Path(path).relative_to(cfg.PROJECT_ROOT).as_posix() for path in paths]


def missing_files(topic, dataset):
    paths = dataset_paths(topic, dataset)
    needed = [paths["labels"]] + paths["train"] + paths["validation"] + paths["test"]
    return [path for path in needed if not path.exists()]


def all_pairs():
    return [(topic, dataset) for topic in cfg.TOPICS for dataset in cfg.DATASET_NAMES]


def ready_pairs(topic=None, dataset=None):
    """ON 이고 데이터 파일이 모두 있는 (topic, dataset) 목록. topic/dataset 로 좁힐 수 있다."""
    pairs = []
    for t, d in all_pairs():
        if (topic and t != topic) or (dataset and d != dataset):
            continue
        if not is_enabled(t, d):
            if topic or dataset:   # 직접 지정했는데 꺼져 있으면 이유를 알려 준다
                print(f"[건너뜀] {t}/{d}: baseline_bert_config.py 의 DATASET_SWITCH 가 False(OFF)입니다.")
            continue
        missing = missing_files(t, d)
        if missing:
            print(f"[건너뜀] {t}/{d}: 데이터 파일이 없습니다 -> {missing[0]}")
            continue
        pairs.append((t, d))
    return pairs


def dataset_result_dir(topic, dataset):
    """데이터셋 폴더: 4개 모델 폴더와 모델 간 비교/추천 결과가 모이는 곳."""
    return cfg.ARTIFACTS_ROOT / topic / dataset


def model_dir(model_key, topic, dataset):
    return dataset_result_dir(topic, dataset) / model_key


def best_dir(model_key, topic, dataset):
    return model_dir(model_key, topic, dataset) / cfg.BEST_DIR_NAME


def model_learning_rates(model_key):
    """config 의 LEARNING_RATES 를 항상 리스트로 돌려 준다. (숫자 하나여도, 리스트여도 동작)"""
    value = cfg.LEARNING_RATES[model_key]
    return list(value) if isinstance(value, (list, tuple)) else [value]


def run_dirs(model_key, topic, dataset):
    """날짜 이름(20261012_195030)의 run 폴더 목록. 오래된 순."""
    base = model_dir(model_key, topic, dataset)
    if not base.exists():
        return []
    return sorted(d for d in base.iterdir() if d.is_dir() and RUN_NAME_RE.match(d.name))


def run_is_complete(run_dir):
    """run_status.json 은 학습이 끝까지 성공했을 때만 만든다. (중간에 끊긴 run 구분용)"""
    return (Path(run_dir) / "run_status.json").exists()


def complete_run_dirs(model_key, topic, dataset):
    return [d for d in run_dirs(model_key, topic, dataset) if run_is_complete(d)]


def mark_complete(run_dir):
    save_json(Path(run_dir) / "run_status.json",
              {"status": "complete", "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})


def new_run_dir(model_key, topic, dataset):
    """오늘 날짜·시간(초)으로 새 run 폴더를 만든다. 같은 초에 이미 있으면 1초씩 늦춘다."""
    base = model_dir(model_key, topic, dataset)
    base.mkdir(parents=True, exist_ok=True)
    moment = time.time()
    while True:
        path = base / time.strftime(cfg.RUN_TIME_FORMAT, time.localtime(moment))
        try:
            path.mkdir()
            return path
        except FileExistsError:
            moment += 1


def clean_incomplete_runs(model_key, topic, dataset):
    """중간에 끊겨 완료 표시가 없는 run 폴더를 지운다. (이 프로그램이 만든 날짜 폴더만 대상)"""
    for run_dir in run_dirs(model_key, topic, dataset):
        if not run_is_complete(run_dir):
            print(f"  [{model_key}] 중간에 끊긴 run 폴더를 지웁니다: {run_dir.name}")
            safe_delete(run_dir)


def begin_run(model_key, topic, dataset):
    """학습 시작 전: 끊긴 run 을 정리하고 새 날짜 폴더를 만든다."""
    clean_incomplete_runs(model_key, topic, dataset)
    return new_run_dir(model_key, topic, dataset)


def safe_delete(path):
    """artifacts 폴더 안의 파일/폴더만 삭제한다. 데이터(CSV)나 코드는 지울 수 없다."""
    path = Path(path).resolve()
    if cfg.ARTIFACTS_ROOT.resolve() not in path.parents:
        raise ValueError(f"artifacts 폴더 안의 것만 삭제할 수 있습니다: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def dir_size_mb(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 2**20
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20


# ── 로그: 화면 출력을 파일에도 저장 / 현재 설정 출력 ──
class _Tee:
    """print 출력을 화면과 파일에 함께 기록한다."""

    def __init__(self, stream, file):
        self._stream, self._file = stream, file

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()   # 파일에도 바로 기록한다. (비정상 종료 때 마지막 줄이 빠지지 않게)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def __getattr__(self, name):   # isatty, encoding 등은 원래 화면 스트림 것을 그대로 쓴다
        return getattr(self._stream, name)


@contextlib.contextmanager
def session_log(name):
    """with 블록 안의 모든 print 출력을 LOG_DIR/<name>_<시각>.log 에도 저장한다."""
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.LOG_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    original = sys.stdout
    with path.open("w", encoding="utf-8") as file:
        sys.stdout = _Tee(original, file)
        try:
            print(f"[로그 파일] {path}")
            yield path
        finally:
            sys.stdout = original


def describe_settings():
    """현재 실험 설정을 사람이 읽는 줄 목록으로 만든다. (로그 맨 위에 남긴다)"""
    tc = cfg.TRAINING_CONFIG
    truncation = tc["truncation"]
    if truncation == "head_tail":
        truncation += f"(앞 {tc['head_tokens']} + 뒤 {tc['max_length'] - 2 - tc['head_tokens']})"
    patience = tc["early_stopping_patience"]
    target = tc["target_macro_f1"]
    on = [f"{t}/{d}" for t, d in all_pairs() if is_enabled(t, d)]
    lines = [
        f"실행 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"ON 데이터셋: {', '.join(on) if on else '없음'}",
        f"데이터 폴더: {cfg.DATA_ROOT} (모델 추천 제외: {', '.join(cfg.RECOMMEND_EXCLUDED) or '없음'})",
        f"에폭 최대 {tc['epochs']} | 조기 종료: {f'Macro-F1 {patience}에폭 연속 개선 없으면 중단' if patience else '끔'} | "
        f"목표 점수 중단: {f'Macro-F1 {target} 이상' if target is not None else '끔'}",
        f"배치 {tc['batch_size']}문장 x 누적 {tc['gradient_accumulation_steps']} "
        f"(실질 {tc['batch_size'] * tc['gradient_accumulation_steps']}문장마다 가중치 수정) | "
        f"weight_decay {tc['weight_decay']} | warmup {tc['warmup_ratio']} | seed {tc['seed']}",
        f"입력: max_length {tc['max_length']} | 자르는 방식 {truncation} | "
        f"gradient_checkpointing {tc['gradient_checkpointing']}",
        "Learning Rate: " + " | ".join(f"{k} {model_learning_rates(k)}" for k in cfg.LEARNING_RATES),
        "모델: " + " | ".join(f"{k} {v['model_id']}" for k, v in cfg.PRETRAINED_MODELS.items()),
        f"TF-IDF: {cfg.TFIDF_CONFIG} | 로지스틱 회귀: {cfg.LOGISTIC_REGRESSION_CONFIG}",
        f"결과 저장: {cfg.ARTIFACTS_ROOT}  (모델 폴더마다 {cfg.BEST_DIR_NAME} + 날짜 폴더)",
    ]
    # 에폭·조기 종료 줄 바로 아래에 점수 기준 종료 설정을 넣는다.
    lines.insert(4, f"점수 기준 종료: {describe_score_stop(score_stop_settings())}")
    return lines


def print_settings():
    print("[현재 설정]")
    for line in describe_settings():
        print("  " + line)


# ═════════════════════════ 2. 데이터 읽기 · 검사 ═════════════════════════
def read_rows(path):
    # utf-8-sig: Excel 이 만든 BOM 이 있는 CSV 도 같은 방식으로 읽는다.
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_split(paths):
    """파일 목록을 순서대로 읽어 하나의 분할로 합친다. (원본 + add_*.csv)"""
    rows = []
    for path in paths:
        rows.extend(read_rows(path))
    return rows


def read_labels(path):
    labels = json.loads(Path(path).read_text(encoding="utf-8-sig"))["labels"]
    if (not isinstance(labels, list) or not labels
            or not all(isinstance(x, str) and x.strip() for x in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("labels.json 에는 중복 없는 라벨 문자열 목록이 필요합니다.")
    return labels


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def check_splits(splits, labels):
    """분할 데이터 검사.

    에러(ValueError): 빈 값, 모르는 라벨, id 중복(분할 안/밖 모두), 분할 간 group_id·문장 겹침
    경고(report["warnings"]): 같은 group_id 안에 서로 다른 라벨이 섞인 경우(advanced 데이터는 정상),
                              어떤 분할에 특정 라벨이 하나도 없는 경우
    """
    seen = {"id": set(), "group_id": set(), "text": set()}
    report = {"splits": {}, "warnings": []}
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"{name}: 데이터가 비어 있습니다.")
        for number, row in enumerate(rows, start=2):
            for column in cfg.REQUIRED_COLUMNS:
                if not (row.get(column) or "").strip():
                    raise ValueError(f"{name}: {number}행의 '{column}' 값이 비어 있습니다.")
        row_labels = {row[cfg.LABEL_COLUMN] for row in rows}
        unknown = row_labels - set(labels)
        if unknown:
            raise ValueError(f"{name}: labels.json 에 없는 라벨이 있습니다 -> {sorted(unknown)}")
        if row_labels != set(labels):
            report["warnings"].append(f"{name}: 라벨 {sorted(set(labels) - row_labels)} 가 없습니다.")

        ids = [row[cfg.ID_COLUMN] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name}: 같은 분할 안에 중복된 id 가 있습니다.")
        group_labels = {}
        for row in rows:
            group_labels.setdefault(row[cfg.GROUP_COLUMN], set()).add(row[cfg.LABEL_COLUMN])
        mixed = sum(1 for values in group_labels.values() if len(values) > 1)
        if mixed:
            report["warnings"].append(
                f"{name}: 한 group_id 에 서로 다른 라벨이 섞인 그룹 {mixed}개 "
                "(advanced 는 정상, basic 은 같은 그룹의 라벨이 같아야 하므로 확인 필요)")

        current = {"id": set(ids), "group_id": set(group_labels),
                   "text": {normalize_text(row[cfg.TEXT_COLUMN]) for row in rows}}
        for key, previous in seen.items():
            overlap = previous & current[key]
            if overlap:
                example = sorted(overlap)[0][:40]
                raise ValueError(f"{name}: 앞선 분할과 {key} 가 {len(overlap)}건 겹칩니다 (예: {example})")
            previous.update(current[key])
        report["splits"][name] = {"rows": len(rows), "groups": len(group_labels),
                                  "class_counts": {label: sum(1 for r in rows if r[cfg.LABEL_COLUMN] == label)
                                                   for label in labels}}
    return report


def load_data(topic, dataset, with_test=False):
    """train / validation 을 읽고 검사한다. test 는 최종 평가(with_test=True)에서만 읽는다."""
    paths = dataset_paths(topic, dataset)
    labels = read_labels(paths["labels"])
    if labels != cfg.LABELS[topic]:
        raise ValueError(f"labels.json {labels} 와 baseline_bert_config.LABELS {cfg.LABELS[topic]} 가 다릅니다.")
    splits = {"train": read_split(paths["train"]), "validation": read_split(paths["validation"])}
    if with_test:
        splits["test"] = read_split(paths["test"])
    report = check_splits(splits, labels)
    for message in report["warnings"]:
        print("  [경고]", message)
    return {"labels": labels, "report": report, **splits}


# ═════════════════════════ 3. 지표 · 오분류 저장 ═════════════════════════
def classification_scores(actual, predicted, labels):
    ids = list(range(len(labels)))   # 0, 1, 2 를 항상 같은 라벨 순서로 해석한다.
    return {
        "n": len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=ids, average="macro", zero_division=0)),
        "per_class": classification_report(actual, predicted, labels=ids, target_names=labels,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=ids).tolist(),
        "confusion_matrix_axes": {"rows": "actual", "columns": "predicted", "labels": labels},
    }


def correct_counts(scores, labels):
    """혼동행렬에서 정답 개수를 센다. 반환: (전체 정답 수, 전체 문장 수, {라벨: (정답 수, 문장 수)})"""
    matrix = scores["confusion_matrix"]     # 행 = 실제 라벨, 열 = 예측 라벨. 대각선이 정답
    per_label = {label: (matrix[i][i], sum(matrix[i])) for i, label in enumerate(labels)}
    correct = sum(matrix[i][i] for i in range(len(labels)))
    total = sum(sum(row) for row in matrix)
    return correct, total, per_label


def format_correct(scores, labels):
    """'114/120 (95.0%)  [라벨별] shipping 38/40 (95.0%) | ...' 형태의 한 줄을 만든다."""
    correct, total, per_label = correct_counts(scores, labels)
    detail = " | ".join(f"{label} {ok}/{n} ({ok / n:.1%})" if n else f"{label} 0/0"
                        for label, (ok, n) in per_label.items())
    return f"{correct}/{total} ({correct / total:.1%})  [라벨별] {detail}"


def save_errors(path, rows, predictions, labels):
    fields = ["id", "text", "actual", "predicted", "group_id"]
    errors = [{"id": row[cfg.ID_COLUMN], "text": row[cfg.TEXT_COLUMN], "actual": row[cfg.LABEL_COLUMN],
               "predicted": labels[prediction], "group_id": row[cfg.GROUP_COLUMN]}
              for row, prediction in zip(rows, predictions, strict=True)
              if row[cfg.LABEL_COLUMN] != labels[prediction]]
    write_csv(path, fields, errors)


def make_summary(model_key, model_id, run_name, learning_rate, scores, loss, best_epoch,
                 train_seconds, inference_seconds, peak_gpu_mb, train_rows, validation_rows, extra=None):
    """run 하나의 핵심 결과. best_model 비교와 모델 추천은 이 값만 읽는다."""
    summary = {"model_key": model_key, "model_id": model_id, "run_name": run_name,
               "learning_rate": learning_rate, "best_epoch": best_epoch,
               "validation_macro_f1": scores["macro_f1"], "validation_accuracy": scores["accuracy"],
               "validation_loss": loss, "train_seconds": train_seconds,
               "validation_inference_seconds": inference_seconds, "peak_gpu_memory_mb": peak_gpu_mb,
               "train_rows": train_rows, "validation_rows": validation_rows}
    summary.update(extra or {})
    return summary


# ═════════════════════════ 4. best_model 비교 · 교체 ═════════════════════════
def model_score(summary):
    """같은 모델의 run 끼리 비교하는 점수: validation Macro-F1 이 높을수록, 동점이면 validation loss 가 낮을수록 좋다."""
    loss = summary.get("validation_loss")
    return (round(summary["validation_macro_f1"], 6), -(loss if loss is not None else 0.0))


def decide_best(new, old_best):
    """반환: (best_model 을 교체할지, 이유 문장)"""
    if old_best is None:
        return True, "이 모델의 첫 결과라서 best_model 로 저장"
    new_f1, new_loss = new["validation_macro_f1"], new.get("validation_loss")
    old_f1, old_loss = old_best["validation_macro_f1"], old_best.get("validation_loss")
    n, o = model_score(new), model_score(old_best)
    old_run = old_best["run_name"]
    if n[0] > o[0]:
        return True, f"validation Macro-F1 {new_f1:.4f} > 이전 best {old_f1:.4f} ({old_run})"
    if n[0] == o[0] and n[1] > o[1]:
        return True, (f"Macro-F1 동점({new_f1:.4f}) -> validation loss {new_loss:.4f} < "
                      f"이전 best {old_loss:.4f} ({old_run})")
    if n[0] == o[0] and n[1] == o[1]:
        return False, f"이전 best({old_run})와 점수가 같아 유지 (Macro-F1 {new_f1:.4f})"
    if n[0] == o[0]:
        return False, (f"Macro-F1 동점({new_f1:.4f})이지만 validation loss "
                       f"{new_loss:.4f} >= 이전 best {old_loss:.4f} ({old_run}) 라서 유지")
    return False, f"validation Macro-F1 {new_f1:.4f} <= 이전 best {old_f1:.4f} ({old_run}) 라서 유지"


def read_best_info(model_key, topic, dataset):
    path = best_dir(model_key, topic, dataset) / BEST_INFO
    return load_json(path) if path.exists() else None


def collect_runs(model_key, topic, dataset, extra=None):
    """완료된 run 들의 training_summary 목록. extra: 아직 완료 표시 전인 방금 run 의 summary."""
    summaries = [load_json(d / "training_summary.json") for d in complete_run_dirs(model_key, topic, dataset)]
    if extra is not None and all(s["run_name"] != extra["run_name"] for s in summaries):
        summaries.append(extra)
    return summaries


RUN_FIELDS = ["run_name", "learning_rate", "best_epoch", "validation_macro_f1", "validation_accuracy",
              "validation_loss", "train_seconds", "peak_gpu_memory_mb"]


def finalize_run(model_key, topic, dataset, run_dir, summary, weights_dir):
    """run 하나가 끝난 뒤 호출: best_model 과 비교해서 더 좋으면 교체하고, 아니면 가중치를 지운다.

    - 더 좋음  : best_model 을 이번 run 의 결과 파일 + 가중치로 교체 (이전 best 의 가중치는 사라짐)
    - 아니면   : best_model 은 그대로 두고, 이번 run 의 가중치는 지움 (점수·로그는 날짜 폴더에 남음)
    어느 경우든 best_model/best_info.json 에 현재 best 와 전체 run 순위표를 다시 적는다.
    """
    run_dir, weights_dir = Path(run_dir), Path(weights_dir)
    best = best_dir(model_key, topic, dataset)
    old_info = load_json(best / BEST_INFO) if (best / BEST_INFO).exists() else None
    old_best = old_info["best"] if old_info else None
    replaced, reason = decide_best(summary, old_best)

    note = f"  [{model_key}/{run_dir.name}] best_model {'갱신' if replaced else '유지'}: {reason}"
    print(note)
    log = run_dir / "train.log"
    if log.exists():   # 판정 결과를 이 run 의 로그에도 남긴다 (best_model 로 복사되기 전에)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(note + "\n")

    best_summary = summary if replaced else old_best
    ranked = sorted(collect_runs(model_key, topic, dataset, extra=summary),
                    key=lambda s: (tuple(-x for x in model_score(s)), s["run_name"]))
    info = {
        "topic": topic, "dataset": dataset, "model_key": model_key,
        "selection_rule": "validation Macro-F1 이 높은 쪽, 동점이면 validation loss 가 낮은 쪽. "
                          "더 좋아졌을 때만 best_model 을 교체한다. (test 점수는 사용하지 않음)",
        "best": best_summary,
        "last_run": {"run_name": summary["run_name"], "replaced_best": replaced, "reason": reason,
                     "validation_macro_f1": summary["validation_macro_f1"],
                     "validation_loss": summary.get("validation_loss")},
        "previous_best": ({"run_name": old_best["run_name"],
                           "validation_macro_f1": old_best["validation_macro_f1"],
                           "validation_loss": old_best.get("validation_loss")} if replaced and old_best else None),
        "runs": [{**{k: s.get(k) for k in RUN_FIELDS}, "is_best": s["run_name"] == best_summary["run_name"]}
                 for s in ranked],
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if not replaced:
        save_json(best / BEST_INFO, info)
        if weights_dir.exists():
            safe_delete(weights_dir)
        return replaced, reason

    # 새 best 를 임시 폴더에 완성한 뒤 한 번에 바꿔 끼운다. (중간에 끊겨도 이전 best 가 망가지지 않게)
    new = best.with_name(cfg.BEST_DIR_NAME + "_new")
    if new.exists():
        safe_delete(new)
    new.mkdir(parents=True)
    for file in run_dir.iterdir():
        if file.is_file():
            shutil.copy2(file, new / file.name)          # 점수·로그·오분류·학습 조건 기록
    if weights_dir.exists():
        for file in weights_dir.iterdir():
            shutil.move(str(file), str(new / file.name)) # 가중치(+토크나이저)는 best_model 로 옮김
        safe_delete(weights_dir)
    save_json(new / BEST_INFO, info)

    old_tmp = None
    if best.exists():
        _archive_test_files(best, model_dir(model_key, topic, dataset), old_best)
        old_tmp = best.with_name(cfg.BEST_DIR_NAME + "_old")
        if old_tmp.exists():
            safe_delete(old_tmp)
        try:
            best.rename(old_tmp)
        except OSError:        # 파일이 열려 있어 이름을 못 바꾸면 그냥 지우고 교체
            safe_delete(best)
            old_tmp = None
    new.rename(best)
    if old_tmp is not None and old_tmp.exists():
        safe_delete(old_tmp)
    return replaced, reason


def _archive_test_files(old_best_dir, model_path, old_best):
    """이전 best 에 test 평가 결과가 있으면, 그 run 의 날짜 폴더로 옮겨 보관한다. (교체돼도 기록이 안 사라지게)"""
    files = [f for f in ("test_metrics.json", "test_errors.csv") if (old_best_dir / f).exists()]
    if not files:
        return
    print("  [주의] test 평가를 이미 마친 모델을 교체합니다. test 이후에 모델을 바꾸는 것이므로,")
    print("         이전 test 결과와 새 모델의 결과를 구분해 기록하세요. (이전 test 결과는 그 run 폴더에 보관)")
    target = Path(model_path) / old_best["run_name"] if old_best else None
    if target is not None and target.exists():
        for name in files:
            shutil.copy2(old_best_dir / name, target / name)


# ═════════════════════════ 5. 사전학습 모델 학습 ═════════════════════════
class EpochTracker:
    """에폭마다 점수를 받아 '이번이 best 에폭인지'와 '학습을 멈출지'를 판단한다. (torch 없이 테스트 가능)

    best 에폭 : validation Macro-F1 이 높은 에폭. 동점이면 validation loss 가 낮은 에폭.
    중단 조건 : (1) Macro-F1 이 patience 에폭 연속으로 새 최고점을 못 넘으면, (2) target 이상이 되면.
    """

    def __init__(self, patience=0, target_f1=None, score_stop=None):
        self.patience, self.target_f1, self.score_stop = patience, target_f1, score_stop
        self.best_key, self.top_f1, self.wait, self.stop_reason = None, None, 0, None
        self.epoch = 0

    def update(self, macro_f1, val_loss, correct=None, total=None):
        self.epoch += 1
        f1 = round(macro_f1, 6)
        key = (f1, -val_loss)
        is_best = self.best_key is None or key > self.best_key
        if is_best:
            self.best_key = key
        if self.top_f1 is None or f1 > self.top_f1:
            self.top_f1, self.wait = f1, 0
        else:
            self.wait += 1
        reached = self._score_reached(f1, val_loss, correct, total)
        if reached:
            self.stop_reason = reached
        elif self.target_f1 is not None and f1 >= self.target_f1:
            self.stop_reason = f"목표 점수 도달 (val_macro_f1 {f1:.4f} >= {self.target_f1})"
        elif self.patience and self.wait >= self.patience:
            self.stop_reason = f"validation Macro-F1 이 {self.patience}에폭 연속 개선되지 않음"
        return is_best

    def _score_reached(self, f1, val_loss, correct, total):
        """점수 기준 조기 종료: 켜져 있고 최소 에폭을 채웠고 설정한 조건이 모두 만족되면 사유 문장을, 아니면 None 을 돌려 준다."""
        s = self.score_stop
        if not s or not s.get("enabled") or self.epoch < (s.get("min_epochs") or 1):
            return None
        reasons = []
        if s.get("min_accuracy") is not None:
            if not total or correct / total < s["min_accuracy"]:
                return None
            reasons.append(f"정답 {correct}/{total}" + ("(만점)" if correct == total else ""))
        if s.get("min_macro_f1") is not None:
            if f1 < s["min_macro_f1"]:
                return None
            reasons.append(f"Macro-F1 {f1:.4f}")
        if s.get("max_val_loss") is not None:
            if val_loss > s["max_val_loss"]:
                return None
            reasons.append(f"val_loss {val_loss:.4f} <= {s['max_val_loss']}")
        return ("점수 기준 도달: " + ", ".join(reasons)) if reasons else None

    @property
    def should_stop(self):
        return self.stop_reason is not None


def score_stop_settings():
    """config 의 점수 기준 종료 설정을 dict 로 읽는다."""
    tc = cfg.TRAINING_CONFIG
    return {"enabled": bool(tc.get("stop_when_scores_reached")), "min_accuracy": tc.get("stop_min_accuracy"),
            "min_macro_f1": tc.get("stop_min_macro_f1"), "max_val_loss": tc.get("stop_max_val_loss"),
            "min_epochs": tc.get("stop_min_epochs", 1)}


def describe_score_stop(settings):
    """점수 기준 종료 설정을 사람이 읽는 문장으로 만든다."""
    if not settings["enabled"]:
        return "끔"
    parts = []
    if settings["min_accuracy"] is not None:
        parts.append(f"정답률 {settings['min_accuracy']:.0%} 이상" + (" (만점)" if settings["min_accuracy"] >= 1 else ""))
    if settings["min_macro_f1"] is not None:
        parts.append(f"Macro-F1 {settings['min_macro_f1']} 이상")
    if settings["max_val_loss"] is not None:
        parts.append(f"val_loss {settings['max_val_loss']} 이하")
    if not parts:
        return "조건 없음"
    return " & ".join(parts) + f" ({settings['min_epochs']}에폭 이후)"


def resolve_device(name=None):
    import torch
    name = name or cfg.DEVICE
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 를 사용할 수 없습니다. baseline_bert_config.py 의 DEVICE 를 확인하세요.")
    return torch.device(name)


def free_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def encode_text(text, tokenizer, max_length, truncation="right", head_tokens=None):
    """문장 하나를 모델 입력(input_ids 등)으로 바꾼다. 반환: (입력 dict, 잘렸는지 여부)

    학습·검증·test 평가(그리고 나중의 서비스)가 모두 이 함수를 써야 같은 방식으로 입력된다.
      right     : 앞쪽 (max_length - 특수토큰) 개만 남기고 뒤를 자른다.
      head_tail : 앞 head_tokens 개 + 뒤쪽 나머지를 남기고 가운데를 자른다.
    """
    # 긴 문장은 아래에서 직접 자르므로, 토크나이저의 "길이 초과" 경고는 끈다. (속성이 없는 버전이면 그냥 넘어감)
    warned = getattr(tokenizer, "deprecation_warnings", None)
    if isinstance(warned, dict):
        warned["sequence-length-is-longer-than-the-specified-maximum"] = True
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)   # [CLS], [SEP] 자리는 제외
    truncated = len(ids) > budget
    if truncated:
        if truncation == "head_tail":
            head = min(head_tokens if head_tokens is not None else budget // 2, budget)
            tail = budget - head
            ids = ids[:head] + (ids[-tail:] if tail > 0 else [])
        elif truncation == "right":
            ids = ids[:budget]
        else:
            raise ValueError(f"truncation 은 'right' 또는 'head_tail' 이어야 합니다: {truncation}")
    input_ids = tokenizer.build_inputs_with_special_tokens(ids)
    item = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
    if "token_type_ids" in tokenizer.model_input_names:
        item["token_type_ids"] = tokenizer.create_token_type_ids_from_sequences(ids)
    return item, truncated


def encode_rows(rows, tokenizer, label2id, max_length, truncation="right", head_tokens=None):
    """행 목록을 입력으로 바꾼다. padding 은 DataLoader 가 배치로 묶을 때 맞춘다.
    반환: (items, 잘린 행 수)"""
    items, cut = [], 0
    for row in rows:
        item, truncated = encode_text(row[cfg.TEXT_COLUMN], tokenizer, max_length, truncation, head_tokens)
        item["labels"] = label2id[row[cfg.LABEL_COLUMN]]
        items.append(item)
        cut += truncated
    return items, cut


def validate_model(model, loader, device, labels):
    """검증: 점수는 계산하지만 가중치는 바꾸지 않는다."""
    import torch
    model.eval()
    predictions, probabilities, actual, total_loss, count = [], [], [], 0.0, 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            size = batch["labels"].shape[0]
            total_loss += output.loss.item() * size
            count += size
            predictions.extend(output.logits.argmax(-1).cpu().tolist())
            probabilities.extend(torch.softmax(output.logits.float(), dim=-1).cpu().tolist())
            actual.extend(batch["labels"].cpu().tolist())
    seconds = time.perf_counter() - started
    return (classification_scores(actual, predictions, labels), total_loss / count, predictions,
            probabilities, seconds)


def train_transformer_run(model_key, topic, dataset, learning_rate, data):
    """모델 1개 x Learning Rate 1개 = run 1개. 새 날짜 폴더를 만들어 학습하고 best_model 과 비교한다."""
    run_dir = begin_run(model_key, topic, dataset)
    try:
        return _fit_transformer(run_dir, model_key, topic, dataset, learning_rate, data)
    except BaseException:
        safe_delete(run_dir)    # 실패·중단(Ctrl+C)한 run 은 폴더째 지운다. (화면 출력은 logs/ 에 남아 있음)
        raise


def _fit_transformer(run_dir, model_key, topic, dataset, learning_rate, data):
    import numpy as np
    import torch
    import transformers
    from torch.utils.data import DataLoader
    from transformers.utils import logging as hf_logging
    from result_files import weights_info, write_predictions, write_result_files
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    tc = cfg.TRAINING_CONFIG
    model_id = cfg.PRETRAINED_MODELS[model_key]["model_id"]
    labels = data["labels"]
    label2id = {label: index for index, label in enumerate(labels)}
    run_name = run_dir.name
    weights_dir = run_dir / WEIGHTS_TMP     # best 에폭의 가중치를 잠시 두는 곳. 끝나면 best_model 로 옮기거나 지운다.
    lines = []   # 이 run 의 로그(화면에 찍히는 줄). 학습이 끝나면 run 폴더에 train.log 로 저장한다.

    def say(message):
        print(message)
        lines.append(message)

    device = resolve_device()
    use_cuda = device.type == "cuda"
    epochs, accumulation = tc["epochs"], tc["gradient_accumulation_steps"]
    if min(epochs, accumulation, tc["batch_size"], tc["max_length"]) < 1 or not 0 < learning_rate < 1:
        raise ValueError("에폭·배치·길이는 1 이상, Learning Rate 는 0과 1 사이여야 합니다.")
    if tc["truncation"] not in ("right", "head_tail"):
        raise ValueError(f"TRAINING_CONFIG['truncation'] 은 'right' 또는 'head_tail' 이어야 합니다: {tc['truncation']}")
    if tc["max_length"] > 512:
        raise ValueError("max_length 는 이 모델들의 한계인 512 이하여야 합니다.")

    # 같은 설정이면 같은 결과가 나오도록 난수를 고정한다. (분류층 초기값도 모델 로딩 전에 고정)
    seed = tc["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=tc["local_files_only"])
    # "Some weights ... newly initialized" 안내는 분류층을 새로 만드는 정상 동작이라 학습할 때는 숨긴다.
    # (저장한 모델을 다시 불러오는 test 평가에서는 숨기지 않는다. 거기서 나오면 가중치가 제대로 불러와지지 않았다는 뜻이다)
    previous_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id,
            local_files_only=tc["local_files_only"])
    finally:
        hf_logging.set_verbosity(previous_verbosity)
    for parameter in model.parameters():       # 본체(인코더)와 분류층을 모두 학습한다.
        parameter.requires_grad_(True)
    if use_cuda and tc["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    train_items, train_cut = encode_rows(data["train"], tokenizer, label2id, tc["max_length"],
                                         tc["truncation"], tc["head_tokens"])
    validation_items, validation_cut = encode_rows(data["validation"], tokenizer, label2id,
                                                   tc["max_length"], tc["truncation"], tc["head_tokens"])
    train_loader = DataLoader(train_items, batch_size=tc["batch_size"], shuffle=True,
                              collate_fn=collator, num_workers=0,
                              generator=torch.Generator().manual_seed(seed))
    validation_loader = DataLoader(validation_items, batch_size=tc["batch_size"], shuffle=False,
                                   collate_fn=collator, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=tc["weight_decay"])
    total_updates = epochs * math.ceil(len(train_loader) / accumulation)
    warmup_updates = max(1, int(total_updates * tc["warmup_ratio"]))

    def lr_factor(step):   # 처음 10% 는 서서히 올리고, 그 뒤 0 까지 직선으로 줄인다.
        if step < warmup_updates:
            return (step + 1) / warmup_updates
        return max(0.0, (total_updates - step) / max(1, total_updates - warmup_updates))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    dtype = torch.bfloat16 if (not use_cuda or torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and dtype == torch.float16)

    patience, target = tc["early_stopping_patience"], tc["target_macro_f1"]
    score_stop = score_stop_settings()
    say(f"  [{model_key}/{run_name}] 설정: {topic}/{dataset} | train {len(data['train'])}행 / "
        f"validation {len(data['validation'])}행 | LR {learning_rate:g} | 에폭 최대 {epochs} | "
        f"배치 {tc['batch_size']}x{accumulation} (실질 {tc['batch_size'] * accumulation}) | "
        f"가중치 업데이트 최대 {total_updates}회 | {device}"
        f"{' ' + str(dtype).replace('torch.', '') if use_cuda else ''}")
    say(f"  [{model_key}/{run_name}] 중단 조건: "
        f"{f'Macro-F1 {patience}에폭 연속 개선 없음' if patience else '조기 종료 끔'}"
        f"{f' / Macro-F1 {target} 이상' if target is not None else ''}")
    say(f"  [{model_key}/{run_name}] 점수 기준 종료: {describe_score_stop(score_stop)}")
    head_note = f"(앞 {tc['head_tokens']})" if tc["truncation"] == "head_tail" else ""
    say(f"  [{model_key}/{run_name}] 입력: max_length {tc['max_length']} {tc['truncation']}{head_note} | "
        f"max_length 를 넘어 잘린 행: train {train_cut}/{len(train_items)}, "
        f"validation {validation_cut}/{len(validation_items)}")

    # 이 run 의 학습 조건 기록. (Hugging Face 가 가중치와 함께 저장하는 config.json 과 이름이 겹치지 않게 run_config.json)
    save_json(run_dir / "run_config.json", {
        "topic": topic, "dataset": dataset, "model_key": model_key, "model_id": model_id,
        "run_name": run_name, "learning_rate": learning_rate, "epochs_max": epochs,
        "early_stopping_patience": patience, "target_macro_f1": target, "score_stop": score_stop,
        "batch_size": tc["batch_size"], "gradient_accumulation_steps": accumulation,
        "max_length": tc["max_length"], "truncation": tc["truncation"], "head_tokens": tc["head_tokens"],
        "truncated_rows": {"train": train_cut, "validation": validation_cut},
        "weight_decay": tc["weight_decay"],
        "warmup_ratio": tc["warmup_ratio"], "seed": seed, "labels": labels,
        "train_rows": len(data["train"]), "validation_rows": len(data["validation"]),
        "train_csv": relative_paths(dataset_paths(topic, dataset)["train"]),
        "validation_csv": relative_paths(dataset_paths(topic, dataset)["validation"]),
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
        "precision": str(dtype).replace("torch.", "") if use_cuda else "float32",
        "gradient_checkpointing": bool(use_cuda and tc["gradient_checkpointing"]),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__})

    tracker = EpochTracker(patience, target, score_stop)
    best = None
    updates = 0
    epochs_run = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_train_loss, count = 0.0, 0
        for index, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            window_start = (index // accumulation) * accumulation
            window_size = min(accumulation, len(train_loader) - window_start)   # 마지막 구간은 짧을 수 있다
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_cuda):
                output = model(**batch)
                loss = output.loss / window_size
            scaler.scale(loss).backward()
            size = batch["labels"].shape[0]
            total_train_loss += output.loss.item() * size
            count += size
            if (index + 1) % accumulation == 0 or index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["max_grad_norm"])
                scaler.step(optimizer)           # 가중치가 실제로 바뀌는 지점
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1

        scores, val_loss, predictions, probabilities, val_seconds = validate_model(
            model, validation_loader, device, labels)
        elapsed = time.perf_counter() - started
        correct, total, _ = correct_counts(scores, labels)
        is_best = tracker.update(scores["macro_f1"], val_loss, correct, total)
        record = {"epoch": epoch, "train_loss": total_train_loss / count, "validation_loss": val_loss,
                  "validation_accuracy": scores["accuracy"], "validation_macro_f1": scores["macro_f1"],
                  "optimizer_steps": updates, "elapsed_seconds": elapsed}
        with (run_dir / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        say(f"  [{model_key}/{run_name}] epoch {epoch}/{epochs}  train_loss={record['train_loss']:.4f}  "
            f"val_loss={val_loss:.4f}  val_macro_f1={scores['macro_f1']:.4f}  "
            f"정답 {correct}/{total} ({correct / total:.1%})"
            f"{'  <- best' if is_best else ''}  ({elapsed:.0f}s)")

        # best 에폭(Macro-F1 최고, 동점이면 val_loss 최저)의 가중치만 임시 폴더에 저장한다.
        if is_best:
            weights_dir.mkdir(exist_ok=True)
            model.save_pretrained(weights_dir, safe_serialization=True)
            tokenizer.save_pretrained(weights_dir)
            best = {"epoch": epoch, "scores": scores, "loss": val_loss, "predictions": predictions,
                    "probs": probabilities, "val_seconds": val_seconds}
        if tracker.should_stop:
            say(f"  [{model_key}/{run_name}] 조기 종료: epoch {epoch}/{epochs} - {tracker.stop_reason}")
            break

    train_seconds = time.perf_counter() - started
    stop_reason = tracker.stop_reason or f"최대 에폭({epochs}) 도달"
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if use_cuda else None
    num_parameters, model_size_mb = weights_info(weights_dir)   # 저장된 가중치 파일에서 파라미터 수·크기를 읽는다
    summary = make_summary(model_key, model_id, run_name, learning_rate, best["scores"], best["loss"],
                           best["epoch"], train_seconds, best["val_seconds"], peak_mb,
                           len(data["train"]), len(data["validation"]),
                           extra={"epochs_run": epochs_run, "epochs_max": epochs, "stop_reason": stop_reason,
                                  "num_parameters": num_parameters, "model_size_mb": model_size_mb})
    save_json(run_dir / "validation_metrics.json", best["scores"])
    save_json(run_dir / "training_summary.json", summary)
    save_errors(run_dir / "validation_errors.csv", data["validation"], best["predictions"], labels)
    write_predictions(run_dir / "validation_predictions.csv", data["validation"], best["predictions"],
                      best["probs"], labels)   # 모든 문장의 예측 (팀원과 비교용)
    gpu_note = f", GPU 최대 {peak_mb:.0f}MB" if peak_mb is not None else ""
    say(f"  [{model_key}/{run_name}] 완료: best epoch {best['epoch']} (총 {epochs_run}에폭 학습), "
        f"val_macro_f1={best['scores']['macro_f1']:.4f}, {train_seconds:.0f}초{gpu_note}")
    say(f"  [{model_key}/{run_name}] validation 정답(best epoch {best['epoch']}): "
        f"{format_correct(best['scores'], labels)}")
    (run_dir / "train.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_result_files(run_dir, weights_dir)   # main.json / sub.json (best 이면 best_model 로 함께 복사됨)

    finalize_run(model_key, topic, dataset, run_dir, summary, weights_dir)   # best_model 과 비교해서 교체/유지
    mark_complete(run_dir)   # 여기까지 와야 '완료'로 인정한다.
    return summary


def train_model_runs(model_key, topic, dataset):
    """한 모델의 Learning Rate 를 차례로 학습한다. (값마다 날짜 폴더가 하나씩 생기고 best_model 과 비교한다)

    한 run 이 실패(GPU 메모리 부족 등)해도 기록만 하고 다음 run 으로 넘어간다.
    반환값: 실패한 run 이름 목록
    """
    model_id = cfg.PRETRAINED_MODELS[model_key]["model_id"]
    learning_rates = model_learning_rates(model_key)
    if len(set(learning_rates)) != len(learning_rates):
        raise ValueError(f"LEARNING_RATES['{model_key}'] 에 같은 값이 중복되어 있습니다: {learning_rates}")
    print(f"\n=== {topic}/{dataset} · {model_key} ({model_id}) · LR {learning_rates} ===")
    data = load_data(topic, dataset)
    if resolve_device().type == "cpu":
        print("  [주의] GPU(CUDA)를 찾지 못해 CPU 로 학습합니다. 매우 느릴 수 있습니다.")

    failures = []
    for learning_rate in learning_rates:
        try:
            train_transformer_run(model_key, topic, dataset, learning_rate, data)
        except Exception:
            print(f"  [{model_key}] LR {learning_rate:g} 학습 실패:")
            traceback.print_exc()
            failures.append(f"{model_key}/LR{learning_rate:g}")
        free_memory()
    # 새 run 이 best_model 을 못 이겨 옛 best 가 그대로 남았다면, 그 폴더에 main/sub/문장별 예측이 없을 수 있어 채운다.
    from result_files import ensure_best_result_files
    ensure_best_result_files(model_key, topic, dataset)
    return failures


def run_cli(description, trainer):
    """train_*.py 를 직접 실행할 때 쓰는 공통 진입점. --topic / --dataset 으로 좁힐 수 있다."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--topic", choices=cfg.TOPICS, help="생략하면 ON 인 주제 전부")
    parser.add_argument("--dataset", choices=cfg.DATASET_NAMES, help="생략하면 ON 인 데이터셋 전부")
    args = parser.parse_args()
    pairs = ready_pairs(args.topic, args.dataset)
    if not pairs:
        print("실행할 데이터셋이 없습니다. baseline_bert_config.py 의 DATASET_SWITCH 를 확인하세요.")
        return
    with session_log(Path(sys.argv[0]).stem):   # 예: logs/train_bert_20261012_153000.log
        print_settings()
        for topic, dataset in pairs:
            trainer(topic, dataset)


# ═════════════════════════ 6. 모델 추천 (best_model 의 validation 결과만 사용) ═════════════════════════
def rank_key(summary):
    """모델끼리 비교: validation Macro-F1 높은 순 -> 동점이면 validation loss 낮은 순 -> 학습 시간 짧은 순."""
    return (-round(summary["validation_macro_f1"], 6), summary["validation_loss"], summary["train_seconds"])


def _fmt(value, spec):
    return "-" if value is None else format(value, spec)


def _print_table(rows):
    print(f"  {'model':<14}{'best run':<17}{'best_ep':>8}{'val_f1':>9}{'val_acc':>9}{'val_ok':>10}{'val_loss':>10}"
          f"{'train_s':>9}{'gpu_mb':>8}")
    for row in rows:
        name = row["model_key"] + ("(ref)" if row.get("reference") else "")
        ok = f"{round(row['validation_accuracy'] * row['validation_rows'])}/{row['validation_rows']}"
        print(f"  {name:<14}{row['run_name']:<17}{_fmt(row['best_epoch'], 'd'):>8}"
              f"{row['validation_macro_f1']:>9.4f}{row['validation_accuracy']:>9.4f}{ok:>10}"
              f"{_fmt(row['validation_loss'], '.4f'):>10}{row['train_seconds']:>9.0f}"
              f"{_fmt(row['peak_gpu_memory_mb'], '.0f'):>8}")


def recommend_model(topic, dataset):
    """bert / roberta / koelectra 의 best_model 중 validation 결과가 가장 좋은 모델 1개를 추천한다. test 는 보지 않는다."""
    if is_recommend_excluded(dataset):
        print(f"\n=== {topic}/{dataset} · 모델 추천 제외 (서버용 데이터셋이라 추천 대상이 아닙니다) ===")
        return None
    print(f"\n=== {topic}/{dataset} · 모델 추천 (각 모델 best_model 의 validation 결과만 사용) ===")
    candidates = []
    for key in TRANSFORMER_KEYS:
        info = read_best_info(key, topic, dataset)
        if info is None:
            print(f"  [주의] {key}: best_model 이 없어(학습 전) 비교에서 제외합니다.")
        else:
            candidates.append(info["best"])
    if not candidates:
        print("  추천할 결과가 없습니다. 먼저 1번(학습)을 실행하세요.")
        return None

    ranked = sorted(candidates, key=rank_key)
    best = ranked[0]
    baseline_info = read_best_info(BASELINE_KEY, topic, dataset)
    reference = {**baseline_info["best"], "reference": True} if baseline_info else None
    _print_table(ranked + ([reference] if reference else []))

    if len(ranked) == 1:
        reason = f"완료된 후보가 {best['model_key']} 하나뿐입니다."
    else:
        second = ranked[1]
        if abs(best["validation_macro_f1"] - second["validation_macro_f1"]) < 1e-6:
            reason = (f"validation Macro-F1 이 {best['validation_macro_f1']:.4f}로 {second['model_key']}와 동점이라 "
                      f"validation loss({best['validation_loss']:.4f} vs {second['validation_loss']:.4f})로 결정했습니다.")
        else:
            gap = round((best["validation_accuracy"] - second["validation_accuracy"]) * best["validation_rows"])
            reason = (f"validation Macro-F1 {best['validation_macro_f1']:.4f}로 1위 "
                      f"({second['model_key']} {second['validation_macro_f1']:.4f}, 정확히 맞힌 문장 약 {gap}건 차이 / "
                      f"validation {best['validation_rows']}건).")
    print(f"\n  >> 추천: {best['model_key']} ({best['model_id']}), best_model = {best['run_name']}")
    print(f"     이유: {reason}")
    notes = []
    if len(candidates) < len(TRANSFORMER_KEYS):
        notes.append(f"{len(candidates)}개 모델만 비교했습니다. 나머지 모델은 아직 학습 결과가 없습니다.")
    if reference and reference["validation_macro_f1"] >= best["validation_macro_f1"] - 1e-6:
        notes.append(f"TF-IDF baseline(val_f1 {reference['validation_macro_f1']:.4f})이 같거나 더 높습니다. "
                     "사전학습 모델이 꼭 이겨야 하는 것은 아니니 이 사실을 보고서에 그대로 기록하세요.")
    tested = [k for k in TRANSFORMER_KEYS if (best_dir(k, topic, dataset) / "test_metrics.json").exists()]
    if tested and best["model_key"] not in tested:
        notes.append(f"이미 {tested} 의 test 평가가 끝나 있습니다. 추천이 바뀌었다면 test 이후에 모델을 다시 "
                     "고른 셈이므로 기존 test 결과와 추가 실험을 구분해 기록하세요.")
    for note in notes:
        print(f"     [참고] {note}")

    result_dir = dataset_result_dir(topic, dataset)
    save_json(result_dir / "recommendation.json", {
        "topic": topic, "dataset": dataset, "recommended_model": best["model_key"],
        "model_id": best["model_id"], "run_name": best["run_name"], "learning_rate": best["learning_rate"],
        "selection_rule": "각 모델 best_model 의 validation Macro-F1 (동점: validation loss, 그다음 학습 시간). "
                          "test 점수는 사용하지 않음.",
        "reason": reason, "notes": notes, "validation_macro_f1": best["validation_macro_f1"],
        "compared_models": [c["model_key"] for c in ranked],
        "baseline_reference_validation_macro_f1": reference["validation_macro_f1"] if reference else None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"     저장: {result_dir / 'recommendation.json'}")
    return best


# ═════════════════════════ 7. test 평가 · 초기화 ═════════════════════════
def predict_with_model_dir(model_path, rows, labels, device, batch_size=16):
    """best_model 폴더의 가중치를 불러와 예측한다. (학습 때와 같은 라벨 순서·입력 방식 사용)"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    model_path = Path(model_path)
    if not (model_path / "model.safetensors").exists():
        raise FileNotFoundError(f"저장된 가중치가 없습니다: {model_path / 'model.safetensors'}")
    run_config = load_json(model_path / "run_config.json")
    # 학습 때와 같은 방식으로 입력해야 한다. (truncation 기록이 없는 예전 run 은 앞쪽만 남기는 'right')
    max_length = run_config["max_length"]
    truncation = run_config.get("truncation", "right")
    head_tokens = run_config.get("head_tokens")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, use_safetensors=True).to(device).eval()
    if [model.config.id2label[i] for i in range(model.config.num_labels)] != labels:
        raise ValueError("학습 때의 라벨 순서와 현재 데이터의 라벨 순서가 다릅니다.")
    # test 평가는 학습 때와 같은 방식으로 입력한다. 반환: (예측, 라벨별 확률, 추론 시간(초), 입력이 잘린 행 수)
    encoded = [encode_text(row[cfg.TEXT_COLUMN], tokenizer, max_length, truncation, head_tokens) for row in rows]
    items_all = [item for item, _ in encoded]
    truncated = sum(1 for _, cut in encoded if cut)
    predictions, probabilities = [], []
    started = time.perf_counter()    # 토큰화는 제외하고 모델 계산만 잰다. (validation 추론 시간과 같은 기준)
    for start in range(0, len(items_all), batch_size):
        features = {key: value.to(device)
                    for key, value in collator(items_all[start:start + batch_size]).items()}
        with torch.inference_mode():
            logits = model(**features).logits
        predictions.extend(logits.argmax(-1).cpu().tolist())
        probabilities.extend(torch.softmax(logits.float(), dim=-1).cpu().tolist())
    seconds = time.perf_counter() - started
    del model
    free_memory()
    return predictions, probabilities, seconds, truncated


def evaluate_final(topic, dataset, mode="recommended"):
    """test 최종 평가. 본체는 test_eval.py 에 있다.

    mode: 'recommended' = 추천 모델 1개(+ baseline), 'all' = baseline + 3개 모델 모두(비교용).
    """
    from test_eval import evaluate_test
    return evaluate_test(topic, dataset, mode)


def reset_targets(topic, dataset, scope):
    """초기화로 지워질 경로. scope: 'all' 또는 모델 키('baseline', 'bert', 'roberta', 'koelectra').

    CSV 데이터와 logs/ 폴더는 절대 포함되지 않는다. 모델 하나만 지우면 그 모델이 들어 있던 모델 간 비교/추천
    결과도 낡은 값이 되므로 함께 지운다.
    """
    result_dir = dataset_result_dir(topic, dataset)
    if scope == "all":
        paths = [result_dir]
    else:
        paths = [model_dir(scope, topic, dataset), result_dir / "comparison", result_dir / "recommendation.json"]
    return [path for path in paths if path.exists()]


def reset_results(topic, dataset, scope):
    deleted = []
    for path in reset_targets(topic, dataset, scope):
        safe_delete(path)
        deleted.append(path)
    return deleted
