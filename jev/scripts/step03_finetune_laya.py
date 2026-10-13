"""
laya 모델의 파인튜닝이 가능한지 확인한다. (완성형 학습 스크립트가 아닌 동작 확인용)

데이터 확인 -> laya.train.finetune으로 학습/보정/저장 -> 저장된 모델을 다시 불러 validation 예측
결과는 jev/tmp/<topic>-<데이터 종류>-<run_name>/ 에 저장한다.

루트 디렉토리 기준 실행 코드
uv run python ./jev/scripts/step03_finetune_laya.py --topic inquiries
uv run python ./jev/scripts/step03_finetune_laya.py --topic inquiries --limit 60 --epochs 1

Windows에서는 저장된 토크나이저 설정을 laya가 cp949로 읽으려다 경고를 낼 수 있으므로
PYTHONUTF8=1 환경 변수를 켜고 실행하는 것을 권장한다.
"""

import argparse
import json
import random
import time
from pathlib import Path

import laya
from laya.train import TrainConfig, finetune
from sklearn.metrics import accuracy_score, f1_score
from util.common_args import common_parser
from util.path_extractor import extract_path

from settings import MODEL_PATH, TMP_DIR
from step01_read_data import read_labels, read_rows
from step02_check_data import check_splits, group_rule_for

QID = "label"
PREVIEW_ROWS = 3
SEED = 42


# 빠른 확인을 위해 앞쪽 일부 행만 jsonl로 다시 쓴다. finetune은 파일 경로를 입력으로 받는다.
def write_subset(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


# 저장된 체크포인트를 다시 불러 validation을 예측하고 점수를 계산한다.
def evaluate_saved_model(output_dir: Path, rows: list[dict], labels: list[str],
                         batch_size: int) -> dict:
    agent = laya.load(str(output_dir))
    questions = rows[0]["questions"]
    results = agent.predict_batch([row["state"] for row in rows], questions, batch_size=batch_size)

    predictions = [result["answers"][QID]["choice"] for result in results]
    actual = [row["expected"][QID] for row in rows]
    for row, prediction in list(zip(rows, predictions))[:PREVIEW_ROWS]:
        print(f"\n문장: {row['state'][:80]}")
        print(f"정답: {row['expected'][QID]} / 예측: {prediction}")

    return {
        "rows": len(rows),
        "accuracy": accuracy_score(actual, predictions),
        "macro_f1": f1_score(actual, predictions, labels=labels, average="macro", zero_division=0),
    }


def main():
    ap = argparse.ArgumentParser(parents=[common_parser()])
    ap.add_argument("--run-name", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512, help="state 토큰 길이 제한")
    ap.add_argument("--limit", type=int, default=None, help="train/validation에서 N행만 무작위로 사용")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    # 1. 데이터를 읽고 train과 validation을 확인한다. test는 읽지 않는다.
    data_path = extract_path(a.topic, a.advanced, a.extended)
    labels = read_labels(data_path / "labels.json")
    splits = {name: read_rows(data_path / f"{name}.jsonl") for name in ("train", "validation")}
    check_splits(splits, labels, group_rule_for(data_path.name))

    output_dir = TMP_DIR / f"{a.topic}-{data_path.name}-{a.run_name}"
    if output_dir.exists():
        raise FileExistsError(f"기존 실험을 덮어쓰지 않습니다. --run-name을 바꾸세요: {output_dir}")

    # 2. --limit이 있으면 일부 행만 따로 저장해서 학습한다.
    train_file, validation_file = data_path / "train.jsonl", data_path / "validation.jsonl"
    if a.limit:
        # 파일이 라벨 순으로 정렬되어 있을 수 있어 섞은 뒤 자른다.
        splits = {name: random.Random(SEED).sample(rows, min(a.limit, len(rows)))
                  for name, rows in splits.items()}
        train_file = write_subset(splits["train"], output_dir / "data" / "train.jsonl")
        validation_file = write_subset(splits["validation"], output_dir / "data" / "validation.jsonl")
    print("데이터:", data_path, {name: len(rows) for name, rows in splits.items()})

    # 3. 학습 -> 온도 보정 -> 저장까지 laya SDK가 처리한다.
    #    validation은 학습 전/후 비교용 평가 데이터로만 사용된다.
    config = TrainConfig(
        epochs=a.epochs,
        micro_batch=a.micro_batch,
        grad_accum=a.grad_accum,
        max_len=a.max_len,
        eval_data=str(validation_file),
        seed=SEED,
    )
    started = time.perf_counter()
    summary = finetune(data=str(train_file), model_dir=str(MODEL_PATH),
                       output_dir=str(output_dir), config=config, device=a.device)
    elapsed = time.perf_counter() - started
    print("에폭별 loss:", summary["epoch_loss"])

    # 4. 저장된 모델을 다시 불러 예측이 되는지 확인한다.
    metrics = evaluate_saved_model(output_dir, splits["validation"], labels, a.micro_batch)
    metrics["train_seconds"] = round(elapsed, 1)
    print("\n" + json.dumps(metrics, ensure_ascii=False, indent=2))
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장 위치:", output_dir)


if __name__ == "__main__":
    main()
