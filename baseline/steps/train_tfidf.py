"""TF-IDF + 로지스틱 회귀 baseline 학습 (baseline/steps/train_tfidf.py).

train 으로 TF-IDF 어휘/IDF 와 분류 가중치를 학습하고, validation 으로 점수를 기록한다.
결과: artifacts/<topic>/<dataset>/baseline/<날짜 폴더>/ 에 저장하고, 지금까지의 best_model 보다 좋으면
      artifacts/<topic>/<dataset>/baseline/best_model/ 을 교체한다. (모델 파일은 baseline.joblib)
      main.json / sub.json / validation_predictions.csv 도 함께 만든다. (result_files.py)

직접 실행 (프로젝트 루트 project_2 에서):
    uv run python baseline/steps/train_tfidf.py                                   # ON 인 데이터셋 전부
    uv run python baseline/steps/train_tfidf.py --topic inquiries --dataset basic
보통은 run_menu.py 의 1번 메뉴에서 함께 실행됩니다.
"""
import sys
import time
from pathlib import Path

# 설정 파일(baseline_bert_config.py)은 프로젝트 루트(project_2)에 있다. (이 파일에서 두 단계 위)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import baseline_bert_config as cfg  # noqa: E402
import common  # noqa: E402
import joblib  # noqa: E402
import sklearn  # noqa: E402
from result_files import (  # noqa: E402
    ensure_best_result_files,
    weights_info,
    write_predictions,
    write_result_files,
)
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

KEY = common.BASELINE_KEY


def train(topic, dataset):
    """반환값: 실패 목록(성공이면 빈 리스트). run_menu 가 다른 모델 trainer 와 같은 방식으로 다룬다."""
    print(f"\n=== {topic}/{dataset} · baseline (TF-IDF + 로지스틱 회귀) ===")
    data = common.load_data(topic, dataset)
    labels = data["labels"]
    label2id = {label: index for index, label in enumerate(labels)}
    x_train = [row[cfg.TEXT_COLUMN] for row in data["train"]]
    y_train = [label2id[row[cfg.LABEL_COLUMN]] for row in data["train"]]
    x_val = [row[cfg.TEXT_COLUMN] for row in data["validation"]]
    y_val = [label2id[row[cfg.LABEL_COLUMN]] for row in data["validation"]]

    run_dir = common.begin_run(KEY, topic, dataset)      # 새 날짜 폴더
    try:
        run_name = run_dir.name
        # fit 은 train 에서만 한다. TF-IDF 의 어휘와 IDF 도 이 단계에서 train 으로만 계산된다.
        model = make_pipeline(TfidfVectorizer(**cfg.TFIDF_CONFIG),
                              LogisticRegression(**cfg.LOGISTIC_REGRESSION_CONFIG))
        started = time.perf_counter()
        model.fit(x_train, y_train)
        train_seconds = time.perf_counter() - started

        started = time.perf_counter()
        predictions = model.predict(x_val).tolist()
        inference_seconds = time.perf_counter() - started
        probabilities = model.predict_proba(x_val).tolist()      # 문장별 예측 파일에 쓰는 라벨별 확률
        scores = common.classification_scores(y_val, predictions, labels)

        weights_dir = run_dir / common.WEIGHTS_TMP       # 모델 파일은 best 일 때만 best_model 로 옮긴다
        weights_dir.mkdir()
        joblib.dump(model, weights_dir / "baseline.joblib")
        num_parameters, model_size_mb = weights_info(weights_dir)

        common.save_json(run_dir / "run_config.json", {
            "topic": topic, "dataset": dataset, "model_key": KEY, "run_name": run_name, "labels": labels,
            "tfidf": cfg.TFIDF_CONFIG, "logistic_regression": cfg.LOGISTIC_REGRESSION_CONFIG,
            "seed": cfg.SEED, "train_rows": len(x_train), "validation_rows": len(x_val),
            "train_csv": common.relative_paths(common.dataset_paths(topic, dataset)["train"]),
            "validation_csv": common.relative_paths(common.dataset_paths(topic, dataset)["validation"]),
            "sklearn": sklearn.__version__})
        summary = common.make_summary(
            KEY, "TF-IDF(char n-gram) + LogisticRegression", run_name, None, scores, None, None,
            train_seconds, inference_seconds, None, len(x_train), len(x_val),
            extra={"num_parameters": num_parameters, "model_size_mb": model_size_mb})
        common.save_json(run_dir / "validation_metrics.json", scores)
        common.save_json(run_dir / "training_summary.json", summary)
        common.save_errors(run_dir / "validation_errors.csv", data["validation"], predictions, labels)
        write_predictions(run_dir / "validation_predictions.csv", data["validation"], predictions,
                          probabilities, labels)            # 모든 문장의 예측 (팀원과 비교용)

        lines = [
            f"  [{KEY}/{run_name}] 설정: {topic}/{dataset} | train {len(x_train)}행 / validation {len(x_val)}행 | "
            f"analyzer {cfg.TFIDF_CONFIG['analyzer']} ngram {cfg.TFIDF_CONFIG['ngram_range']} "
            f"min_df {cfg.TFIDF_CONFIG['min_df']} sublinear_tf {cfg.TFIDF_CONFIG['sublinear_tf']} | "
            f"로지스틱 회귀 max_iter {cfg.LOGISTIC_REGRESSION_CONFIG['max_iter']} "
            f"class_weight {cfg.LOGISTIC_REGRESSION_CONFIG['class_weight']}",
            f"  [{KEY}/{run_name}] 완료: validation macro F1 {scores['macro_f1']:.4f} / "
            f"accuracy {scores['accuracy']:.4f} ({train_seconds:.1f}초)",
            f"  [{KEY}/{run_name}] validation 정답: {common.format_correct(scores, labels)}",
        ]
        for line in lines:
            print(line)
        (run_dir / "train.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_result_files(run_dir, weights_dir)          # main.json / sub.json

        common.finalize_run(KEY, topic, dataset, run_dir, summary, weights_dir)   # best_model 과 비교
        common.mark_complete(run_dir)
    except BaseException:
        common.safe_delete(run_dir)      # 실패·중단한 run 폴더는 지운다
        raise
    print(f"  저장: {run_dir}")
    ensure_best_result_files(KEY, topic, dataset)   # 새 run 이 best 를 못 이겼을 때 옛 best 에 빠진 파일을 채운다
    return []


def main():
    common.run_cli("TF-IDF + 로지스틱 회귀 baseline 학습", train)


if __name__ == "__main__":
    main()
