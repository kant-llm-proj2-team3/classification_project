"""test 최종 평가 (baseline/steps/test_eval.py).

  recommended : 추천 모델 1개의 best_model 을 test 로 평가한다. baseline 점수도 같이 기록한다. (공식 점수)
  all         : baseline + bert + roberta + koelectra 를 각 best_model 로 모두 평가한다. (비교용)

어느 쪽이든 최종 모델은 validation 으로 정한 추천 모델이고, test 점수는 모델을 다시 고르는 데 쓰지 않는다.
그래서 결과에 "공식(official)" 표시를 남긴다: 추천 모델의 점수만 공식이고 나머지는 참고용이다.

저장되는 것 (각 모델의 best_model 폴더)
  test_metrics.json       : test 점수와 혼동행렬 (kit 의 test_metrics.json 과 같은 구조 + official 표시)
  test_errors.csv         : test 에서 틀린 문장
  test_predictions.csv    : test 모든 문장의 예측 (팀원과 비교용)
  main.json 의 test 칸    : 몇 문제 중 몇 개 정답, F1 등 (+ test_meta 에 공식 여부)
모델별 test 비교표는 화면에만 출력하고 파일로는 저장하지 않는다. (모아 볼 때는 artifacts/viz 의 all_main.csv 를 쓴다)
"""
import sys
import time
from pathlib import Path

# 설정 파일(baseline_bert_config.py)은 프로젝트 루트(project_2)에 있다. (이 파일에서 두 단계 위)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import baseline_bert_config as cfg  # noqa: E402
import common  # noqa: E402
import result_files  # noqa: E402

MODES = ("recommended", "all")


def _predict(model_key, best, rows, labels):
    """반환: (예측, 라벨별 확률, 추론 시간(초), 입력이 잘린 행 수 또는 None)"""
    if model_key != common.BASELINE_KEY:
        return common.predict_with_model_dir(best, rows, labels, common.resolve_device())
    import joblib
    model = joblib.load(best / "baseline.joblib")
    texts = [row[cfg.TEXT_COLUMN] for row in rows]
    started = time.perf_counter()
    probabilities = model.predict_proba(texts).tolist()
    seconds = time.perf_counter() - started
    return model.predict(texts).tolist(), probabilities, seconds, None


def write_test_comparison(topic, dataset):
    """각 모델 best_model 의 main.json 에 있는 test 점수를 모아 비교표를 화면에 출력한다. (이미 평가한 모델도 포함. 파일로는 저장하지 않는다)"""
    rows = []
    for key in common.ALL_MODEL_KEYS:
        path = common.best_dir(key, topic, dataset) / "main.json"
        if not path.exists():
            continue
        main = common.load_json(path)
        test = main.get("test")
        if not test:
            continue
        rows.append({"model_name": key, "run_id": main["run_id"], "test_n": test["n"],
                     "test_correct": test["correct"], "test_accuracy": test["accuracy"],
                     "test_macro_f1": test["macro_f1"], "validation_macro_f1": main["validation"]["macro_f1"],
                     "official": (main.get("test_meta") or {}).get("official"),
                     "test_inference_seconds": main["cost"].get("test_inference_seconds")})
    if rows:
        print(f"\n  [test 비교표] {topic}/{dataset}")
        print(f"  {'model':<12}{'best run':<17}{'test 정답':>12}{'test_f1':>9}{'validation_f1':>15}  구분")
        for row in rows:
            tag = "공식" if row["official"] else ("참고" if row["official"] is not None else "-")
            print(f"  {row['model_name']:<12}{row['run_id']:<17}"
                  f"{str(row['test_correct']) + '/' + str(row['test_n']):>12}{row['test_macro_f1']:>9.4f}"
                  f"{row['validation_macro_f1']:>15.4f}  {tag}")
    return rows


def evaluate_test(topic, dataset, mode="recommended"):
    """test 평가. mode 가 'recommended' 면 추천 모델 1개, 'all' 이면 4개 모델 모두. 각 모델은 한 번만 평가한다."""
    if mode not in MODES:
        raise ValueError(f"mode 는 {MODES} 중 하나여야 합니다: {mode}")
    excluded = common.is_recommend_excluded(dataset)     # 추천 제외 데이터셋(total_merge): 공식 모델이 없으므로 4개 모델을 모두 평가한다
    if excluded:
        mode, selected, selected_run = "all", None, None
    else:
        recommendation_path = common.dataset_result_dir(topic, dataset) / "recommendation.json"
        if not recommendation_path.exists():
            raise RuntimeError("추천 결과가 없습니다. 먼저 3번(모델 추천)을 실행하세요.")
        recommendation = common.load_json(recommendation_path)
        selected, selected_run = recommendation["recommended_model"], recommendation["run_name"]
        info = common.read_best_info(selected, topic, dataset)
        if info is None or info["best"]["run_name"] != selected_run:
            raise RuntimeError(f"추천 이후 {selected} 의 best_model 이 바뀌었습니다. 3번(모델 추천)을 다시 실행하세요.")
        selected_path = common.best_dir(selected, topic, dataset) / "test_metrics.json"
        if mode == "recommended" and selected_path.exists():
            raise FileExistsError(f"test 결과가 이미 있습니다. 덮어쓰지 않습니다: {selected_path}")

    if excluded:
        print(f"\n=== {topic}/{dataset} · test 평가: 4개 모델 모두 (모델 추천 제외 데이터셋이라 공식 모델 표시 없음) ===")
    else:
        title = "추천 모델 1개" if mode == "recommended" else "4개 모델 모두(비교용)"
        print(f"\n=== {topic}/{dataset} · test 평가: {title} · 공식 모델 = {selected} (best_model = {selected_run}) ===")
    data = common.load_data(topic, dataset, with_test=True)   # train/validation/test 분할 겹침까지 검사
    labels, test_rows = data["labels"], data["test"]
    label2id = {label: index for index, label in enumerate(labels)}
    actual = [label2id[row[cfg.LABEL_COLUMN]] for row in test_rows]

    outcomes = {}   # 모델 키 -> (best 요약, (예측, 확률, 시간, 잘린 행 수))
    for key in ([selected] if mode == "recommended" else list(common.TRANSFORMER_KEYS)):
        best = common.best_dir(key, topic, dataset)
        best_info = common.read_best_info(key, topic, dataset)
        if best_info is None:
            print(f"  [{key}] best_model 이 없어 건너뜁니다.")
            continue
        if (best / "test_metrics.json").exists():
            print(f"  [{key}] 이미 test 평가를 했습니다. 덮어쓰지 않고 건너뜁니다.")
            continue
        outcomes[key] = (best_info["best"], _predict(key, best, test_rows, labels))

    # baseline 은 항상 평가한다. (비교 기준선이며 모델 선택에 쓰이지 않는다. 결정적이라 다시 해도 같은 값)
    baseline_dir = common.best_dir(common.BASELINE_KEY, topic, dataset)
    baseline_info = common.read_best_info(common.BASELINE_KEY, topic, dataset)
    if baseline_info is not None and (baseline_dir / "baseline.joblib").exists():
        outcomes[common.BASELINE_KEY] = (baseline_info["best"],
                                         _predict(common.BASELINE_KEY, baseline_dir, test_rows, labels))
    else:
        print("  [주의] baseline 결과가 없어 baseline test 점수는 기록하지 않습니다.")

    scores = {key: common.classification_scores(actual, result[0], labels) for key, (_, result) in outcomes.items()}
    baseline_scores = scores.get(common.BASELINE_KEY)
    for key, (best_summary, (predictions, probabilities, seconds, truncated)) in outcomes.items():
        best = common.best_dir(key, topic, dataset)
        official = None if excluded else key == selected   # 추천 제외 데이터셋은 공식 여부를 표시하지 않는다(None)
        report = {"topic": topic, "dataset": dataset, "split": "test", "selected_model": selected,
                  "evaluated_model": key, "official": official, "selected_run": best_summary["run_name"],
                  "model_id": best_summary["model_id"], "classifier": scores[key], "baseline": baseline_scores}
        if baseline_scores is not None:
            report["baseline_run"] = baseline_info["best"]["run_name"]
        common.save_json(best / "test_metrics.json", report)
        common.save_errors(best / "test_errors.csv", test_rows, predictions, labels)
        result_files.write_predictions(best / "test_predictions.csv", test_rows, predictions, probabilities, labels)
        result_files.record_test_result(best, scores[key], labels, len(test_rows), seconds, truncated,
                                        official=official, with_meta=excluded)
        tag = "" if excluded else ("  [공식]" if official else "  [참고]")
        print(f"  [{key}] test macro F1 {scores[key]['macro_f1']:.4f} (accuracy {scores[key]['accuracy']:.4f}){tag}")
        print(f"  [{key}] test 정답: {common.format_correct(scores[key], labels)}")

    write_test_comparison(topic, dataset)
    if excluded:
        print("\n  [참고] 이 데이터셋은 모델 추천 대상이 아니라 서버용입니다. 4개 모델을 같은 조건으로 평가했고, 공식 모델 표시는 하지 않았습니다.")
        return scores
    print(f"\n  [중요] 공식 모델은 validation 으로 정한 {selected} 입니다. 다른 모델의 test 점수는 참고용이며,")
    print("         이 점수를 보고 모델이나 설정을 다시 고르지 마세요. 바꾸면 추가 실험으로 구분해 기록해야 합니다.")
    return scores
