"""test 최종 평가 (baseline/steps/test_eval.py).

baseline + bert + roberta + koelectra 를 각 best_model 로 모두 평가한다. 공식 모델은 이 test 점수로 정하므로
(recommend_model, 메뉴 3번) test 평가를 먼저 한다. 그래서 여기서는 공식(official) 표시를 하지 않는다. (test_meta.official = None)

저장되는 것 (각 모델의 best_model 폴더)
  test_metrics.json       : test 점수와 혼동행렬 (kit 의 test_metrics.json 과 같은 구조 + official 표시)
  test_errors.csv         : test 에서 틀린 문장
  test_predictions.csv    : test 모든 문장의 예측 (팀원과 비교용)
  main.json 의 test 칸    : 몇 문제 중 몇 개 정답, F1 등 (+ test_meta 에 공식 여부)
모델별 test 비교표는 화면에만 출력하고 파일로는 저장하지 않는다. (그래프는 make_charts.py 가 main.json 으로 만든다)
"""
import sys
import time
from pathlib import Path

# 설정 파일(baseline_bert_config.py)은 프로젝트 루트(project_2)에 있다. (이 파일에서 두 단계 위)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import baseline_bert_config as cfg  # noqa: E402
import common  # noqa: E402
import result_files  # noqa: E402

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


def external_status(topic, dataset, selected=None):
    """외부 모델(팀원 제공)의 test 결과가 main.json 에 있는지 화면에 알려 준다. 우리 코드는 외부 모델의 test 를 직접 평가하지 않는다."""
    for key in common.EXTERNAL_KEYS:
        path = common.best_dir(key, topic, dataset) / "main.json"
        if not path.exists():
            print(f"  [{key}(외부)] 결과 파일이 없어 비교에서 빠집니다.")
        elif not common.load_json(path).get("test"):
            warn = "  <- 공식 모델인데 test 결과가 없습니다. 팀원에게 main.json 의 test 칸을 채워 달라고 요청하세요" if key == selected else ""
            print(f"  [{key}(외부)] test 결과가 main.json 에 아직 없습니다.{warn}")


def write_test_comparison(topic, dataset):
    """각 모델 best_model 의 main.json 에 있는 test 점수를 모아 비교표를 화면에 출력한다. (이미 평가한 모델도 포함. 파일로는 저장하지 않는다)"""
    rows = []
    for key in common.RESULT_MODEL_KEYS:
        path = common.best_dir(key, topic, dataset) / "main.json"
        if not path.exists():
            continue
        main = common.load_json(path)
        test = main.get("test")
        if not test:
            continue
        rows.append({"model_name": key, "run_id": main["run_id"], "test_n": test["n"],
                     "test_correct": test["correct"], "test_accuracy": test["accuracy"],
                     "test_macro_f1": test["macro_f1"], "validation_macro_f1": (main.get("validation") or {}).get("macro_f1"),
                     "test_inference_seconds": main["cost"].get("test_inference_seconds")})
    if rows:
        print(f"\n  [test 비교표] {topic}/{dataset}")
        print(f"  {'model':<12}{'best run':<17}{'test 정답':>12}{'test_f1':>9}{'validation_f1':>15}")
        for row in rows:
            print(f"  {row['model_name']:<12}{row['run_id']:<17}"
                  f"{str(row['test_correct']) + '/' + str(row['test_n']):>12}{row['test_macro_f1']:>9.4f}"
                  f"{common._fmt(row['validation_macro_f1'], '.4f'):>15}")
    return rows


def evaluate_test(topic, dataset, mode="all"):
    """test 평가: baseline + 3개 모델을 각 best_model 로 모두 평가한다. 각 모델은 한 번만 평가한다. (mode 는 예전 호출과 맞추려고 남겨 둔 값)
    공식 모델은 이 점수로 정한다 -> 평가가 끝나면 3번(모델 추천)을 실행한다."""
    print(f"\n=== {topic}/{dataset} · test 평가: 4개 모델 모두 ===")
    data = common.load_data(topic, dataset, with_test=True)   # train/validation/test 분할 겹침까지 검사
    labels, test_rows = data["labels"], data["test"]
    label2id = {label: index for index, label in enumerate(labels)}
    actual = [label2id[row[cfg.LABEL_COLUMN]] for row in test_rows]

    outcomes = {}   # 모델 키 -> (best 요약, (예측, 확률, 시간, 잘린 행 수))
    for key in common.TRANSFORMER_KEYS:
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
        report = {"topic": topic, "dataset": dataset, "split": "test", "selected_model": None,
                  "evaluated_model": key, "official": None, "selected_run": best_summary["run_name"],
                  "model_id": best_summary["model_id"], "classifier": scores[key], "baseline": baseline_scores}
        if baseline_scores is not None:
            report["baseline_run"] = baseline_info["best"]["run_name"]
        common.save_json(best / "test_metrics.json", report)
        common.save_errors(best / "test_errors.csv", test_rows, predictions, labels)
        result_files.write_predictions(best / "test_predictions.csv", test_rows, predictions, probabilities, labels)
        result_files.record_test_result(best, scores[key], labels, len(test_rows), seconds, truncated,
                                        official=None, with_meta=True)
        print(f"  [{key}] test macro F1 {scores[key]['macro_f1']:.4f} (accuracy {scores[key]['accuracy']:.4f})")
        print(f"  [{key}] test 정답: {common.format_correct(scores[key], labels)}")

    write_test_comparison(topic, dataset)
    external_status(topic, dataset)
    print("\n  [다음] 공식 모델은 이 test 점수로 정합니다. 3번(모델 추천)을 실행하세요. (8번은 자동으로 이어서 실행)")
    return scores
