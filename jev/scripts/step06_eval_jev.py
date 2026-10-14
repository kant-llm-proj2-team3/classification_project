"""
jev(TypeSafe System One API)를 학습 없이 평가해서, 다른 로컬 모델과 비교할 수 있는 형태로 artifacts에 저장한다.

jev는 System One 계열 모델(laya, clm)의 baseline이다. 파인튜닝하지 않고 TypeSafe API를 불러서 얻은 분류 결과만 쓴다.
(API 사용법: https://docs.typesafe.ai) laya, clm과 같은 문장, 같은 질문(labels.json의 라벨과 설명), 같은 라벨 순서로 묻고,
결과는 step04, step05와 같은 구조로 저장한다. 학습이 없으므로 가중치 파일은 없다.
  artifacts/<topic>/<dataset>/jev/
      best_model/        validation이 가장 좋았던 실행의 결과 (main.json, sub.json, 예측, 오답, API 사용량)
      20261014_120000/   실행 1회 = 폴더 1개
같은 문장은 데이터셋이 달라도 API를 한 번만 부르고(jev/tmp/jev_responses 캐시), 끝난 요청은 중간에 끊겨도 남아서 이어서 할 수 있다.
API 키는 .env의 JEV_API_KEY에 적는다. 키는 화면, 로그, 결과 파일에 남기지 않는다.

공통 표와 그래프에 넣으려면 baseline_bert_config.py의 EXTERNAL_MODELS["jev"]를 True로 바꾼다. (jev는 모델 추천에서는 빠지고 참고용으로만 들어간다)

루트 디렉토리 기준 실행 코드 (Windows는 PYTHONUTF8=1 권장)
uv run python ./jev/scripts/step06_eval_jev.py --check                                     # API 키와 연결만 확인 (문장은 보내지 않는다)
uv run python ./jev/scripts/step06_eval_jev.py --dry-run                                   # API를 부르지 않고 보낼 요청 수와 예상 비용만 출력
uv run python ./jev/scripts/step06_eval_jev.py                                             # ON인 데이터셋 전부의 validation 평가
uv run python ./jev/scripts/step06_eval_jev.py --topic documents --dataset basic
uv run python ./jev/scripts/step06_eval_jev.py --test                                      # best_model로 test 평가 (모델마다 1회)
"""

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

import jev_settings as S
from util.data_convert import QID, load_questions
from util.jev_client import (JevAuthError, JevClient, ResponseCache, build_payload, cache_path, model_names,
                             request_key)

# 동료 코드의 설정, 지표 계산, best_model 교체, 결과 파일 함수를 그대로 사용해 형식을 맞춘다.
sys.path.insert(0, str(S.ROOT))
sys.path.insert(0, str(S.ROOT / "baseline" / "steps"))
import baseline_bert_config as cfg  # noqa: E402
import common  # noqa: E402
import result_files  # noqa: E402

JEV_KEY = S.JEV_KEY
SPLITS = ("validation", "test")


# ═════════════════════════ 0. 데이터 · 공통 함수 ═════════════════════════
def load_split(topic: str, dataset: str, split: str) -> dict:
    """split의 CSV 행과, laya/clm과 같은 질문(labels.json의 라벨, 설명, 질문 문장)을 읽는다."""
    data = common.load_data(topic, dataset, with_test=(split == "test"))
    labels = data["labels"]
    questions, _ = load_questions(common.dataset_paths(topic, dataset)["labels"], topic)
    if list(questions[QID]["criteria"]) != labels:
        raise ValueError(f"선택지 순서 {list(questions[QID]['criteria'])}가 labels.json {labels}와 다릅니다.")
    return {"labels": labels, "rows": data[split], "questions": questions}


def label_ids(rows: list[dict], labels: list[str]) -> list[int]:
    label2id = {label: index for index, label in enumerate(labels)}
    return [label2id[row[cfg.LABEL_COLUMN]] for row in rows]


def cross_entropy(probabilities: list[list[float]], actual: list[int]) -> float:
    """정답 라벨의 확률로 구한 평균 교차 엔트로피. API 확률은 반올림돼서 0일 수 있어 loss_floor 아래는 그 값으로 바꾼다."""
    floor = S.JEV_CONFIG["loss_floor"]
    return -sum(math.log(max(p[target], floor)) for p, target in zip(probabilities, actual, strict=True)) / len(actual)


def cost_usd(input_tokens: int) -> float:
    return input_tokens / 1e6 * S.JEV_CONFIG["price_per_million_input_tokens_usd"]


def environment() -> dict:
    return {"device": "remote (TypeSafe API)", "gpu": None, "python": platform.python_version(),
            "httpx": httpx.__version__}


def api_info(client: JevClient, stats: dict) -> dict:
    """run_config에 남기는 API 설정. 키와 주소의 인증 정보는 넣지 않는다. (주소는 호스트만)"""
    cfg_s = S.JEV_CONFIG
    return {"host": urlparse(client.base_url).netloc, "endpoint": S.ENDPOINT, "model_requested": client.model,
            "models_returned": stats["models"], "concurrency": client.concurrency, "timeout": cfg_s["timeout"],
            "max_retries": cfg_s["max_retries"], "loss_floor": cfg_s["loss_floor"]}


def evaluate_rows(client: JevClient, rows: list[dict], data: dict, refresh: bool, label: str) -> dict:
    """문장마다 jev에게 라벨을 묻는다. 같은 문장은 한 번만 보내고 캐시에 있으면 보내지 않는다.

    반환: answers(문장별 API 응답), predictions(라벨 번호), probabilities(라벨 순서의 확률), stats(사용량 요약)
    """
    labels, questions = data["labels"], data["questions"]
    items = []
    for row in rows:
        payload = build_payload(client.model, row[cfg.TEXT_COLUMN], questions)
        items.append((request_key(payload), payload))
    results = client.run(items, refresh=refresh, label=label)
    answers = [results[key] for key, _ in items]
    predictions = [labels.index(answer.choice) for answer in answers]
    probabilities = [[answer.probabilities[name] for name in labels] for answer in answers]
    unique = {key for key, _ in items}
    fresh = {key for key in unique if not results[key].cached}      # 이번 실행에서 실제로 보낸 요청
    stats = {"rows": len(rows), "requests_unique": len(unique), "requests_sent": len(fresh),
             "requests_cached": len(unique) - len(fresh),
             "input_tokens": sum(a.input_tokens for a in answers), "output_tokens": sum(a.output_tokens for a in answers),
             "sent_input_tokens": sum(results[key].input_tokens for key in fresh),
             "latency_seconds": sum(a.latency for a in answers),
             "models": sorted({a.model for a in answers})}
    return {"answers": answers, "predictions": predictions, "probabilities": probabilities, "stats": stats}


def write_api_details(path: Path, rows: list[dict], answers: list) -> None:
    """문장별 API 응답의 부가 정보. API가 계산한 confidence, 토큰 수, 응답 시간, 캐시에서 가져왔는지."""
    fields = ["id", "api_choice", "api_confidence", "input_tokens", "output_tokens", "latency_seconds", "from_cache"]
    common.write_csv(path, fields, [
        {"id": row[cfg.ID_COLUMN], "api_choice": answer.choice, "api_confidence": round(answer.confidence, 6),
         "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens,
         "latency_seconds": round(answer.latency, 4), "from_cache": int(answer.cached)}
        for row, answer in zip(rows, answers, strict=True)])


def api_cost_fields(stats: dict, prefix: str = "api_") -> dict:
    """main.json의 cost 칸에 덧붙이는 API 사용량. num_parameters 같은 로컬 모델용 칸은 비워 둔다."""
    return {f"{prefix}requests": stats["requests_unique"], f"{prefix}input_tokens": stats["input_tokens"],
            f"{prefix}output_tokens": stats["output_tokens"], f"{prefix}cost_usd": round(cost_usd(stats["input_tokens"]), 6)}


def usage_line(stats: dict) -> str:
    mean = stats["latency_seconds"] / max(1, stats["rows"])
    return (f"API 요청 {stats['requests_unique']}건 (이번에 보낸 것 {stats['requests_sent']}건, 캐시 {stats['requests_cached']}건) | "
            f"입력 토큰 {stats['input_tokens']:,}, 출력 토큰 {stats['output_tokens']:,} | "
            f"예상 비용 ${cost_usd(stats['input_tokens']):.4f} | 평균 응답 {mean:.2f}초 | 응답 모델 {', '.join(stats['models'])}")


# main.json / sub.json을 동료 코드로 만든 뒤 API 평가에 맞게 train 칸과 cost 칸을 보충한다. (zero-shot과 같은 처리)
def write_main(run_dir: Path, train_extra: dict, cost_extra: dict) -> None:
    main, sub = result_files.build_main_sub(run_dir, None)
    main["train"] = train_extra
    main["curve"] = None
    main["cost"].update(cost_extra)
    common.save_json(run_dir / "main.json", main)
    common.save_json(run_dir / "sub.json", sub)


# ═════════════════════════ 1. validation 평가 ═════════════════════════
def run_validation(topic: str, dataset: str, client: JevClient, refresh: bool) -> dict:
    run_dir = common.begin_run(JEV_KEY, topic, dataset)
    try:
        return _validation(run_dir, topic, dataset, client, refresh)
    except BaseException:
        common.safe_delete(run_dir)          # 끝나지 않은 실행은 지운다. 받아 둔 응답은 캐시에 남아서 다시 실행하면 이어서 한다.
        raise


def _validation(run_dir: Path, topic: str, dataset: str, client: JevClient, refresh: bool) -> dict:
    data = load_split(topic, dataset, "validation")
    labels, rows = data["labels"], data["rows"]
    started = time.perf_counter()
    result = evaluate_rows(client, rows, data, refresh, f"{topic}/{dataset} validation")
    wall = time.perf_counter() - started
    stats = result["stats"]
    actual = label_ids(rows, labels)
    loss = cross_entropy(result["probabilities"], actual)
    scores = common.classification_scores(actual, result["predictions"], labels)
    model_id = stats["models"][0] if len(stats["models"]) == 1 else client.model
    if len(stats["models"]) > 1:
        print(f"  [주의] 응답한 모델 버전이 하나가 아닙니다: {stats['models']} (별칭을 쓰면 중간에 버전이 바뀔 수 있습니다)")

    common.save_json(run_dir / "run_config.json", {
        "topic": topic, "dataset": dataset, "model_key": JEV_KEY, "model_id": model_id, "run_name": run_dir.name,
        "learning_rate": None, "epochs_max": 0, "batch_size": 1, "gradient_accumulation_steps": 1,
        "max_length": None, "truncation": None, "seed": None, "labels": labels, "validation_rows": len(rows),
        "validation_csv": common.relative_paths(common.dataset_paths(topic, dataset)["validation"]),
        "question": data["questions"][QID], "option_order": labels, "api": api_info(client, stats),
        **environment()})
    summary = common.make_summary(
        JEV_KEY, model_id, run_dir.name, None, scores, loss, 0, 0.0, stats["latency_seconds"], None, 0, len(rows),
        extra={"epochs_run": 0, "epochs_max": 0, "stop_reason": "학습 없음 (API 평가)", "wall_seconds": wall,
               "requests_sent": stats["requests_sent"], "requests_cached": stats["requests_cached"],
               "sent_input_tokens": stats["sent_input_tokens"], **api_cost_fields(stats)})
    common.save_json(run_dir / "training_summary.json", summary)
    common.save_json(run_dir / "validation_metrics.json", scores)
    common.save_errors(run_dir / "validation_errors.csv", rows, result["predictions"], labels)
    result_files.write_predictions(run_dir / "validation_predictions.csv", rows, result["predictions"],
                                   result["probabilities"], labels)
    write_api_details(run_dir / "validation_api_details.csv", rows, result["answers"])

    log = (f"  [{JEV_KEY}/{run_dir.name}] {topic}/{dataset} validation {len(rows)}행 | "
           f"val_loss={loss:.4f} val_macro_f1={scores['macro_f1']:.4f}\n"
           f"  [{JEV_KEY}/{run_dir.name}] validation 정답: {common.format_correct(scores, labels)}\n"
           f"  [{JEV_KEY}/{run_dir.name}] {usage_line(stats)}")
    print(log)
    (run_dir / "train.log").write_text(log + "\n", encoding="utf-8")
    write_main(run_dir, {"fine_tuned": False, "api_model": model_id}, api_cost_fields(stats))

    common.finalize_run(JEV_KEY, topic, dataset, run_dir, summary, run_dir / common.WEIGHTS_TMP)
    common.mark_complete(run_dir)
    return stats


# ═════════════════════════ 2. test 평가 (best_model, 한 번만) ═════════════════════════
def evaluate_test(topic: str, dataset: str, client: JevClient, refresh: bool) -> dict | None:
    best = common.best_dir(JEV_KEY, topic, dataset)
    best_info = common.read_best_info(JEV_KEY, topic, dataset)
    if best_info is None:
        print(f"  [{JEV_KEY}] best_model이 없어 건너뜁니다. 먼저 validation 평가를 하세요.")
        return None
    if (best / "test_metrics.json").exists():
        print(f"  [{JEV_KEY}] 이미 test 평가를 했습니다. 덮어쓰지 않고 건너뜁니다.")
        return None
    data = load_split(topic, dataset, "test")
    labels, rows = data["labels"], data["rows"]
    result = evaluate_rows(client, rows, data, refresh, f"{topic}/{dataset} test")
    stats = result["stats"]
    recorded_model = best_info["best"].get("model_id")
    if recorded_model and stats["models"] != [recorded_model]:
        print(f"  [주의] test를 응답한 모델 {stats['models']}이 validation 때의 {recorded_model}과 다릅니다.")
    actual = label_ids(rows, labels)
    scores = common.classification_scores(actual, result["predictions"], labels)
    # jev는 모델 추천 대상이 아니므로 공식(official) 점수가 아니라 참고용으로 기록한다. (laya, clm과 같은 처리)
    official = None if common.is_recommend_excluded(dataset) else False
    common.save_json(best / "test_metrics.json", {
        "topic": topic, "dataset": dataset, "split": "test", "selected_model": None, "evaluated_model": JEV_KEY,
        "official": official, "selected_run": best_info["best"]["run_name"], "model_id": best_info["best"].get("model_id"),
        "classifier": scores, "baseline": None, "api": {"models_returned": stats["models"], "loss": cross_entropy(result["probabilities"], actual)}})
    common.save_errors(best / "test_errors.csv", rows, result["predictions"], labels)
    result_files.write_predictions(best / "test_predictions.csv", rows, result["predictions"], result["probabilities"], labels)
    write_api_details(best / "test_api_details.csv", rows, result["answers"])
    result_files.record_test_result(best, scores, labels, len(rows), stats["latency_seconds"], None,
                                    official=official, with_meta=True)
    main = common.load_json(best / "main.json")
    main["cost"].update(api_cost_fields(stats, "api_test_"))
    common.save_json(best / "main.json", main)
    print(f"  [{JEV_KEY}] test macro F1 {scores['macro_f1']:.4f} (accuracy {scores['accuracy']:.4f}) [참고]")
    print(f"  [{JEV_KEY}] test 정답: {common.format_correct(scores, labels)}")
    print(f"  [{JEV_KEY}] {usage_line(stats)}")
    return stats


# ═════════════════════════ 3. 호출 전 점검 ═════════════════════════
def dry_run(pairs: list, split: str) -> None:
    """API를 부르지 않는다. 데이터셋마다 보낼 요청 수와 어림잡은 비용을 보여 준다. (API 키도 필요 없다)"""
    cache = ResponseCache(cache_path(S.MODEL))
    per_char = S.JEV_CONFIG["dry_run_tokens_per_char"]
    seen, new_tokens = set(), 0.0
    print(f"모델 {S.MODEL} | {split} | 캐시에 있는 응답 {len(cache)}건\n")
    print(f"{'dataset':<27}{'문장':>6}{'서로 다른 요청':>14}{'캐시에 있음':>12}{'새로 보낼 것':>12}")
    for topic, dataset in pairs:
        data = load_split(topic, dataset, split)
        question_chars = len(json.dumps(data["questions"], ensure_ascii=False))
        keys = {}
        for row in data["rows"]:
            text = row[cfg.TEXT_COLUMN]
            keys[request_key(build_payload(S.MODEL, text, data["questions"]))] = len(text) + question_chars
        cached = sum(1 for key in keys if cache.get(key) is not None)
        print(f"{topic + '/' + dataset:<27}{len(data['rows']):>6}{len(keys):>14}{cached:>12}{len(keys) - cached:>12}")
        for key, chars in keys.items():
            if key not in seen and cache.get(key) is None:
                new_tokens += chars * per_char
            seen.add(key)
    sent = sum(1 for key in seen if cache.get(key) is None)
    print(f"\n전체에서 서로 다른 요청 {len(seen)}건 중 새로 보낼 것 {sent}건 (같은 문장은 데이터셋이 달라도 한 번만 보냅니다)")
    print(f"입력 토큰 약 {new_tokens:,.0f}개(글자 수로 넉넉하게 어림잡음) -> 예상 비용 약 ${cost_usd(int(new_tokens)):.3f} "
          f"(입력 토큰 100만 개에 ${S.JEV_CONFIG['price_per_million_input_tokens_usd']}, 출력은 무료)")
    print("실제 사용량은 실행 후 API 응답의 usage로 기록됩니다. 문장이 TypeSafe API로 전송되니 보내도 되는 데이터인지 확인하세요.")


def check_connection() -> None:
    """GET /v1/models만 불러서 키와 연결을 확인한다. 데이터셋의 문장은 보내지 않는다."""
    with JevClient() as client:
        body = client.list_models()
    names = model_names(body)
    print(f"연결 성공: {urlparse(client.base_url).netloc} (키는 올바릅니다)")
    print("이 계정이 쓸 수 있는 모델: " + (", ".join(names) if names else f"(이름을 읽지 못했습니다) {json.dumps(body, ensure_ascii=False)[:300]}"))
    print(f"평가에 쓸 모델: {client.model}  (별칭이 아니라 버전 ID라서 목록에 없어도 요청에는 쓸 수 있습니다)")


# ═════════════════════════ 4. 실행 ═════════════════════════
def main():
    ap = argparse.ArgumentParser(description="jev(TypeSafe System One API) baseline 평가")
    ap.add_argument("--topic", choices=cfg.TOPICS, help="생략하면 ON인 주제 전부")
    ap.add_argument("--dataset", choices=cfg.DATASET_NAMES, help="생략하면 ON인 데이터셋 전부")
    ap.add_argument("--test", action="store_true", help="validation 대신 best_model로 test 평가 (모델마다 1회)")
    ap.add_argument("--dry-run", action="store_true", help="API를 부르지 않고 보낼 요청 수와 예상 비용만 출력")
    ap.add_argument("--check", action="store_true", help="API 키와 연결만 확인 (문장은 보내지 않는다)")
    ap.add_argument("--refresh", action="store_true", help="캐시를 무시하고 다시 요청 (새 응답이 캐시를 덮어쓴다)")
    ap.add_argument("--concurrency", type=int, default=None, help="동시에 보내는 요청 수 임시 변경 (기본: jev_settings의 값)")
    a = ap.parse_args()

    if a.check:
        try:
            check_connection()
        except (RuntimeError, httpx.HTTPError) as error:         # 키 없음, 키 오류(JevAuthError), 연결 실패
            print(f"[연결 실패] {error if str(error) else type(error).__name__}")
            sys.exit(1)
        return
    pairs = common.ready_pairs(a.topic, a.dataset)
    if not pairs:
        print("실행할 데이터셋이 없습니다. baseline_bert_config.py의 DATASET_SWITCH를 확인하세요.")
        return
    split = "test" if a.test else "validation"
    if a.dry_run:
        dry_run(pairs, split)
        return

    try:
        client = JevClient(concurrency=a.concurrency)
    except RuntimeError as error:                                 # API 키가 없는 경우
        print(f"[시작하지 못했습니다] {error}")
        sys.exit(1)
    with common.session_log("test_jev" if a.test else "eval_jev"), client:
        print(f"jev: {client.model} | {urlparse(client.base_url).netloc} | {split} | 동시 {client.concurrency}건 | 캐시 {len(client.cache)}건")
        failures, sent_tokens, sent_requests = [], 0, 0
        for topic, dataset in pairs:
            print(f"\n=== {topic}/{dataset} · jev ===")
            try:
                if a.test:
                    stats = evaluate_test(topic, dataset, client, a.refresh)
                else:
                    stats = run_validation(topic, dataset, client, a.refresh)
                if stats:
                    sent_tokens += stats["sent_input_tokens"]
                    sent_requests += stats["requests_sent"]
            except JevAuthError as error:                         # 키가 잘못되면 다른 데이터셋도 모두 실패하므로 바로 멈춘다
                print(f"  [jev] {error}")
                sys.exit(1)
            except Exception as error:
                print(f"  [jev] {topic}/{dataset} 실패: {type(error).__name__}: {error}")
                failures.append(f"{topic}/{dataset}")
        print(f"\n이번 실행에서 실제로 보낸 요청 {sent_requests}건, 입력 토큰 {sent_tokens:,}개 (예상 비용 ${cost_usd(sent_tokens):.4f})")
        if not cfg.EXTERNAL_MODELS.get(JEV_KEY):
            print(f"[안내] 공통 표와 그래프에 넣으려면 baseline_bert_config.py의 EXTERNAL_MODELS[\"{JEV_KEY}\"]를 True로 바꾸세요.")
        if failures:
            print(f"\n실패한 데이터셋: {failures} (받아 둔 응답은 캐시에 남아 있어서 다시 실행하면 이어서 합니다)")
            sys.exit(1)


if __name__ == "__main__":
    main()
