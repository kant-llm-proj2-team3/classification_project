"""Project2 분류 모델 실험 메뉴 (baseline/steps/run_menu.py).

실행 (프로젝트 루트 project_2 에서):
    uv run python baseline/steps/run_menu.py

  1) 학습      : baseline + bert + roberta + koelectra 를 차례로 학습한다.
                 학습할 때마다 날짜 폴더(예: 20261012_195030)가 새로 생기고, 그 모델의 best_model 보다
                 validation 점수가 좋아졌을 때만 best_model 을 교체한다.
                 (현재 설정을 로그 맨 위에 남기고, 화면 출력 전체를 logs/ 에 저장)
  2) 초기화    : 학습 결과(artifacts)를 지운다. CSV 데이터와 logs/ 는 건드리지 않는다.
  3) 추천      : 각 모델 best_model 의 validation 결과만 보고 공식 모델 1개를 정한다 (bert / roberta / koelectra / laya / clm 중)
  4) test 평가 : baseline + bert + roberta + koelectra 를 각 best_model 로 test 평가 (모델마다 한 번, 공식 표시는 추천 결과를 따름)
  5) 길이 점검 : CSV 파일별로 몇 행이 192/512토큰을 넘는지 확인 (입력이 잘리는 정도)
  6) 결과 파일 : 이미 있는 결과에 main.json / sub.json / 문장별 예측 CSV 를 만든다 (다시 학습 안 함)
  7) 그래프   : 모든 best_model 의 main.json 을 읽어 PNG 그래프를 artifacts/viz/charts 에 만든다 (matplotlib 필요)
  8) 한 번에 마무리 : 결과 파일 보강 -> 모델 추천(validation 기준) -> 4개 모델 test -> 그래프 (다시 학습 안 함, 학습 끝난 뒤에)
"""
import sys
import time
import traceback
from pathlib import Path

# bert/steps 의 학습 파일을 가져오기 위해 경로를 추가한다. (baseline/steps 는 자동으로 포함됨)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bert" / "steps"))

import check_lengths  # noqa: E402
import common  # noqa: E402
import make_charts  # noqa: E402
import result_files  # noqa: E402
import train_bert  # noqa: E402
import train_koelectra  # noqa: E402
import train_roberta  # noqa: E402
import train_tfidf  # noqa: E402

TRAINERS = [(common.BASELINE_KEY, train_tfidf.train), ("bert", train_bert.train),
            ("roberta", train_roberta.train), ("koelectra", train_koelectra.train)]


def ask(prompt):
    try:
        return input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "0"


def status_text(topic, dataset):
    """한 데이터셋의 진행 상황을 한 줄로 보여 준다. (모델별 학습 횟수와 best_model 의 validation Macro-F1)"""
    parts = []
    for key in common.ALL_MODEL_KEYS:
        runs = len(common.complete_run_dirs(key, topic, dataset))
        info = common.read_candidate(key, topic, dataset)      # main.json 기준
        best = f"(best {info['validation_macro_f1']:.4f})" if info else ""
        parts.append(f"{key} {runs}회{best}")
    for key in common.EXTERNAL_KEYS:     # 외부 모델은 학습 횟수 대신 결과가 있는지만 보여 준다
        summary = common.read_candidate(key, topic, dataset)
        parts.append(f"{key}(외부) " + (f"best {summary['validation_macro_f1']:.4f}" if summary else "-"))
    recommendation = common.dataset_result_dir(topic, dataset) / "recommendation.json"
    if common.is_recommend_excluded(dataset):
        parts.append("추천 제외(서버용)")
    else:
        parts.append("추천 " + (common.load_json(recommendation)["recommended_model"] if recommendation.exists() else "-"))
    tested = any((common.best_dir(k, topic, dataset) / "test_metrics.json").exists()
                 for k in common.TRANSFORMER_KEYS)
    parts.append("test " + ("완료" if tested else "-"))
    return " | ".join(parts)


def usable(topic, dataset):
    return common.is_enabled(topic, dataset) and not common.missing_files(topic, dataset)


def choose_pairs():
    """데이터셋 선택. 종료하면 None."""
    pairs = common.all_pairs()
    while True:
        print("\n[데이터셋 선택]")
        for number, (topic, dataset) in enumerate(pairs, start=1):
            if not common.is_enabled(topic, dataset):
                info = "OFF  (baseline_bert_config.py 의 DATASET_SWITCH 에서 True 로 바꾸면 사용)"
            elif common.missing_files(topic, dataset):
                info = "ON   (데이터 파일 없음)"
            else:
                info = status_text(topic, dataset)
            print(f"  {number:>2}) {topic}/{dataset:<14} {info}")
        print("   a) ON 이고 데이터가 있는 데이터셋 전부")
        print("   0) 종료")
        choice = ask("번호 선택 > ")
        if choice == "0":
            return None
        if choice == "a":
            chosen = [pair for pair in pairs if usable(*pair)]
        elif choice.isdigit() and 1 <= int(choice) <= len(pairs):
            chosen = [pairs[int(choice) - 1]]
            if not usable(*chosen[0]):
                print("  이 데이터셋은 OFF 이거나 데이터 파일이 없어 사용할 수 없습니다.")
                continue
        else:
            print("  올바른 번호를 입력하세요.")
            continue
        if chosen:
            return chosen
        print("  사용할 수 있는 데이터셋이 없습니다.")


def action_train(pairs):
    per_dataset = 1 + sum(len(common.model_learning_rates(k)) for k in common.TRANSFORMER_KEYS)
    print(f"\n{len(pairs)}개 데이터셋 x 모델 {len(common.ALL_MODEL_KEYS)}개 (데이터셋당 학습 {per_dataset}번)을 시작합니다.")
    print("학습할 때마다 날짜 폴더가 새로 생기고, 그 모델의 best_model 보다 좋을 때만 best_model 이 교체됩니다.")
    print("(이미 학습한 결과가 있어도 항상 새로 학습합니다. 이전 결과를 지우려면 2번 초기화)")
    if ask("시작할까요? (y/n) > ") != "y":
        return
    with common.session_log("train"):          # 이 아래의 화면 출력 전체가 logs/train_<시각>.log 에도 저장된다
        common.print_settings()
        print(f"  대상 데이터셋: {', '.join(f'{t}/{d}' for t, d in pairs)}")
        started = time.perf_counter()
        failures = []
        for topic, dataset in pairs:
            for key, trainer in TRAINERS:
                try:
                    failures += [f"{topic}/{dataset}/{name}" for name in trainer(topic, dataset)]
                except Exception:
                    print(f"  [{key}] 실행 실패:")
                    traceback.print_exc()
                    failures.append(f"{topic}/{dataset}/{key}")
        print(f"\n학습 종료 ({(time.perf_counter() - started) / 60:.1f}분)")
        if failures:
            print("실패한 항목:", ", ".join(failures))
            print("원인을 해결한 뒤 1번을 다시 실행하세요. (실패한 run 은 폴더가 남지 않습니다)")


def action_reset(pairs):
    options = {"1": "all", "2": "baseline", "3": "bert", "4": "roberta", "5": "koelectra"}
    print("\n초기화 범위: 1) 전체  2) baseline  3) bert  4) roberta  5) koelectra  0) 취소")
    print("(선택한 범위의 best_model 과 날짜 폴더가 모두 지워집니다. 팀원이 넘겨 준 외부 모델(laya 등) 결과는 지우지 않습니다)")
    scope = options.get(ask("선택 > "))
    if scope is None:
        print("취소했습니다.")
        return
    targets = [(pair, path) for pair in pairs for path in common.reset_targets(*pair, scope)]
    if not targets:
        print("지울 학습 결과가 없습니다.")
        return
    print("\n다음 학습 결과가 삭제됩니다. (CSV 데이터, 코드, logs/ 폴더의 실행 로그는 삭제되지 않습니다)")
    print("이전 결과를 보관하려면 지우기 전에 해당 폴더를 다른 곳에 복사해 두세요.")
    for (topic, dataset), path in targets:
        print(f"  - {path}  ({common.dir_size_mb(path):.1f}MB)")
    if ask("정말 삭제할까요? 'yes' 를 입력하세요 > ") != "yes":
        print("취소했습니다.")
        return
    for pair in pairs:
        common.reset_results(*pair, scope)
    print("삭제했습니다.")


def action_recommend(pairs):
    with common.session_log("recommend"):
        for pair in pairs:
            common.recommend_model(*pair)


def action_test(pairs):
    print("\ntest 평가: baseline + bert + roberta + koelectra 를 각 best_model 로 평가합니다. (모델마다 한 번만, 이미 한 모델은 건너뜀)")
    print("공식 모델은 validation 으로 정한 추천 모델입니다. test 점수를 보고 모델·설정을 다시 고르면 안 됩니다. (3번을 먼저 하면 공식 표시가 바로 붙음)")
    if ask("진행할까요? (y/n) > ") != "y":
        return
    with common.session_log("test"):
        for pair in pairs:
            try:
                common.evaluate_final(*pair)
            except (RuntimeError, FileExistsError, FileNotFoundError, ValueError) as error:
                print(f"  [{pair[0]}/{pair[1]}] 건너뜀: {error}")


def action_lengths(pairs):
    print("\nCSV 파일별로 몇 행이 192/512토큰을 넘는지 확인합니다. (선택한 데이터셋과 무관하게 전체 파일을 점검)")
    check_lengths.main()


def action_rebuild(pairs):
    print("\n이미 있는 학습 결과에 main.json / sub.json / 문장별 예측 파일을 만듭니다. (다시 학습하지 않습니다)")
    print("best_model 의 가중치로 validation 을 한 번 더 예측해서 문장별 예측을 만들기 때문에 모델마다 몇 초 걸립니다.")
    print("(이미 만들어진 파일은 건너뜁니다)")
    if ask("진행할까요? (y/n) > ") != "y":
        return
    with common.session_log("rebuild"):
        for topic, dataset in pairs:
            print(f"\n[{topic}/{dataset}]")
            result_files.rebuild_result_files(topic, dataset)


def action_charts(pairs):
    print("\n모든 데이터셋 x 모델의 best_model/main.json 을 읽어 그래프(PNG)를 만듭니다. (선택한 데이터셋과 무관, 위치: artifacts/viz/charts)")
    with common.session_log("charts"):
        make_charts.run()


def action_finish(pairs):
    print("\n한 번에 마무리: 빠진 결과 파일 보강 -> 모델 추천(validation 기준) -> 4개 모델 test 평가 -> 그래프 만들기")
    print("선택한 데이터셋 전체에 현재 best_model 로 진행합니다. (다시 학습하지 않습니다)")
    print("주의: 학습이 모두 끝난 뒤에 실행하세요. test 는 모델마다 한 번만 평가하고,")
    print("      공식 모델은 validation 으로 정한 추천 모델입니다. test 점수를 보고 모델을 바꾸면 안 됩니다.")
    if ask("진행할까요? (y/n) > ") != "y":
        return
    with common.session_log("finish"):
        for topic, dataset in pairs:
            print(f"\n[{topic}/{dataset}]")
            try:
                result_files.rebuild_result_files(topic, dataset)
                common.recommend_model(topic, dataset)
                common.evaluate_final(topic, dataset)
            except (RuntimeError, FileExistsError, FileNotFoundError, ValueError) as error:
                print(f"  [{topic}/{dataset}] 건너뜀: {error}")
        print("\n[그래프 만들기]")
        make_charts.run()


ACTIONS = {"1": action_train, "2": action_reset, "3": action_recommend,
           "4": action_test, "5": action_lengths, "6": action_rebuild, "7": action_charts,
           "8": action_finish}


def migrate_names_if_needed():
    """예전 이름(basic_plus, advanced_plus)으로 저장된 결과 폴더가 남아 있으면 새 이름(basic_merge, advanced_merge)으로 옮기도록 묻는다."""
    found = common.legacy_result_folders()
    if not found:
        return
    print("\n[예전 이름의 결과 폴더가 발견되었습니다]")
    for topic, old, new in found:
        print(f"  artifacts/{topic}/{old}  ->  artifacts/{topic}/{new}")
    print("데이터셋 이름이 basic_plus -> basic_merge, advanced_plus -> advanced_merge 로 바뀌었습니다.")
    print("이름을 바꾸고, 폴더 안 json/csv 에 적힌 옛 이름도 함께 바꿉니다. (가중치·점수 내용은 그대로입니다)")
    print("바꾸지 않으면 기존 학습 결과가 새 이름에서 보이지 않습니다.")
    if ask("지금 바꿀까요? (y/n) > ") != "y":
        print("  건너뜁니다. (다음 실행 때 다시 묻습니다)")
        return
    with common.session_log("migrate"):
        moved, skipped, failed = common.migrate_legacy_dataset_names()
        for line in moved:
            print(f"  옮김: {line}")
        for line in skipped:
            print(f"  [건너뜀] {line}")
        for line in failed:
            print(f"  [실패] {line}")
        if failed:
            print("  실패한 폴더는 OneDrive 동기화 중이거나 다른 프로그램이 열어 둔 경우일 수 있습니다. 닫고 다시 실행해 보세요.")


def main():
    migrate_names_if_needed()
    while True:
        pairs = choose_pairs()
        if pairs is None:
            print("종료합니다.")
            return
        names = ", ".join(f"{t}/{d}" for t, d in pairs)
        while True:
            print(f"\n[선택된 데이터셋] {names}")
            print("  1) 학습  (baseline + bert + roberta + koelectra, best_model 과 비교해 좋으면 교체)")
            print("  2) 학습 결과 초기화")
            print("  3) 모델 추천 - 각 모델 best_model 의 validation 결과로 공식 모델 선정")
            print("  4) test 평가 - 4개 모델을 각 best_model 로 평가 (공식 표시는 추천 결과를 따름)")
            print("  5) 입력 길이(토큰 수) 점검 - 파일별로 192/512토큰을 넘는 행 수")
            print("  6) 결과 파일 만들기 - 이미 있는 결과에 main.json / sub.json / 문장별 예측 추가")
            print("  7) 그래프 만들기 - best_model/main.json 으로 PNG 그래프 (artifacts/viz/charts)")
            print("  8) 한 번에 마무리 - 결과 파일 보강 -> 추천(validation) -> 4개 모델 test -> 그래프 (학습 끝난 뒤)")
            print("  b) 데이터셋 다시 선택")
            print("  0) 종료")
            choice = ask("메뉴 선택 > ")
            if choice == "0":
                print("종료합니다.")
                return
            if choice == "b":
                break
            if choice in ACTIONS:
                ACTIONS[choice](pairs)
            else:
                print("  올바른 번호를 입력하세요.")


if __name__ == "__main__":
    main()
