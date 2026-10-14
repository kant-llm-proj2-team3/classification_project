"""그래프 만들기 (baseline/steps/make_charts.py).

artifacts/<주제>/<데이터셋>/<모델>/best_model/main.json 만 읽어서 PNG 그래프를 만든다. (CSV 표는 만들지 않는다)
저장 위치: artifacts/viz/charts/   (초기화(2번)는 이 폴더를 지우지 않는다)

실행 (프로젝트 루트 project_2 에서):
    uv run python baseline/steps/make_charts.py
    uv run python baseline/steps/make_charts.py --with-total-merge      # 서버용 total_merge 도 그래프에 포함
메뉴 7번(그래프 만들기), 8번(마무리 마지막 단계)에서도 같은 코드가 실행된다.
처음 한 번 matplotlib 설치가 필요하다:  uv add matplotlib

만드는 그래프
  01_heatmap_test.png          데이터셋 x 모델 test 정확도 히트맵 (정답 수 표시, 공식 모델은 굵은 테두리)
  02_heatmap_validation.png    같은 표의 validation 버전 (모델 선정에 쓰인 점수)
  03_baseline_vs_official.png  baseline 대비 공식 모델의 test 정확도 막대
  04_learning_curves.png       데이터셋별 validation Macro-F1 학습 곡선 (별표 = 저장된 best 에폭)
  05_cost_vs_accuracy.png      모델 크기·추론 속도와 평균 test 정확도
  06_confusion_official.png    데이터셋별 공식 모델의 test 혼동행렬
외부 모델(laya, jev, clm)은 설정에서 켜져 있고 main.json 이 있으면 같은 그래프에 함께 들어간다.
main.json 에 없는 값(test, curve, 파라미터 수 등)은 그 그래프에서만 빠지고 오류는 나지 않는다.
"""
import argparse
import math
import statistics
import sys
from pathlib import Path

# 설정 파일(baseline_bert_config.py)은 프로젝트 루트(project_2)에 있다. (이 파일에서 두 단계 위)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import baseline_bert_config as cfg  # noqa: E402
import common  # noqa: E402
import result_files  # noqa: E402

CHART_DIR_NAME = "charts"
MODEL_COLORS = {"baseline": "#8a8a8a", "bert": "#2a78d6", "roberta": "#eb6834", "koelectra": "#1baf7a",
                "laya": "#8f5bd6", "jev": "#c9a227", "clm": "#d6458f"}
KOREAN = True            # 한글 글꼴을 찾으면 True. 못 찾으면 영어 제목을 쓴다.


def _t(korean, english):
    return korean if KOREAN else english


def _setup_matplotlib():
    """화면 없이 PNG 로 저장하도록 설정하고 한글 글꼴을 고른다. matplotlib 이 없으면 ImportError."""
    global KOREAN
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt
    wanted = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Noto Sans CJK JP"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = [name for name in wanted if name in available]
    KOREAN = bool(chosen)
    plt.rcParams["font.family"] = chosen + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    if not KOREAN:
        print("  [주의] 한글 글꼴(맑은 고딕 등)을 찾지 못해 그래프 제목을 영어로 만듭니다. (데이터셋·모델 이름은 그대로)")
    return plt


# ═════════════════════════ 데이터 정리 (main.json 만 사용) ═════════════════════════
def _label(dataset_id):
    return dataset_id.replace("/", " · ")


def _datasets(results):
    present = {r["dataset_id"] for r in results}
    return [f"{t}/{d}" for t in cfg.TOPICS for d in cfg.DATASET_NAMES if f"{t}/{d}" in present]


def _models(results):
    present = {r["model_name"] for r in results}
    return [m for m in common.RESULT_MODEL_KEYS if m in present]


def _official(main):
    return bool((main.get("test_meta") or {}).get("official"))


def _save(plt, fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ═════════════════════════ 그래프 ═════════════════════════
def chart_heatmap(plt, idx, datasets, models, split, path):
    import numpy as np
    values = np.full((len(datasets), len(models)), np.nan)
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            block = (idx.get((dataset, model)) or {}).get(split)
            if block:
                values[i, j] = block["accuracy"]
    if np.isnan(values).all():
        print(f"  [건너뜀] {path.name}: {split} 결과가 하나도 없습니다.")
        return None
    fig, ax = plt.subplots(figsize=(1.9 * len(models) + 3.6, 0.66 * len(datasets) + 2.0))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#eeeeee")
    ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0.6, vmax=1.0, aspect="auto")
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            main = idx.get((dataset, model))
            block = (main or {}).get(split)
            if not block:
                ax.text(j, i, "-", ha="center", va="center", fontsize=11, color="#999999")
                continue
            dark = (block["accuracy"] - 0.6) / 0.4 > 0.55
            ax.text(j, i, f"{block['accuracy']:.1%}\n{block['correct']}/{block['n']}", ha="center", va="center",
                    fontsize=10, color="white" if dark else "#222222")
            if _official(main):
                ax.add_patch(plt.Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, edgecolor="#ff8c00", linewidth=3.2))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([_label(d) for d in datasets], fontsize=10)
    ax.set_xticks([x - 0.5 for x in range(len(models) + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(len(datasets) + 1)], minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    name = _t("test 정확도", "Test accuracy") if split == "test" else _t("validation 정확도 (모델 선정에 사용)", "Validation accuracy (used for selection)")
    fig.suptitle(f"{name}  ({_t('정답 수/전체', 'correct/total')})", fontsize=13, y=0.995)
    fig.text(0.5, 0.0, _t("굵은 주황 테두리 = 공식 모델 (validation 으로 선정)    회색 = 결과 없음",
                          "Orange border = official model (selected on validation)    Gray = no result"),
             ha="center", va="top", fontsize=9, color="#555555")
    return _save(plt, fig, path)


def chart_baseline_vs_official(plt, idx, datasets, path):
    rows = []
    for dataset in datasets:
        base = (idx.get((dataset, "baseline")) or {}).get("test")
        official = [(m, r) for (d, m), r in idx.items() if d == dataset and _official(r) and r.get("test")]
        if base and official:
            rows.append((dataset, base["accuracy"], official[0][0], official[0][1]["test"]["accuracy"]))
    if not rows:
        print(f"  [건너뜀] {path.name}: baseline 과 공식 모델의 test 결과가 함께 있는 데이터셋이 없습니다. (3번·8번을 먼저 실행하세요)")
        return None
    fig, ax = plt.subplots(figsize=(max(7.5, 1.35 * len(rows) + 2), 5.2))
    xs = list(range(len(rows)))
    width = 0.38
    for offset, key, color, label in ((-width / 2, 1, MODEL_COLORS["baseline"], "baseline"),
                                      (width / 2, 3, "#2a78d6", _t("공식 모델", "Official model"))):
        values = [row[key] * 100 for row in rows]
        bars = ax.bar([x + offset for x in xs], values, width, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{_label(row[0]).replace(' · ', chr(10))}\n({row[2]})" for row in rows], fontsize=9)
    ax.set_ylim(0, 118)
    ax.set_ylabel(_t("test 정확도 (%)", "Test accuracy (%)"))
    ax.set_title(_t("baseline 대비 공식 모델 (test 정확도)", "Official model vs baseline (test accuracy)"), fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return _save(plt, fig, path)


def chart_learning_curves(plt, idx, datasets, models, path):
    shown = [d for d in datasets if any((idx.get((d, m)) or {}).get("curve") for m in models)]
    if not shown:
        print(f"  [건너뜀] {path.name}: 학습 곡선(curve)이 있는 결과가 없습니다.")
        return None
    cols = min(4, len(shown))
    rows = math.ceil(len(shown) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.3 * rows), squeeze=False, sharey=True)
    handles = {}
    for k, dataset in enumerate(shown):
        ax = axes[k // cols][k % cols]
        for model in models:
            main = idx.get((dataset, model))
            curve = (main or {}).get("curve")
            if not curve:
                continue
            color = MODEL_COLORS.get(model, "#555555")
            line, = ax.plot(curve["epoch"], curve["val_macro_f1"], color=color, linewidth=1.8, marker="o", markersize=3)
            handles[model] = line
            best = (main.get("train") or {}).get("best_epoch")
            if best in curve["epoch"]:
                ax.scatter([best], [curve["val_macro_f1"][curve["epoch"].index(best)]], marker="*", s=170, color=color,
                           edgecolor="black", linewidth=0.8, zorder=5)
        ax.set_title(_label(dataset), fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.set_xlabel(_t("에폭", "Epoch"), fontsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for k in range(len(shown), rows * cols):
        axes[k // cols][k % cols].axis("off")
    for r in range(rows):
        axes[r][0].set_ylabel("validation Macro-F1", fontsize=9)
    fig.legend(list(handles.values()), list(handles.keys()), loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=10)
    fig.text(0.5, -0.01, _t("별표 = 저장된 best 에폭 (곡선의 마지막 점은 학습이 멈춘 시점)",
                            "Star = saved best epoch (last point = where training stopped)"),
             ha="center", va="top", fontsize=9, color="#555555")
    fig.tight_layout()
    return _save(plt, fig, path)


def chart_cost_vs_accuracy(plt, idx, datasets, models, path):
    stats = {}
    for model in models:
        accs, millis, params = [], [], None
        for dataset in datasets:
            main = idx.get((dataset, model))
            if not main or not main.get("test"):
                continue
            accs.append(main["test"]["accuracy"] * 100)
            cost = main.get("cost") or {}
            if cost.get("test_inference_seconds") is not None and main["test"]["n"]:
                millis.append(cost["test_inference_seconds"] / main["test"]["n"] * 1000)
            params = cost.get("num_parameters") or params
        if accs:
            stats[model] = {"acc": statistics.mean(accs), "n": len(accs), "ms": statistics.median(millis) if millis else None,
                            "params": params}
    panels = [("params", _t("파라미터 수 (백만 개, 로그 눈금)", "Parameters (millions, log scale)"), 1e-6),
              ("ms", _t("문장당 추론 시간 (밀리초, 로그 눈금)", "Inference time per sentence (ms, log scale)"), 1.0)]
    usable = [p for p in panels if any(s.get(p[0]) for s in stats.values())]
    if not usable:
        print(f"  [건너뜀] {path.name}: 크기·속도 값이 있는 test 결과가 없습니다.")
        return None
    fig, axes = plt.subplots(1, len(usable), figsize=(6.2 * len(usable), 4.8), squeeze=False)
    for ax, (key, xlabel, scale) in zip(axes[0], usable):
        for model, s in stats.items():
            if not s.get(key):
                continue
            ax.scatter([s[key] * scale], [s["acc"]], s=150, color=MODEL_COLORS.get(model, "#555555"), edgecolor="black", zorder=3)
            ax.annotate(f"{model} ({s['n']})", (s[key] * scale, s["acc"]), textcoords="offset points", xytext=(8, 6), fontsize=10)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(_t("평균 test 정확도 (%)", "Mean test accuracy (%)"))
        ax.grid(alpha=0.3, which="both")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(_t("모델 크기·속도와 정확도", "Model size / speed vs accuracy"), fontsize=13)
    fig.text(0.5, -0.01, _t("( ) = 평균에 쓴 데이터셋 수. 데이터셋 수가 다르면 평균을 그대로 비교하기 어렵습니다.",
                            "( ) = number of datasets averaged. Averages over different datasets are not directly comparable."),
             ha="center", va="top", fontsize=9, color="#555555")
    fig.tight_layout()
    return _save(plt, fig, path)


def chart_confusion(plt, idx, datasets, path):
    items = []
    for dataset in datasets:
        official = [(m, r) for (d, m), r in idx.items() if d == dataset and _official(r) and (r.get("test") or {}).get("confusion_matrix")]
        if official:
            items.append((dataset, official[0][0], official[0][1]["test"]))
    if not items:
        print(f"  [건너뜀] {path.name}: 공식 모델의 test 혼동행렬이 없습니다. (3번·8번을 먼저 실행하세요)")
        return None
    cols = min(4, len(items))
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.7 * cols, 3.6 * rows), squeeze=False)
    for k, (dataset, model, block) in enumerate(items):
        ax = axes[k // cols][k % cols]
        matrix, labels = block["confusion_matrix"], block["labels"]
        top = max(max(row) for row in matrix) or 1
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=top)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha="center", va="center", fontsize=11, color="white" if value > top / 2 else "#222222")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(_t("예측", "Predicted"), fontsize=9)
        ax.set_ylabel(_t("실제", "Actual"), fontsize=9)
        ax.set_title(f"{_label(dataset)}\n({model})", fontsize=9)
    for k in range(len(items), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(_t("공식 모델의 test 혼동행렬 (행 = 실제, 열 = 예측)", "Official model test confusion matrices (row = actual, column = predicted)"),
                 fontsize=12)
    fig.tight_layout()
    return _save(plt, fig, path)


# ═════════════════════════ 실행 ═════════════════════════
def run(include_total_merge=False):
    """main.json 을 읽어 그래프를 만든다. 만든 파일 목록을 돌려 준다."""
    try:
        plt = _setup_matplotlib()
    except ImportError:
        print("  [오류] matplotlib 이 설치되어 있지 않습니다. 터미널에서 'uv add matplotlib' 를 한 번 실행한 뒤 다시 시도하세요.")
        return []
    results = result_files.load_all_main()
    if not include_total_merge:
        kept = [r for r in results if not common.is_recommend_excluded(r["dataset"])]
        if len(kept) != len(results):
            print(f"  (서버용 데이터셋은 그래프에서 뺍니다: {', '.join(cfg.RECOMMEND_EXCLUDED)}  / 넣으려면 --with-total-merge)")
        results = kept
    if not results:
        print("  읽을 main.json 이 없습니다. 먼저 1번(학습)을 실행하세요.")
        return []
    idx = {(r["dataset_id"], r["model_name"]): r for r in results}
    datasets, models = _datasets(results), _models(results)
    out = cfg.ARTIFACTS_ROOT / "viz" / CHART_DIR_NAME
    out.mkdir(parents=True, exist_ok=True)
    print(f"  읽은 결과: 데이터셋 {len(datasets)}개 x 모델 {len(models)}개 ({', '.join(models)})")
    jobs = [("01_heatmap_test.png", lambda p: chart_heatmap(plt, idx, datasets, models, "test", p)),
            ("02_heatmap_validation.png", lambda p: chart_heatmap(plt, idx, datasets, models, "validation", p)),
            ("03_baseline_vs_official.png", lambda p: chart_baseline_vs_official(plt, idx, datasets, p)),
            ("04_learning_curves.png", lambda p: chart_learning_curves(plt, idx, datasets, models, p)),
            ("05_cost_vs_accuracy.png", lambda p: chart_cost_vs_accuracy(plt, idx, datasets, models, p)),
            ("06_confusion_official.png", lambda p: chart_confusion(plt, idx, datasets, p))]
    made = []
    for name, job in jobs:
        try:
            path = job(out / name)
        except Exception as error:
            print(f"  [실패] {name}: {type(error).__name__}: {error}")
            continue
        if path:
            made.append(path)
            print(f"  저장: {path}")
    print(f"  그래프 {len(made)}개를 만들었습니다. 위치: {out}")
    return made


def main():
    parser = argparse.ArgumentParser(description="best_model/main.json 으로 그래프(PNG)를 만든다.")
    parser.add_argument("--with-total-merge", action="store_true", help="서버용 total_merge 데이터셋도 그래프에 포함")
    args = parser.parse_args()
    run(include_total_merge=args.with_total_merge)


if __name__ == "__main__":
    main()
