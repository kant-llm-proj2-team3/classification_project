"""그래프 만들기 (baseline/steps/make_charts.py).

artifacts/<주제>/<데이터셋>/<모델>/best_model/main.json 을 읽어서 PNG 그래프를 만든다. (CSV 표는 만들지 않는다)
저장 위치: artifacts/viz/charts/   (초기화(2번)는 이 폴더를 지우지 않는다)

실행 (프로젝트 루트 project_2 에서):
    uv run python baseline/steps/make_charts.py
    uv run python baseline/steps/make_charts.py --with-total-merge      # 서버용 total_merge 도 그래프에 포함
메뉴 7번(그래프 만들기), 8번(마무리 마지막 단계)에서도 같은 코드가 실행된다.
처음 한 번 matplotlib 설치가 필요하다:  uv add matplotlib

만드는 그래프
  01_heatmap_test.png          데이터셋 x 모델 test 정확도 히트맵 (정답 수 표시, 줄당 1개 굵은 주황 테두리 = test 점수가 가장 높은 모델,
                               RED_BOX_DATASETS 줄은 빨간 네모, 주제가 바뀌는 곳에는 빈 칸)
  02_heatmap_validation.png    같은 표의 validation 버전 (주황 테두리 = validation 점수가 가장 높은 모델)
  03_baseline_vs_official.png  baseline_1(TF-IDF)·baseline_2(JEV) 대비 공식 모델의 test 정확도 막대 (막대 아래에 공식 모델 이름)
  04_learning_curves.png       데이터셋별 validation Macro-F1 학습 곡선 (별표 = 저장된 best 에폭, 왼쪽에 주제 이름)
  05_cost_vs_accuracy.png      모델 크기·추론 속도와 평균 test 정확도
베스트(공식) 모델은 줄마다 그 그림의 점수가 가장 높은 모델이다. baseline 과 jev 는 후보에서 뺀다. 동점이면 validation loss -> 학습 시간 순.
03 번의 공식 모델도 test 점수가 가장 높은 모델이다. (recommendation.json 은 쓰지 않는다)
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
GAP_HEIGHT = 0.5                         # 히트맵에서 inquiries / documents 사이 빈 칸 높이 (1 = 한 줄)
RED_BOX_DATASETS = ("advanced_merge",)   # 히트맵에서 줄 전체를 빨간 네모로 감쌀 데이터셋 (total_merge 와 비교할 줄)
JEV_BAR_COLOR = "#c4c4c4"                # 03번 그래프 baseline_2 (JEV) 막대 색 (baseline 회색보다 밝은 회색)


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


def _best_model(idx, dataset, split):
    """한 데이터셋에서 split("test" / "validation") 점수가 가장 높은 모델 이름. 줄마다 1개.
    후보는 baseline 과 참고용 모델(jev)을 뺀 나머지. 순위 규칙은 모델 추천(common.rank_key)과 같다:
    Macro-F1 높은 순 -> 동점이면 validation loss 낮은 순 -> 학습 시간 짧은 순. (test 는 loss 를 저장하지 않아 validation loss 를 쓴다)"""
    skip = {common.BASELINE_KEY, *cfg.RECOMMEND_REFERENCE_ONLY}
    ranked = []
    for (d, model), main in idx.items():
        block = main.get(split)
        if d != dataset or model in skip or not block:
            continue
        loss = (main.get("validation") or {}).get("loss")
        seconds = (main.get("cost") or {}).get("train_seconds")
        ranked.append(((-round(block["macro_f1"], 6), float("inf") if loss is None else loss,
                        float("inf") if seconds is None else seconds), model))
    return min(ranked)[1] if ranked else None


def _save(plt, fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ═════════════════════════ 그래프 ═════════════════════════
def chart_heatmap(plt, idx, datasets, models, split, path):
    import numpy as np
    # 주제(inquiries / documents)가 바뀌는 곳에는 반 줄 높이의 빈 칸(None)을 끼워 간격을 둔다.
    layout = []
    for dataset in datasets:
        if layout and dataset.split("/", 1)[0] != [d for d in layout if d][-1].split("/", 1)[0]:
            layout.append(None)
        layout.append(dataset)
    heights = [GAP_HEIGHT if d is None else 1.0 for d in layout]
    edges = np.concatenate([[-0.5], -0.5 + np.cumsum(heights)])       # 줄 경계의 y 좌표
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(layout))]
    values = np.full((len(layout), len(models)), np.nan)
    for i, dataset in enumerate(layout):
        if dataset is None:
            continue
        for j, model in enumerate(models):
            block = (idx.get((dataset, model)) or {}).get(split)
            if block:
                values[i, j] = block["accuracy"]
    if np.isnan(values).all():
        print(f"  [건너뜀] {path.name}: {split} 결과가 하나도 없습니다.")
        return None
    best_models = {d: _best_model(idx, d, split) for d in layout if d}      # 줄마다 이 그림의 점수(test / validation)가 가장 높은 모델
    fig, ax = plt.subplots(figsize=(1.9 * len(models) + 3.6, 0.66 * sum(heights) + 2.0))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#eeeeee")
    ax.pcolormesh(np.arange(len(models) + 1) - 0.5, edges, np.ma.masked_invalid(values), cmap=cmap, vmin=0.6, vmax=1.0)
    for i, dataset in enumerate(layout):
        if dataset is None:
            ax.add_patch(plt.Rectangle((-0.5, edges[i]), len(models), heights[i], facecolor="white", edgecolor="none", zorder=3))   # 빈 칸
            continue
        for j, model in enumerate(models):
            main = idx.get((dataset, model))
            block = (main or {}).get(split)
            if not block:
                ax.text(j, centers[i], "-", ha="center", va="center", fontsize=11, color="#999999")
                continue
            dark = (block["accuracy"] - 0.6) / 0.4 > 0.55
            ax.text(j, centers[i], f"{block['accuracy']:.1%}\n{block['correct']}/{block['n']}", ha="center", va="center",
                    fontsize=10, color="white" if dark else "#222222")
            if model == best_models[dataset]:
                ax.add_patch(plt.Rectangle((j - 0.46, centers[i] - 0.46), 0.92, 0.92, fill=False, edgecolor="#ff8c00", linewidth=3.2))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks(centers)
    ax.set_yticklabels([_label(d) if d else "" for d in layout], fontsize=10)
    ax.set_xticks([x - 0.5 for x in range(len(models) + 1)], minor=True)
    ax.set_yticks(list(edges), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # 빨간 네모(RED_BOX_DATASETS 줄): 행 이름 글자 위치를 알아야 해서 한 번 그려 본 뒤 추가한다.
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(edges[-1], -0.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = ax.transData.inverted()
    label_boxes = [tick.get_window_extent(renderer) for tick in ax.get_yticklabels()]
    x_right = len(models) - 0.5 + 0.04
    for i, dataset in enumerate(layout):
        if dataset and dataset.split("/", 1)[1] in RED_BOX_DATASETS:
            x_left = inverse.transform((label_boxes[i].x0, label_boxes[i].y0))[0] - 0.12
            ax.add_patch(plt.Rectangle((x_left, edges[i]), x_right - x_left, heights[i], fill=False, edgecolor="#e02020",
                                       linewidth=3.0, clip_on=False, zorder=10))
    name = _t("test 정확도", "Test accuracy") if split == "test" else _t("validation 정확도 (모델 선정에 사용)", "Validation accuracy (used for selection)")
    fig.suptitle(f"{name}  ({_t('정답 수/전체', 'correct/total')})", fontsize=13, y=0.995)
    fig.text(0.5, 0.0, _t("굵은 주황 테두리 = 공식 모델    회색 = 결과 없음",
                          "Orange border = official model    Gray = no result"),
             ha="center", va="top", fontsize=9, color="#555555")
    return _save(plt, fig, path)


def chart_baseline_vs_official(plt, idx, datasets, path):
    rows = []          # (데이터셋, baseline_1 TF-IDF 정확도, 공식 모델 이름, 공식 모델 정확도, baseline_2 JEV 정확도 또는 None)
    for dataset in datasets:
        base = (idx.get((dataset, "baseline")) or {}).get("test")
        jev = (idx.get((dataset, "jev")) or {}).get("test")
        best = _best_model(idx, dataset, "test")          # test 점수가 가장 높은 모델 (baseline·jev 제외)
        if base and best:
            rows.append((dataset, base["accuracy"], best, idx[(dataset, best)]["test"]["accuracy"],
                         jev["accuracy"] if jev else None))
    if not rows:
        print(f"  [건너뜀] {path.name}: baseline 과 공식 모델의 test 결과가 함께 있는 데이터셋이 없습니다. (3번·8번을 먼저 실행하세요)")
        return None
    fig, ax = plt.subplots(figsize=(max(8.5, 1.7 * len(rows) + 2), 5.4))
    xs = list(range(len(rows)))
    width = 0.26
    series = ((-width, 1, MODEL_COLORS["baseline"], "baseline_1 (TF-IDF)"),      # 회색 = 항상 TF-IDF baseline
              (0.0, 4, JEV_BAR_COLOR, "baseline_2 (JEV)"),                        # 밝은 회색 = JEV (결과가 없는 데이터셋은 막대 없음)
              (width, 3, "#2a78d6", _t("공식 모델", "Official model")))           # 파란색 = 공식 모델 (이름은 막대 아래)
    for offset, key, color, label in series:
        points = [(x, row[key] * 100) for x, row in zip(xs, rows) if row[key] is not None]
        if not points:
            continue
        bars = ax.bar([x + offset for x, _ in points], [value for _, value in points], width, color=color, label=label)
        for bar, (_, value) in zip(bars, points):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{_label(row[0]).replace(' · ', chr(10))}\n({row[2]})" for row in rows], fontsize=9)
    for tick, row in zip(ax.get_xticklabels(), rows):       # RED_BOX_DATASETS (advanced_merge) 줄은 글씨를 굵고 진하게
        if row[0].split("/", 1)[1] in RED_BOX_DATASETS:
            tick.set_fontweight("bold")
            tick.set_color("#e02020")
    ax.set_ylim(0, 118)
    ax.set_ylabel(_t("test 정확도 (%)", "Test accuracy (%)"))
    ax.set_title(_t("baseline 대비 공식 모델 (test 정확도)", "Official model vs baselines (test accuracy)"), fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return _save(plt, fig, path)


def chart_learning_curves(plt, idx, datasets, models, path):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    shown = [d for d in datasets if any((idx.get((d, m)) or {}).get("curve") for m in models)]
    if not shown:
        print(f"  [건너뜀] {path.name}: 학습 곡선(curve)이 있는 결과가 없습니다.")
        return None
    # 주제(inquiries / documents)마다 한 줄, 데이터셋은 왼쪽부터
    groups = [[d for d in shown if d.split("/", 1)[0] == topic] for topic in cfg.TOPICS]
    groups = [group for group in groups if group]
    cols, rows = max(len(group) for group in groups), len(groups)
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.3 * rows), squeeze=False, sharey=True)
    handles = {}
    for r, group in enumerate(groups):
        for c, dataset in enumerate(group):
            ax = axes[r][c]
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
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        for c in range(len(group), cols):
            axes[r][c].axis("off")
    for r in range(rows):
        axes[r][0].set_ylabel("Macro-F1", fontsize=9)
    star = Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#555555", markeredgecolor="black", markersize=12)
    fig.legend(list(handles.values()) + [star],
               list(handles.keys()) + [_t("best 에폭의 validation 점수", "validation score at best epoch")],
               loc="upper center", ncol=len(handles) + 1, frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=10)
    fig.text(0.5, -0.01, _t("곡선 = validation Macro-F1 (마지막 점은 학습이 멈춘 시점)",
                            "Curve = validation Macro-F1 (last point = where training stopped)"),
             ha="center", va="top", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0.035, 0, 1, 0.93))      # 위쪽 7% 는 범례 자리, 왼쪽 3.5% 는 주제(inquiries/documents) 라벨 자리
    for r, group in enumerate(groups):              # 줄마다 왼쪽에 주제 이름
        box = axes[r][0].get_position()
        fig.text(0.012, (box.y0 + box.y1) / 2, group[0].split("/", 1)[0], rotation=90, ha="center", va="center",
                 fontsize=15, fontweight="bold")
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
            ("05_cost_vs_accuracy.png", lambda p: chart_cost_vs_accuracy(plt, idx, datasets, models, p))]
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
