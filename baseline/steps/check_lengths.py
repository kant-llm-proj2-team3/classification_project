"""CSV 파일별 입력 길이(토큰 수) 점검 (baseline/steps/check_lengths.py).

모델 입력은 토큰 수가 max_length 를 넘으면 잘린다. 이 스크립트는 모델별 토크나이저로
CSV 파일마다 몇 행이 192토큰 / 512토큰을 넘는지 센다. (192 = 1차 실험의 max_length, 512 = 모델 최대)

출력
  1) 파일별: 행 수, 토큰 수 중앙값/최대, 192·512 초과 행 수 (모델별)
  2) 데이터셋별: 학습에 실제로 쓰이는 파일을 합친 기준으로, 현재 설정의 max_length 를 넘는 행 수
  3) artifacts/input_length_report.csv 로 저장, 화면 출력은 logs/ 에도 저장

실행 (프로젝트 루트 project_2 에서):
    uv run python baseline/steps/check_lengths.py
보통은 run_menu.py 의 5번 메뉴에서 실행됩니다. 토크나이저만 쓰므로 학습보다 훨씬 빠릅니다(수 초).
"""
import statistics
import sys
from pathlib import Path

# 설정 파일(baseline_bert_config.py)은 프로젝트 루트(project_2)에 있다. (이 파일에서 두 단계 위)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import baseline_bert_config as cfg  # noqa: E402
import common  # noqa: E402

LIMITS = (192, 512)


def csv_files():
    """점검할 CSV 목록: (표시 이름, 경로). 주제 x 데이터 폴더(원본·추가본·합본 6개) x 파일 3종."""
    files = []
    for topic in cfg.TOPICS:
        for folder in cfg.DATA_FOLDERS:
            for split in ("train", "validation", "test"):
                name = cfg.DATA_FILES[split]
                files.append((f"{topic}/{folder}/{name}", Path(cfg.DATA_ROOT) / topic / folder / name))
    return files


def load_tokenizers():
    from transformers import AutoTokenizer
    tokenizers = {}
    for key, spec in cfg.PRETRAINED_MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(
            spec["model_id"], local_files_only=cfg.TRAINING_CONFIG["local_files_only"])
        warned = getattr(tokenizer, "deprecation_warnings", None)   # "길이 초과" 경고 끄기
        if isinstance(warned, dict):
            warned["sequence-length-is-longer-than-the-specified-maximum"] = True
        tokenizers[key] = tokenizer
    return tokenizers


def token_lengths(rows, tokenizer):
    texts = [row[cfg.TEXT_COLUMN] for row in rows]
    return [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]   # [CLS], [SEP] 포함


def over(lengths, limit):
    return sum(1 for x in lengths if x > limit)


def report(tokenizers=None):
    tokenizers = tokenizers or load_tokenizers()
    keys = list(tokenizers)
    max_length = cfg.TRAINING_CONFIG["max_length"]
    lengths, csv_rows = {}, []

    print(f"[파일별] 모델 입력 길이 (토큰 수, [CLS]/[SEP] 포함)   >{LIMITS[0]} / >{LIMITS[1]} = 그 길이를 넘는 행 수")
    print(f"  {'파일':<44}{'행':>5}" + "".join(f" | {k + ' 중앙/최대  >192 >512':<26}" for k in keys))
    for label, path in csv_files():
        if not path.exists():
            print(f"  {label:<44}  (파일 없음)")
            continue
        rows = common.read_rows(path)
        lengths[path] = {k: token_lengths(rows, tokenizers[k]) for k in keys}
        cells = ""
        for k in keys:
            values = lengths[path][k]
            cells += (f" | {int(statistics.median(values)):>8}/{max(values):<6}"
                      f"{over(values, LIMITS[0]):>4} {over(values, LIMITS[1]):>4}  ")
            csv_rows.append({
                "file": label, "rows": len(rows), "model": k,
                "median_tokens": int(statistics.median(values)), "max_tokens": max(values),
                f"over_{LIMITS[0]}": over(values, LIMITS[0]), f"over_{LIMITS[1]}": over(values, LIMITS[1]),
                "over_current_max_length": over(values, max_length)})
        print(f"  {label:<44}{len(rows):>5}{cells}")

    print(f"\n[데이터셋별] 학습에 실제로 쓰이는 파일을 합친 기준 - 현재 설정 max_length {max_length} 를 넘는 행 수")
    print(f"  ({cfg.TRAINING_CONFIG['truncation']} 방식이면 이 행들은 가운데가 잘리고, 앞/뒤는 남습니다)")
    print(f"  {'데이터셋':<26}{'분할':<12}{'행':>6}" + "".join(f"{k:>12}" for k in keys))
    for topic, dataset in common.all_pairs():
        paths = common.dataset_paths(topic, dataset)
        for split in ("train", "validation", "test"):
            files = paths[split]
            if any(f not in lengths for f in files):
                print(f"  {topic + '/' + dataset:<26}{split:<12}  (파일 없음)")
                continue
            total = sum(len(lengths[f][keys[0]]) for f in files)
            counts = "".join(f"{sum(over(lengths[f][k], max_length) for f in files):>12}" for k in keys)
            print(f"  {topic + '/' + dataset:<26}{split:<12}{total:>6}{counts}")

    out = cfg.ARTIFACTS_ROOT / "input_length_report.csv"
    fields = ["file", "rows", "model", "median_tokens", "max_tokens",
              f"over_{LIMITS[0]}", f"over_{LIMITS[1]}", "over_current_max_length"]
    common.write_csv(out, fields, csv_rows)
    print(f"\n저장: {out}")


def main():
    with common.session_log("lengths"):
        report()


if __name__ == "__main__":
    main()
