"""
JSONL 한 행을 파이썬 딕셔너리로 읽고 문장과 라벨을 확인한다.

루트 디렉토리 기준 실행 코드
uv run python ./jev/scripts/step01_read_data.py --topic --advanced documents --extended
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from util.common_args import common_parser
from util.path_extractor import extract_path
from typing import Any

from settings import *


PREVIEW_ROWS = 3


# 파일을 읽어오는 함수
def read_rows(path: Path) -> list[dict[str, Any]]:
    print(path)
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{Path(path).name}:{number} JSON 오류: {error}") from error
    return rows


# 라벨을 읽어오는 함수
def read_labels(labels_file: Path) -> list[str]:
    value = json.loads(labels_file.read_text(encoding="utf-8-sig"))
    labels = value["labels"]
    if (not isinstance(labels, list) or not labels
            or not all(isinstance(label, str) and label.strip() for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("labels.json에는 중복 없는 라벨 문자열 목록이 필요합니다.")
    return labels


def main():
    ap = argparse.ArgumentParser(parents=[common_parser()])
    a = ap.parse_args()

    advanced = a.advanced
    topic = a.topic
    extended = a.extended

    data_path = extract_path(topic, advanced, extended)
    train, valid, test = data_path / "train.jsonl", data_path / "validation.jsonl", data_path / "test.jsonl"
    labels = read_labels(data_path / "labels.json")

    for path in [train, valid, test]:
        rows = read_rows(path)
        print("읽은 파일:", path.name)
        print("행 수:", len(rows))
        print("라벨 순서:", labels)
        for row in rows[:PREVIEW_ROWS]:
            print("\nID:", row["tags"][0])
            print("문장:", row["state"])
            print("정답:", row["expected"]["label"], "상황 묶음:", row["tags"][1])


if __name__ == "__main__":
    main()