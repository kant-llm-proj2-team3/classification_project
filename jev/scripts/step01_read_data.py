"""
JSONL 한 행을 파이썬 딕셔너리로 읽고 문장과 라벨을 확인한다.

루트 디렉토리 기준 실행 코드
uv run python ./jev/steps/step01_read_data.py
"""

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.common_args import common_parser
from typing import Any

from settings import *


PREVIEW_ROWS = 3


# 몇 줄만 읽어오는 함수
def read_rows(path: Path) -> list[dict[str, Any]]:
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
def read_labels(data_dir: Path) -> list[str]:
    # TODO: 파일 구조에 따른 경로 변경이 필요함
    #value = json.loads((Path(data_dir) / "labels.json").read_text(encoding="utf-8-sig"))
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
    plus = a.plus

    # TODO: 파일 구조에 따른 경로 변경이 필요함
    #rows = read_rows(advanced, topic, plus, "train.jsonl")
    #labels = read_labels(advanced, topic, plus)
    #print("읽은 파일:", DATA_DIR / "project2-advanced-data" if advanced else "project2-data" / "data")


if __name__ == "__main__":
    main()