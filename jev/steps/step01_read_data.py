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

from settings import *


PREVIEW_ROWS = 3


def main():
    ap = argparse.ArgumentParser(parents=[common_parser()])
    a = ap.parse_args()

    advanced = a.advanced
    topic = a.topic
    plus = a.plus

    rows = read_rows(advanced, topic, plus, "train.jsonl")
    labels = read_labels(advanced, topic, plus)
    print("읽은 파일:", DATA_DIR / "project2-advanced-data" if advanced else "project2-data" / "data")


if __name__ == "__main__":
    main()