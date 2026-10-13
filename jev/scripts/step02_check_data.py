"""
JSONL의 구조, 빈 값, 라벨, 질문 정의, 상황 묶음 중복을 확인한다.

uv run python ./jev/steps/step02_check_data.py
"""

import json
import unicodedata
from collections import Counter


# 레코드 한 줄의 모양을 확인한다. 데이터가 틀리면 위치와 이유를 알려준다.
# TODO
def check_record(where: str, record, labels: list[str]) -> None:
    pass


# 모든 분할의 질문 정의가 같고, 선택지가 labels.json과 같은 순서인지 확인한다.
# TODO
def check_question(splits_records: dict[str, list[dict]], labels: list[str]) -> dict:
    pass


# json의 구조 확인
# TODO
def check_splits(splits: dict[str, list[dict]], labels: list[str]) -> dict:
    pass


# 파일 존재를 확인한다.
# TODO
def check_files(data_dir, names=("train", "validation")) -> dict:
    pass


def main():
    report = check_files(DATA_DIR)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("train과 validation의 구조, 질문, 정의, 필수 값, 분할 중복 검사를 통과했습니다.")


if __name__ == "__main__":
    main()