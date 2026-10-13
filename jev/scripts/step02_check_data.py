"""
JSONL의 구조, 빈 값, 라벨, 질문 정의, 상황 묶음 중복을 확인한다.
dataset 하위의 모든 주제/데이터 종류(basic, basic_add, ...)를 한 번에 검사한다.

루트 디렉토리 기준 실행 코드
uv run python ./jev/scripts/step02_check_data.py
"""

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from settings import DATA_DIR, TOPICS
from step01_read_data import read_labels, read_rows

QID = "label"
RECORD_KEYS = {"state", "questions", "expected", "tags"}
TAG_KEYS = ("id", "group", "source")


# tags(["id:...", "group:...", "source:..."])를 딕셔너리로 바꾼다.
def parse_tags(tags: list[str]) -> dict[str, str]:
    return dict(tag.split(":", 1) for tag in tags)


# 검사할 때만 유니코드와 공백을 정규화한다. 원본 JSONL은 수정하지 않는다.
def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


# 레코드 한 줄의 모양을 확인한다. 데이터가 틀리면 위치와 이유를 알려준다.
def check_record(where: str, record, labels: list[str]) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{where}: 레코드가 객체가 아닙니다.")
    if set(record) != RECORD_KEYS:
        raise ValueError(f"{where}: 필드가 {sorted(RECORD_KEYS)}와 다릅니다: {sorted(record)}")

    state = record["state"]
    if not isinstance(state, str) or not state.strip():
        raise ValueError(f"{where}: state가 비어 있습니다.")

    expected = record["expected"]
    if not isinstance(expected, dict) or set(expected) != {QID}:
        raise ValueError(f"{where}: expected에는 '{QID}' 하나만 있어야 합니다.")
    if expected[QID] not in labels:
        raise ValueError(f"{where}: 알 수 없는 라벨 {expected[QID]!r}")

    tags = record["tags"]
    if (not isinstance(tags, list) or len(tags) != len(TAG_KEYS)
            or not all(isinstance(tag, str) and ":" in tag for tag in tags)):
        raise ValueError(f"{where}: tags는 'key:value' 문자열 {len(TAG_KEYS)}개여야 합니다.")
    parsed = parse_tags(tags)
    if tuple(parsed) != TAG_KEYS:
        raise ValueError(f"{where}: tags 순서는 {list(TAG_KEYS)}여야 합니다.")
    if any(not value.strip() for value in parsed.values()):
        raise ValueError(f"{where}: tags에 빈 값이 있습니다.")


# 모든 분할의 질문 정의가 같고, 선택지가 labels.json과 같은 순서인지 확인한다.
def check_question(splits_records: dict[str, list[dict]], labels: list[str]) -> dict:
    reference = None
    for split_name, records in splits_records.items():
        for number, record in enumerate(records, start=1):
            if reference is None:
                reference = record["questions"]
            elif record["questions"] != reference:
                raise ValueError(f"{split_name}:{number} 질문 정의가 다른 행과 다릅니다.")

    if not isinstance(reference, dict) or set(reference) != {QID}:
        raise ValueError(f"questions에는 '{QID}' 하나만 있어야 합니다.")
    question = reference[QID]
    if question.get("type") != "choice":
        raise ValueError("질문 type은 'choice'여야 합니다.")
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("질문 instructions가 비어 있습니다.")
    criteria = question.get("criteria")
    if not isinstance(criteria, dict) or list(criteria) != labels:
        raise ValueError(f"선택지 {list(criteria or {})}가 labels.json 순서 {labels}와 다릅니다.")
    if any(not isinstance(value, str) or not value.strip() for value in criteria.values()):
        raise ValueError("선택지 설명이 비어 있습니다.")

    return {"instructions": instructions, "choices": list(criteria)}


# 데이터 폴더 이름으로 상황 묶음 규칙을 정한다.
# basic 계열은 묶음의 라벨이 모두 같고(same), advanced 계열은 묶음 안에 모든 라벨이 한 번씩 있다(each).
# total_merge는 두 계열이 섞여 있어 묶음마다 둘 중 하나를 만족하면 된다(either).
def group_rule_for(name: str) -> str:
    if name.startswith("advanced"):
        return "each"
    if name.startswith("total"):
        return "either"
    return "same"


# json의 구조 확인
def check_splits(splits: dict[str, list[dict]], labels: list[str], group_rule: str = "same") -> dict:
    if group_rule not in ("same", "each", "either"):
        raise ValueError(f"알 수 없는 상황 묶음 규칙: {group_rule}")
    seen = {"id": set(), "group": set(), "text": set()}
    report = {}
    for split_name, records in splits.items():
        if not records:
            raise ValueError(f"{split_name}: 데이터가 비어 있습니다.")
        for number, record in enumerate(records, start=1):
            check_record(f"{split_name}:{number}", record, labels)
        if {record["expected"][QID] for record in records} != set(labels):
            raise ValueError(f"{split_name}: 라벨 목록과 데이터가 다릅니다.")

        tags = [parse_tags(record["tags"]) for record in records]
        current = {
            "id": {tag["id"] for tag in tags},
            "group": {tag["group"] for tag in tags},
            "text": {normalize(record["state"]) for record in records},
        }
        if len(current["id"]) != len(records):
            raise ValueError(f"{split_name}: 중복된 id가 있습니다.")
        if len(current["text"]) != len(records):
            raise ValueError(f"{split_name}: 중복된 문장이 있습니다.")

        group_labels = {}
        for tag, record in zip(tags, records):
            group_labels.setdefault(tag["group"], []).append(record["expected"][QID])
        for group, values in group_labels.items():
            same = len(set(values)) == 1
            each = Counter(values) == Counter(labels)
            if group_rule == "each" and not each:
                raise ValueError(f"{split_name}: 상황 묶음 {group}에 모든 라벨이 한 번씩 있어야 합니다.")
            if group_rule == "same" and not same:
                raise ValueError(f"{split_name}: 상황 묶음 {group}에 서로 다른 라벨이 있습니다.")
            if group_rule == "either" and not (same or each):
                raise ValueError(f"{split_name}: 상황 묶음 {group}의 라벨 구성이 basic/advanced 규칙과 모두 다릅니다.")

        for key, previous in seen.items():
            if previous & current[key]:
                raise ValueError(f"{split_name}: 분할 사이에 {key} 중복이 있습니다.")
            previous.update(current[key])

        counts = Counter(record["expected"][QID] for record in records)
        report[split_name] = {
            "rows": len(records), "groups": len(current["group"]),
            "class_counts": {label: counts[label] for label in labels},
        }
    return report


# 데이터 폴더 하나를 검사한다.
def check_dataset(data_path: Path, names: tuple[str, ...]) -> dict:
    files = [data_path / "labels.json"] + [data_path / f"{name}.jsonl" for name in names]
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        raise ValueError(f"파일이 없습니다: {missing}")

    labels = read_labels(data_path / "labels.json")
    splits = {name: read_rows(data_path / f"{name}.jsonl") for name in names}
    # check_splits에서 레코드 구조를 먼저 확인한 뒤 질문 정의를 비교한다.
    split_report = check_splits(splits, labels, group_rule_for(data_path.name))
    return {"question": check_question(splits, labels), "splits": split_report}


# 파일 존재를 확인한다.
# 모든 주제/데이터 종류를 돌며 검사하고, 실패한 데이터도 이유와 함께 보고서에 남긴다.
def check_files(data_dir, names=("train", "validation")) -> dict:
    report = {}
    for topic in TOPICS:
        topic_dir = Path(data_dir) / topic
        if not topic_dir.is_dir():
            report[topic] = {"status": "error", "error": f"주제 폴더가 없습니다: {topic_dir}"}
            continue
        for data_path in sorted(path for path in topic_dir.iterdir() if path.is_dir()):
            key = f"{topic}/{data_path.name}"
            try:
                report[key] = {"status": "ok", **check_dataset(data_path, names)}
            except (ValueError, KeyError, TypeError, OSError) as error:
                report[key] = {"status": "error", "error": str(error)}
    return report


def main():
    # test는 마지막 평가 단계에서 처음 읽는다.
    report = check_files(DATA_DIR)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    failed = [key for key, value in report.items() if value["status"] != "ok"]
    if failed:
        print(f"\n검사 실패 {len(failed)}/{len(report)}개: {failed}")
        sys.exit(1)
    print(f"\n{len(report)}개 데이터셋의 train과 validation이 "
          "구조, 질문 정의, 필수 값, 분할 중복 검사를 통과했습니다.")


if __name__ == "__main__":
    main()
