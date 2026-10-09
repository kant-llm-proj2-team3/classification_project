import csv
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "dataset_result_data"

FIELDS = ["id", "text", "label", "group_id", "source"]
SPLITS = ["train", "validation", "test"]
EXPECTED_ADDED = 300
SIMILARITY_THRESHOLD = 0.90


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if set(reader.fieldnames or []) != set(FIELDS):
            raise ValueError(f"{path}: CSV 컬럼을 확인해 주세요.")

        rows = list(reader)

    for number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path}: {number}행의 컬럼 수가 잘못되었습니다.")

    return rows


def normalize(text):
    text = text.lower()
    text = text.replace("하였습니다", "했습니다")
    text = text.replace("하였으며", "했으며")
    return re.sub(r"[^\w]", "", text)


def check_and_prepare(topic):
    folder = ROOT / topic
    labels = json.loads(
        (folder / "labels.json").read_text(encoding="utf-8-sig")
    )["labels"]

    original = {
        split: read_csv(folder / f"{split}.csv")
        for split in SPLITS
    }
    added = read_csv(folder / "add_train.csv")

    records = [
        {**row, "_split": split}
        for split, rows in original.items()
        for row in rows
    ]

    # 생성용 데이터도 추가분과의 중복 검사에 사용합니다.
    for split in ["generation_development", "generation_eval"]:
        path = folder / f"{split}.jsonl"

        with path.open(encoding="utf-8-sig") as file:
            for line in file:
                if not line.strip():
                    continue

                row = json.loads(line)
                records.append({
                    **row,
                    "label": row["expected_label"],
                    "_split": split,
                })

    original_count = len(records)
    existing_ids = {row["id"] for row in records}
    existing_groups = {row["group_id"] for row in records}
    errors = []

    # 추가 데이터의 ID와 그룹은 기존 데이터와 겹치면 안 됩니다.
    for row in added:
        if row["id"] in existing_ids:
            errors.append(f"기존 ID와 중복: {row['id']}")

        if row["group_id"] in existing_groups:
            errors.append(f"기존 그룹과 겹침: {row['id']}")

    records.extend(
        {**row, "_split": "train"}
        for row in added
    )

    ids = Counter(row["id"] for row in records)
    groups = defaultdict(list)
    texts = defaultdict(list)

    for row_id, count in ids.items():
        if count > 1:
            errors.append(f"중복 ID: {row_id}")

    for row in records:
        row_id = row["id"]

        if not row["text"].strip():
            errors.append(f"빈 문장: {row_id}")

        if not normalize(row["text"]):
            errors.append(f"유효한 글자가 없는 문장: {row_id}")

        if row["label"] not in labels:
            errors.append(f"잘못된 라벨: {row_id}")

        if not row_id.strip() or not row["group_id"].strip():
            errors.append(f"ID 또는 group_id 누락: {row_id}")

        if not row.get("source", "").strip():
            errors.append(f"source 누락: {row_id}")

        groups[row["group_id"]].append(row)
        texts[normalize(row["text"])].append(row)

    # Basic은 같은 그룹의 라벨이 같아야 합니다.
    for group_id, rows in groups.items():
        splits = {row["_split"] for row in rows}
        answers = {row["label"] for row in rows}

        if len(splits) > 1:
            errors.append(f"분할 간 그룹 겹침: {group_id}")

        if len(answers) > 1:
            errors.append(f"그룹 내 라벨 충돌: {group_id}")

    # 같은 그룹 안에서도 동일 문장을 허용하지 않습니다.
    for rows in texts.values():
        if len(rows) < 2:
            continue

        row_ids = ", ".join(row["id"] for row in rows)
        errors.append(f"동일 문장 중복: {row_ids}")

        if len({row["_split"] for row in rows}) > 1:
            errors.append(f"분할 간 동일 문장: {row_ids}")

        if len({row["label"] for row in rows}) > 1:
            errors.append(f"동일 문장의 라벨 충돌: {row_ids}")

    # 추가분과 기존 전체 데이터, 추가분 내부를 비교합니다.
    # 문자 유사도 검사이며 의미 중복 전체를 판별하지는 않습니다.
    normalized = [normalize(row["text"]) for row in records]
    candidates = []

    for right in range(original_count, len(records)):
        for left in range(right):
            first = normalized[left]
            second = normalized[right]

            if not first or not second or first == second:
                continue

            length_ratio = (
                2 * min(len(first), len(second))
                / (len(first) + len(second))
            )
            if length_ratio < SIMILARITY_THRESHOLD:
                continue

            matcher = SequenceMatcher(
                None, first, second, autojunk=False
            )

            if matcher.quick_ratio() < SIMILARITY_THRESHOLD:
                continue

            score = matcher.ratio()
            if score >= SIMILARITY_THRESHOLD:
                candidates.append((
                    score,
                    records[left]["id"],
                    records[right]["id"],
                ))

    for score, first_id, second_id in sorted(
        candidates, reverse=True
    ):
        errors.append(
            f"유사 문장 검토 필요 ({score:.3f}): "
            f"{first_id}, {second_id}"
        )

    # 중복을 제거해서 수량을 줄이는 처리는 하지 않습니다.
    if len(added) != EXPECTED_ADDED:
        errors.append(
            f"추가 데이터 수량 오류: {len(added)}행 "
            f"(필요: {EXPECTED_ADDED}행)"
        )

    label_counts = Counter(row["label"] for row in added)
    expected_per_label = EXPECTED_ADDED // len(labels)

    if any(
        label_counts[label] != expected_per_label
        for label in labels
    ):
        errors.append(
            f"추가분 라벨별 수량 오류: {dict(label_counts)} "
            f"(라벨별 {expected_per_label}행 필요)"
        )

    if errors:
        unique_errors = list(dict.fromkeys(errors))
        details = "\n".join(unique_errors[:30])

        if len(unique_errors) > 30:
            details += (
                f"\n총 {len(unique_errors)}건 중 "
                "앞의 30건만 표시했습니다."
            )

        raise ValueError(f"[{topic}] 검사 실패\n{details}")

    combined = original["train"] + added

    return combined, {
        "original": len(original["train"]),
        "added": len(added),
        "combined": len(combined),
        "labels": dict(Counter(
            row["label"] for row in combined
        )),
    }


def main():
    prepared = {}
    failures = []

    # 두 주제를 모두 검사한 뒤 결과를 작성합니다.
    for topic in ["documents", "inquiries"]:
        try:
            prepared[topic] = check_and_prepare(topic)
        except (ValueError, KeyError, OSError) as error:
            failures.append(str(error))

    if failures:
        print("\n\n".join(failures))
        print("\n결과 생성 중단: 기존 결과 CSV는 변경하지 않았습니다.")
        print("기존 결과는 수정·재검증 전 학습에 사용하지 마세요.")
        raise SystemExit(1)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    for topic, (rows, summary) in prepared.items():
        path = OUTPUT / f"{topic}_train.csv"

        with path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        print(
            f"[{topic}] 검사 통과 | "
            f"기존 {summary['original']} + "
            f"추가 {summary['added']} = "
            f"합본 {summary['combined']}행"
        )
        print(f"라벨별 수량: {summary['labels']}")

    print(f"\n결과 위치: {OUTPUT}")


if __name__ == "__main__":
    main()