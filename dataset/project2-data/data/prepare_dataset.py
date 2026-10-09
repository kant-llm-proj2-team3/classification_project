import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = ROOT
OUTPUT = ROOT / "dataset_result_basic"

FIELDS = ["id", "text", "label", "group_id", "source"]
SPLITS = ["train", "validation", "test"]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if set(reader.fieldnames or []) != set(FIELDS):
            raise ValueError(f"{path}: CSV 컬럼을 확인해 주세요.")

        return list(reader)


def normalize(text):
    return re.sub(r"[^\w]", "", text.lower())


def check_and_prepare(topic):
    folder = INPUT / topic

    labels = json.loads(
        (folder / "labels.json").read_text(encoding="utf-8")
    )["labels"]

    original = {
        split: read_csv(folder / f"{split}.csv")
        for split in SPLITS
    }
    added = read_csv(folder / "add_train.csv")

    # 기존 분류 데이터와 생성 데이터를 비교 대상으로 사용합니다.
    records = [
        {**row, "_split": split}
        for split, rows in original.items()
        for row in rows
    ]

    for split in ["generation_development", "generation_eval"]:
        with (folder / f"{split}.jsonl").open(
            encoding="utf-8"
        ) as file:
            for line in file:
                if not line.strip():
                    continue

                row = json.loads(line)
                records.append({
                    **row,
                    "label": row["expected_label"],
                    "_split": split,
                })

    existing_ids = {row["id"] for row in records}
    existing_groups = {row["group_id"] for row in records}
    existing_texts = {
        normalize(row["text"]) for row in records
    }

    errors = []

    for row in added:
        if row["id"] in existing_ids:
            errors.append(f"기존 ID와 중복: {row['id']}")

        if row["group_id"] in existing_groups:
            errors.append(f"기존 그룹과 겹침: {row['id']}")

        if normalize(row["text"]) in existing_texts:
            errors.append(f"기존 문장과 중복: {row['id']}")

    # 추가 데이터는 전부 학습용입니다.
    records.extend(
        {**row, "_split": "train"}
        for row in added
    )

    ids = Counter(row["id"] for row in records)
    for row_id, count in ids.items():
        if count > 1:
            errors.append(f"중복 ID: {row_id}")

    groups = defaultdict(list)
    texts = defaultdict(list)

    for row in records:
        if not row["text"].strip():
            errors.append(f"빈 문장: {row['id']}")

        if row["label"] not in labels:
            errors.append(f"잘못된 라벨: {row['id']}")

        if not row["id"].strip() or not row["group_id"].strip():
            errors.append("비어 있는 ID 또는 group_id")

        groups[row["group_id"]].append(row)
        texts[normalize(row["text"])].append(row)

    # 일반 데이터는 같은 그룹의 라벨이 같아야 합니다.
    for group_id, rows in groups.items():
        splits = {row["_split"] for row in rows}
        answers = {row["label"] for row in rows}

        if len(splits) > 1:
            errors.append(f"분할 간 그룹 겹침: {group_id}")

        if len(answers) > 1:
            errors.append(f"그룹 내 라벨 충돌: {group_id}")

    for rows in texts.values():
        splits = {row["_split"] for row in rows}
        answers = {row["label"] for row in rows}
        group_ids = {row["group_id"] for row in rows}
        row_ids = ", ".join(row["id"] for row in rows)

        if len(splits) > 1:
            errors.append(f"분할 간 동일 문장: {row_ids}")

        if len(answers) > 1:
            errors.append(f"동일 문장의 라벨 충돌: {row_ids}")

        if len(group_ids) > 1:
            errors.append(f"다른 그룹의 동일 문장: {row_ids}")

    if errors:
        details = "\n".join(dict.fromkeys(errors))
        raise ValueError(f"[{topic}] 검사 실패\n{details}")

    combined = original["train"] + added

    return combined, {
        "original": len(original["train"]),
        "added": len(added),
        "combined": len(combined),
        "labels": dict(Counter(row["label"] for row in combined)),
    }


def main():
    # 두 주제 모두 검사에 통과해야 결과를 작성합니다.
    prepared = {
        topic: check_and_prepare(topic)
        for topic in ["documents", "inquiries"]
    }

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