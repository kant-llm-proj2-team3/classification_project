import csv
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "dataset_result_data"

FIELDS = ["id", "text", "label", "group_id", "source"]
TOPICS = ["documents", "inquiries"]
SPLITS = ["train", "validation", "test"]

LABELS = {
    "documents": ["announcement", "manual", "minutes"],
    "inquiries": ["shipping", "refund", "account"],
}

ORIGINAL_COUNTS = {
    "train": [120, 120, 120],
    "validation": [40, 40, 40],
    "test": [40, 40, 40],
}

ADDITIONAL_COUNTS = {
    "train": [100, 100, 100],
    "validation": [34, 34, 32],
    "test": [34, 32, 34],
}


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^\w]", "", text)


def read_csv(path):
    with path.open(
        encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file, strict=True)

        if reader.fieldnames != FIELDS:
            raise ValueError(f"{path}: CSV 컬럼 오류")

        rows = list(reader)

    for number, row in enumerate(rows, start=2):
        if None in row or any(
            value is None or not value.strip()
            for value in row.values()
        ):
            raise ValueError(
                f"{path}: {number}행 필수 값 오류"
            )

        if not normalize(row["text"]):
            raise ValueError(
                f"{path}: {number}행 유효한 텍스트 없음"
            )

    return rows


def input_folders():
    original = ROOT / "original"
    additional = ROOT / "additional"

    if original.is_dir() or additional.is_dir():
        if not original.is_dir() or not additional.is_dir():
            raise ValueError(
                "original과 additional 폴더가 "
                "모두 있어야 합니다."
            )

        return original, additional

    if all((ROOT / topic).is_dir() for topic in TOPICS):
        return ROOT, ROOT

    raise ValueError(
        "prepare_dataset.py 옆에 documents와 "
        "inquiries 폴더가 필요합니다."
    )


def validate():
    original_root, additional_root = input_folders()

    records = []
    prepared = {}
    summaries = []

    for topic in TOPICS:
        labels = LABELS[topic]

        for base in {original_root, additional_root}:
            actual = json.loads(
                (base / topic / "labels.json").read_text(
                    encoding="utf-8-sig"
                )
            )["labels"]

            if actual != labels:
                raise ValueError(
                    f"{topic}: labels.json 라벨 순서 오류"
                )

        for split in SPLITS:
            original = read_csv(
                original_root / topic / f"{split}.csv"
            )
            added = read_csv(
                additional_root / topic / f"add_{split}.csv"
            )

            for component, rows, expected in [
                (
                    "original",
                    original,
                    ORIGINAL_COUNTS[split],
                ),
                (
                    "additional",
                    added,
                    ADDITIONAL_COUNTS[split],
                ),
            ]:
                counts = Counter(
                    row["label"] for row in rows
                )

                if (
                    len(rows) != sum(expected)
                    or [counts[label] for label in labels]
                    != expected
                ):
                    raise ValueError(
                        f"{topic}/{split}/{component}: "
                        f"수량 또는 라벨 오류 {dict(counts)}"
                    )

                records.extend(
                    {
                        **row,
                        "_topic": topic,
                        "_split": split,
                        "_component": component,
                    }
                    for row in rows
                )

            # ID와 group_id를 변경하거나 행을 제거하지 않습니다.
            prepared[f"{topic}_{split}.csv"] = original + added

            summaries.append((
                topic,
                split,
                len(original),
                len(added),
                len(original) + len(added),
            ))

    ids = Counter(row["id"] for row in records)

    if any(count > 1 for count in ids.values()):
        raise ValueError("기존·추가 데이터의 사례 ID 중복")

    groups = defaultdict(list)
    texts = Counter()

    for row in records:
        groups[row["group_id"]].append(row)
        texts[normalize(row["text"])] += 1

    for group_id, rows in groups.items():
        # 같은 group_id가 정확히 2행 있는 것은 정상입니다.
        if len(rows) != 2:
            raise ValueError(
                f"그룹당 2행 오류: {group_id} "
                f"({len(rows)}행)"
            )

        if len({row["label"] for row in rows}) != 1:
            raise ValueError(
                f"그룹 내 라벨 충돌: {group_id}"
            )

        locations = {
            (
                row["_topic"],
                row["_split"],
                row["_component"],
            )
            for row in rows
        }

        if len(locations) != 1:
            raise ValueError(
                "그룹의 분할 또는 기존·추가 데이터 겹침: "
                f"{group_id}"
            )

    if any(count > 1 for count in texts.values()):
        raise ValueError("정규화 동일 텍스트 중복")

    manifest_path = ROOT / "additional_groups.json"

    if manifest_path.is_file():
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8-sig")
        )

        added_groups = {
            row["group_id"]
            for row in records
            if row["_component"] == "additional"
        }

        if {
            entry["group_id"] for entry in manifest
        } != added_groups:
            raise ValueError("추가 그룹 관리 기록 불일치")

        families = defaultdict(set)

        for entry in manifest:
            families[
                (entry["topic"], entry["family_id"])
            ].add(entry["split"])

        if any(
            len(splits) > 1
            for splits in families.values()
        ):
            raise ValueError("추가 상황 묶음의 분할 겹침")

    return prepared, summaries


def main():
    try:
        prepared, summaries = validate()
    except (
        ValueError,
        KeyError,
        OSError,
        TypeError,
        csv.Error,
    ) as error:
        print(f"검사 실패: {error}")
        print(
            "결과 생성 중단: 기존 데이터와 이전 결과는 "
            "변경하지 않았습니다."
        )
        raise SystemExit(1)

    # 전체 검사를 통과한 결과를 임시 폴더에 작성합니다.
    with tempfile.TemporaryDirectory(
        prefix="prepare_", dir=ROOT
    ) as temp:
        staging = Path(temp)

        for name, rows in prepared.items():
            with (staging / name).open(
                "w", encoding="utf-8-sig", newline=""
            ) as file:
                writer = csv.DictWriter(
                    file, fieldnames=FIELDS
                )
                writer.writeheader()
                writer.writerows(rows)

        OUTPUT.mkdir(parents=True, exist_ok=True)

        # 같은 경로의 결과 파일을 덮어씁니다.
        for name in prepared:
            os.replace(staging / name, OUTPUT / name)

    for topic, split, old, added, total in summaries:
        print(
            f"[{topic}/{split}] 검사 통과 | "
            f"기존 {old} + 추가 {added} = 합본 {total}행"
        )

    print(f"\n결과 위치: {OUTPUT}")
    print("기존 CSV와 추가 CSV는 수정하지 않았습니다.")
    print("재실행하면 같은 이름의 결과 파일을 덮어씁니다.")


if __name__ == "__main__":
    main()