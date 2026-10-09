import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "dataset_result_data_advanced"

FIELDS = ["id", "text", "label", "group_id", "source"]
SPLITS = ["train", "validation", "test"]
TOPICS = ["documents", "inquiries"]

EXPECTED_ADDED = 150
SIMILARITY_THRESHOLD = 0.90


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []

        if (
            len(columns) != len(FIELDS)
            or set(columns) != set(FIELDS)
        ):
            raise ValueError(f"{path}: CSV 컬럼을 확인해 주세요.")

        rows = list(reader)

    if not rows:
        raise ValueError(f"{path}: 데이터가 없습니다.")

    for number, row in enumerate(rows, start=2):
        if None in row or any(
            value is None for value in row.values()
        ):
            raise ValueError(
                f"{path}: {number}행의 컬럼 수가 잘못되었습니다."
            )

    return rows


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("하였습니다", "했습니다")
    text = text.replace("하였으며", "했으며")

    return re.sub(r"[^\w]", "", text)


def find_added_file(folder, topic):
    # 아래 두 이름 중 하나로 저장하면 됩니다.
    candidates = [
        folder / "add_train.csv",
        folder / f"advanced_{topic}_add_train.csv",
    ]

    found = [
        path for path in candidates
        if path.is_file()
    ]

    if len(found) != 1:
        raise ValueError(
            f"{folder}: add_train.csv 또는 "
            f"advanced_{topic}_add_train.csv 중 "
            "하나만 넣어 주세요."
        )

    return found[0]


def check_and_prepare(topic):
    folder = ROOT / topic

    labels = json.loads(
        (folder / "labels.json").read_text(
            encoding="utf-8-sig"
        )
    )["labels"]

    if (
        not isinstance(labels, list)
        or not labels
        or not all(
            isinstance(label, str) and label.strip()
            for label in labels
        )
        or len(set(labels)) != len(labels)
    ):
        raise ValueError(
            f"[{topic}] labels.json의 labels를 확인해 주세요."
        )

    if EXPECTED_ADDED % len(labels):
        raise ValueError(
            "EXPECTED_ADDED가 라벨 수로 "
            "나누어떨어져야 합니다."
        )

    original = {
        split: read_csv(folder / f"{split}.csv")
        for split in SPLITS
    }

    added = read_csv(find_added_file(folder, topic))

    records = [
        {**row, "_split": split}
        for split, rows in original.items()
        for row in rows
    ]

    # 생성용 데이터도 중복 검사에 포함합니다.
    for split in [
        "generation_development",
        "generation_eval",
    ]:
        path = folder / f"{split}.jsonl"

        with path.open(encoding="utf-8-sig") as file:
            for number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                row = json.loads(line)

                required = [
                    "id",
                    "text",
                    "expected_label",
                    "group_id",
                    "source",
                ]

                if (
                    not isinstance(row, dict)
                    or any(
                        not isinstance(row.get(key), str)
                        for key in required
                    )
                ):
                    raise ValueError(
                        f"{path}: {number}행의 "
                        "필수 항목을 확인해 주세요."
                    )

                records.append({
                    **row,
                    "label": row["expected_label"],
                    "_split": split,
                })

    original_count = len(records)
    existing_ids = {
        row["id"] for row in records
    }
    existing_groups = {
        row["group_id"] for row in records
    }

    errors = []

    # 추가분은 기존 ID와 그룹을 재사용하지 않습니다.
    for row in added:
        if row["id"] in existing_ids:
            errors.append(
                f"기존 ID와 중복: {row['id']}"
            )

        if row["group_id"] in existing_groups:
            errors.append(
                f"기존 그룹과 겹침: {row['id']}"
            )

    records.extend(
        {**row, "_split": "train"}
        for row in added
    )

    ids = Counter(
        row["id"] for row in records
    )
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
            errors.append(
                f"유효한 글자가 없는 문장: {row_id}"
            )

        if row["label"] not in labels:
            errors.append(
                f"잘못된 라벨: {row_id}"
            )

        if (
            not row_id.strip()
            or not row["group_id"].strip()
        ):
            errors.append(
                f"ID 또는 group_id 누락: {row_id}"
            )

        if not row.get("source", "").strip():
            errors.append(
                f"source 누락: {row_id}"
            )

        groups[row["group_id"]].append(row)
        texts[normalize(row["text"])].append(row)

    # Advanced는 같은 그룹의 서로 다른 라벨을 허용합니다.
    # 같은 그룹이 여러 분할에 걸치는 것은 차단합니다.
    for group_id, rows in groups.items():
        splits = {
            row["_split"] for row in rows
        }

        if len(splits) > 1:
            errors.append(
                f"분할 간 그룹 겹침: {group_id}"
            )

    # 동일 문장은 같은 그룹 안에서도 허용하지 않습니다.
    for rows in texts.values():
        if len(rows) < 2:
            continue

        row_ids = ", ".join(
            row["id"] for row in rows
        )

        errors.append(
            f"동일 문장 중복: {row_ids}"
        )

        if len({
            row["_split"] for row in rows
        }) > 1:
            errors.append(
                f"분할 간 동일 문장: {row_ids}"
            )

        if len({
            row["label"] for row in rows
        }) > 1:
            errors.append(
                f"동일 문장의 라벨 충돌: {row_ids}"
            )

    normalized = [
        normalize(row["text"])
        for row in records
    ]

    candidates = []

    # 추가분과 기존 전체 데이터, 추가분 내부를 비교합니다.
    for right in range(original_count, len(records)):
        second = normalized[right]

        for left in range(right):
            first = normalized[left]

            if (
                not first
                or not second
                or first == second
            ):
                continue

            # 짧은 본문을 긴 문맥에 그대로 넣는 재사용 검사
            shorter, longer = sorted(
                (first, second),
                key=len,
            )

            if (
                len(shorter) >= 30
                and shorter in longer
            ):
                errors.append(
                    "본문 포함 재사용: "
                    f"{records[left]['id']}, "
                    f"{records[right]['id']}"
                )
                continue

            length_ratio = (
                2 * min(len(first), len(second))
                / (len(first) + len(second))
            )

            if length_ratio < SIMILARITY_THRESHOLD:
                continue

            matcher = SequenceMatcher(
                None,
                first,
                second,
                autojunk=False,
            )

            if (
                matcher.quick_ratio()
                < SIMILARITY_THRESHOLD
            ):
                continue

            score = matcher.ratio()

            if score >= SIMILARITY_THRESHOLD:
                candidates.append((
                    score,
                    records[left]["id"],
                    records[right]["id"],
                ))

    for score, first_id, second_id in sorted(
        candidates,
        reverse=True,
    ):
        errors.append(
            f"유사 문장 검토 필요 ({score:.3f}): "
            f"{first_id}, {second_id}"
        )

    # 중복을 자동 삭제해 수량을 줄이지 않습니다.
    if len(added) != EXPECTED_ADDED:
        errors.append(
            f"추가 데이터 수량 오류: {len(added)}행 "
            f"(필요: {EXPECTED_ADDED}행)"
        )

    label_counts = Counter(
        row["label"] for row in added
    )

    expected_per_label = (
        EXPECTED_ADDED // len(labels)
    )

    if any(
        label_counts[label] != expected_per_label
        for label in labels
    ):
        errors.append(
            f"추가분 라벨별 수량 오류: "
            f"{dict(label_counts)} "
            f"(라벨별 {expected_per_label}행 필요)"
        )

    if errors:
        unique_errors = list(
            dict.fromkeys(errors)
        )

        details = "\n".join(
            unique_errors[:30]
        )

        if len(unique_errors) > 30:
            details += (
                f"\n총 {len(unique_errors)}건 중 "
                "앞의 30건만 표시했습니다."
            )

        raise ValueError(
            f"[{topic}] 검사 실패\n{details}"
        )

    # 매번 원본 train에서 새로 합칩니다.
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

    # 두 주제 모두 검사 통과해야 저장합니다.
    for topic in TOPICS:
        try:
            prepared[topic] = check_and_prepare(topic)
        except (ValueError, KeyError, OSError) as error:
            failures.append(str(error))

    if failures:
        print("\n\n".join(failures))
        print(
            "\n결과 생성 중단: "
            "원본과 기존 결과 CSV는 변경하지 않았습니다."
        )
        raise SystemExit(1)

    # 날짜 폴더를 만들지 않습니다.
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for topic, (rows, summary) in prepared.items():
        path = OUTPUT / f"{topic}_train.csv"

        # 결과 파일만 같은 위치에 덮어써서 갱신합니다.
        with path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=FIELDS,
            )
            writer.writeheader()
            writer.writerows(rows)

        print(
            f"[{topic}] 검사 통과 | "
            f"기존 {summary['original']} + "
            f"추가 {summary['added']} = "
            f"합본 {summary['combined']}행"
        )
        print(
            f"라벨별 수량: {summary['labels']}"
        )

    print(f"\n결과 위치: {OUTPUT}")
    print(
        "원본 train/validation/test와 "
        "생성용 JSONL은 수정하지 않았습니다."
    )
    print(
        "문자 유사도 검사만으로 "
        "의미 중복 전체를 판별할 수는 없습니다."
    )


if __name__ == "__main__":
    main()