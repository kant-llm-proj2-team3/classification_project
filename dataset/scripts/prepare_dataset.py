"""Validate basic/advanced sources and rebuild their *_merge CSV files."""

import argparse
import csv
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FIELDS = ["id", "text", "label", "group_id", "source"]
TOPICS = ("documents", "inquiries")
SPLITS = ("train", "validation", "test")

LABELS = {
    "documents": ["announcement", "manual", "minutes"],
    "inquiries": ["shipping", "refund", "account"],
}

COUNTS = {
    "basic": {
        "original": {
            "train": [120, 120, 120],
            "validation": [40, 40, 40],
            "test": [40, 40, 40],
        },
        "additional": {
            "train": [100, 100, 100],
            "validation": [34, 34, 32],
            "test": [34, 32, 34],
        },
    },
    "advanced": {
        "original": {
            "train": [60, 60, 60],
            "validation": [20, 20, 20],
            "test": [20, 20, 20],
        },
        "additional": {
            "train": [50, 50, 50],
            "validation": [17, 17, 17],
            "test": [17, 17, 17],
        },
    },
}


def normalize(text):
    return re.sub(
        r"[^\w]",
        "",
        unicodedata.normalize("NFKC", text).casefold(),
    )


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, strict=True)

        if reader.fieldnames != FIELDS:
            raise ValueError(
                f"{path}: CSV columns must be {FIELDS}"
            )

        rows = list(reader)

    for number, row in enumerate(rows, 2):
        if None in row or any(
            value is None or not value.strip()
            for value in row.values()
        ):
            raise ValueError(
                f"{path}:{number}: missing required value"
            )

        if not normalize(row["text"]):
            raise ValueError(
                f"{path}:{number}: invalid text"
            )

    return rows


def validate(root, variant):
    prepared, records, summaries = {}, [], []

    for topic in TOPICS:
        base = root / topic / variant
        added_dir = root / topic / f"{variant}_add"
        merged_dir = root / topic / f"{variant}_merge"

        for directory in (base, added_dir, merged_dir):
            metadata = json.loads(
                (directory / "labels.json").read_text(
                    encoding="utf-8-sig"
                )
            )

            if metadata["labels"] != LABELS[topic]:
                raise ValueError(
                    f"{directory}: label order mismatch"
                )

            if set(metadata["descriptions"]) != set(LABELS[topic]):
                raise ValueError(
                    f"{directory}: label descriptions mismatch"
                )

        for split in SPLITS:
            original = read_csv(base / f"{split}.csv")
            added = read_csv(added_dir / f"{split}.csv")

            for component, rows in (
                ("original", original),
                ("additional", added),
            ):
                expected = COUNTS[variant][component][split]
                counts = Counter(row["label"] for row in rows)

                if (
                    len(rows) != sum(expected)
                    or [counts[label] for label in LABELS[topic]]
                    != expected
                ):
                    raise ValueError(
                        f"{variant}/{topic}/{split}/{component}: "
                        f"label/count mismatch {dict(counts)}"
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

            prepared[merged_dir / f"{split}.csv"] = original + added

            summaries.append(
                (
                    topic,
                    variant,
                    split,
                    len(original),
                    len(added),
                )
            )

    ids = Counter(row["id"] for row in records)

    if any(count > 1 for count in ids.values()):
        raise ValueError(
            f"{variant}: duplicate case ID"
        )

    groups = defaultdict(list)
    texts = Counter()

    for row in records:
        groups[row["group_id"]].append(row)
        texts[normalize(row["text"])] += 1

    for group_id, rows in groups.items():
        expected_size = 2 if variant == "basic" else 3

        if len(rows) != expected_size:
            raise ValueError(
                f"{variant}: {group_id} "
                f"must have {expected_size} rows"
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
                f"{variant}: group overlaps topics, splits "
                f"or source components: {group_id}"
            )

        if (
            variant == "basic"
            and len({row["label"] for row in rows}) != 1
        ):
            raise ValueError(
                f"{variant}: group labels disagree: {group_id}"
            )

        if (
            variant == "advanced"
            and Counter(row["label"] for row in rows)
            != Counter(LABELS[rows[0]["_topic"]])
        ):
            raise ValueError(
                f"{variant}: group must contain "
                f"each label once: {group_id}"
            )

    if any(count > 1 for count in texts.values()):
        raise ValueError(
            f"{variant}: duplicate normalized text"
        )

    return prepared, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--variant",
        choices=("basic", "advanced", "all"),
        default="all",
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate sources and existing combined CSVs without writing",
    )

    args = parser.parse_args()
    root = args.root.resolve()

    variants = (
        ("basic", "advanced")
        if args.variant == "all"
        else (args.variant,)
    )

    try:
        prepared, summaries = {}, []

        for variant in variants:
            files, summary = validate(root, variant)
            prepared.update(files)
            summaries.extend(summary)

        if args.check_only:
            for path, expected in prepared.items():
                if read_csv(path) != expected:
                    raise ValueError(
                        f"{path}: combined CSV differs "
                        f"from original + additional"
                    )

        else:
            with tempfile.TemporaryDirectory(
                prefix="prepare_",
                dir=root,
            ) as temp:
                replacements = []

                for index, (path, rows) in enumerate(
                    prepared.items()
                ):
                    staged = Path(temp) / f"{index}.csv"

                    with staged.open(
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

                    replacements.append((staged, path))

                for staged, path in replacements:
                    os.replace(staged, path)

    except (
        ValueError,
        KeyError,
        OSError,
        TypeError,
        csv.Error,
    ) as error:
        parser.exit(
            1,
            f"Validation failed: {error}\n",
        )

    for topic, variant, split, original, added in summaries:
        print(
            f"[{topic}/{variant}_merge/{split}] "
            f"{original} + {added} = {original + added} rows"
        )

    print(
        "Checked only."
        if args.check_only
        else "Combined CSVs written."
    )


if __name__ == "__main__":
    main()