"""
dataset에서의 csv파일을 Laya JSONL로 변환하는 스크립트

- 변환 대상: dataset하위의 train/validation/test.csv
- 입력으로만 읽음: dataset하위의 train/validation/test.csv 및 labels.json
- 무시: dataset하위의 기본jsonl파일들

사용 예(프로젝트 루트 기준):
uv run python ./jev/scripts/util/data_convert.py
"""
import argparse
import csv
import json
from pathlib import Path


QID = "label"
# 추가할 지시문
INSTRUCTIONS = {
    "inquiries": "이 고객 문의의 주된 요청은 어느 유형에 해당합니까?",
    "documents": "이 글의 목적은 어느 문서 유형에 해당합니까?"
}
DEFAULT_INSTRUCTION = "이 글은 어느 유형에 해당합니까?"
# 대상 토픽
TOPICS = ("documents", "inquiries")
# csv에서 필수로 존재해야하는 컬럼
REQUIRED_COLS = {"id", "text", "label", "group_id", "source"}
# 프로젝트 루트 디렉토리 및 디렉토리 지정
BASE_DIR = Path(__file__).resolve().parent.parent.parent
#DATASET_DIR = BASE_DIR / Path("dataset")


# csv 검사 및 읽기
def read_csv(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # 없는 컬럼
        missing = REQUIRED_COLS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: 필요한 컬럼이 없습니다: {sorted(missing)}")
        return list(reader)


# laya가 학습할 수 있는 데이터 형식에 필요한 필드를 정의하는 함수
def load_questions(labels_file: Path, topic: str):
    lj = json.loads(labels_file.read_text(encoding="utf-8"))
    labels = lj["labels"]
    criteria = {k: lj["descriptions"][k] for k in labels}
    questions = {QID: {"type": "choice",
                       "instructions": INSTRUCTIONS.get(topic, DEFAULT_INSTRUCTION),
                       "criteria": criteria}}
    return questions, criteria


# csv 한 행을 전처리
def row_to_record(r, questions):
    return {
        "state": r["text"],
        "questions": questions,
        "expected": {QID: r["label"]},
        "tags": [f"id:{r['id']}", f"group:{r['group_id']}", f"source:{r['source']}"],
    }


# 파일들 변환
def convert_files(target_file, labels_file, topic):
    questions, criteria = load_questions(labels_file, topic)
    rows = read_csv(target_file)

    # 대상 파일명.jsonl이 출력 파일 형식
    lines = []
    for i, r in enumerate(rows, start=2):
        if not r["text"].strip():
            raise ValueError(f"{target_file.name}:{i} 빈 text")
        if r["label"] not in criteria:
            raise ValueError(f"{target_file.name}:{i} 알 수 없는 라벨{r['label']!r}")
        lines.append(json.dumps(row_to_record(r, questions), ensure_ascii=False))

    out_path = target_file.with_suffix(".jsonl")
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("\n".join(lines) + "\n")
        print(f"[{topic}] {target_file.name} -> {out_path.name} ({len(rows)}줄)")


def main():
    DATA_DIR = Path(__file__).parent.parent.parent.parent / "dataset"
    documents_dir = DATA_DIR / "documents"
    inquiries_dir = DATA_DIR / "inquiries"

    for dir in documents_dir.iterdir():
        train = dir / "train.csv"
        validation = dir / "validation.csv"
        test = dir / "test.csv"
        labels = dir / "labels.json"

        convert_files(train, labels, "documents")
        convert_files(validation, labels, "documents")
        convert_files(test, labels, "documents")

    for dir in inquiries_dir.iterdir():
        train = dir / "train.csv"
        validation = dir / "validation.csv"
        test = dir / "test.csv"
        labels = dir / "labels.json"

        convert_files(train, labels, "inquiries")
        convert_files(validation, labels, "inquiries")
        convert_files(test, labels, "inquiries")


if __name__ == "__main__":
    main()