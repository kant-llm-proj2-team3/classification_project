"""
dataset에서의 csv파일을 Laya JSONL로 변환하는 스크립트

- 변환 대상: dataset/project2-*/data/<topic>/{train,validation,test}.csv
- 입력으로만 읽음: dataset/project2-*/data/<topic>/labels.json
- 무시: generation_*.jsonl

사용 예:
uv run python data_convert.py --advanced true
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
DATASET_DIR = BASE_DIR / Path("dataset")
NORMAL_DATASET_DIR = DATASET_DIR / "project2-data"
ADVANCED_DATASET_DIR = DATASET_DIR / "project2-advanced-data"


# csv 검사 및 읽기
def read_csv(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # 없는 컬럼
        missing = REQUIRED_COLS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: 필요한 컬럼이 없습니다: {sorted(missing)}")
        return list(reader)


# 토픽 유추 함수
def guess_topic(topic_dir: Path, file: Path):
    if topic_dir.name in TOPICS:
        return topic_dir.name
    for t in TOPICS:
        if t in file.name:
            return t
    return None


# laya가 학습할 수 있는 데이터 형식에 필요한 필드를 정의하는 함수
def load_questions(topic_dir: Path, topic: str):
    lj = json.loads((topic_dir / "labels.json").read_text(encoding="utf-8"))
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
def convert_files(target_files, label_dirs):
    # 각 요소에 대해서 따로 처리
    for target_chunk in target_files:
        # 파일들의 경로(출력 경로로 사용)
        dir = target_chunk["dir"]
        # 파일들 추출
        files = target_chunk["files"]
        for file in files:
            # .csv가 아닌 경우 무시
            if file.suffix != ".csv":
                continue
            # 주제 알아내기
            topic = guess_topic(dir, file)
            if topic is None:
                print(f"[스킵] 토픽 판별 불가: {file.name}")
                continue
            if not file.is_file():
                print(f"[스킵] 파일 없음: {file}")
                continue

            questions, criteria = load_questions(label_dirs[topic], topic)
            rows = read_csv(file)

            # 각 파일명.jsonl이 출력 파일의 형식
            lines = []
            for i, r in enumerate(rows, start=2):
                if not r["text"].strip():
                    raise ValueError(f"{file.name}:{i} 빈 text")
                if r["label"] not in criteria:
                    raise ValueError(f"{file.name}:{i} 알 수 없는 라벨{r['label']!r}")
                lines.append(json.dumps(row_to_record(r, questions), ensure_ascii=False))

            out_path = file.with_suffix(".jsonl")
            with open(out_path, "w", encoding="utf-8", newline="\n") as out:
                out.write("\n".join(lines) + "\n")
                print(f"[{topic}] {file.name} -> {out_path.name} ({len(rows)}줄)")


def main():
    # 실행 파라미터 처리
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--advanced", default="false")
    a = ap.parse_args()

    # 심화 파일 유무 확인
    advanced = a.advanced == "true"
    target_dataset_dir = ADVANCED_DATASET_DIR if advanced else NORMAL_DATASET_DIR
    # 처리 대상 파일 목록
    target_files = []

    # documents의 train, validation, test 추가
    documents_dir = target_dataset_dir / "data" / "documents"
    documents_train = documents_dir / "train.csv"
    documents_validation = documents_dir / "validation.csv"
    documents_test = documents_dir / "test.csv"
    target_files.append({
        "dir": documents_dir,
        "files": [documents_train, documents_validation, documents_test],
    })

    # inquiries의 train, validation, test 추가
    inquiries_dir = target_dataset_dir / "data" / "inquiries"
    inquiries_train = inquiries_dir / "train.csv"
    inquiries_validation = inquiries_dir / "validation.csv"
    inquiries_test = inquiries_dir / "test.csv"
    target_files.append({
        "dir": inquiries_dir,
        "files": [inquiries_train, inquiries_validation, inquiries_test],
    })

    # dataset_result_data의 모든 데이터 추가
    # 기본/심화 여부에 따라 추가 데이터 폴더 선택
    plus_folder = (
        "dataset_result_data_advanced"
        if advanced
        else "dataset_result_data"
    )
    plus_dir = target_dataset_dir / "data" / plus_folder

    # 폴더가 존재하면 CSV 파일을 변환 대상에 추가
    if plus_dir.is_dir():
        plus_files = sorted(plus_dir.glob("*.csv"))
        print(f"[추가 데이터] {plus_dir} (CSV {len(plus_files)}개)")
    else:
        plus_files = []
        print(f"[스킵] 추가 데이터 폴더 없음: {plus_dir}")

    target_files.append({
        "dir": plus_dir,
        "files": plus_files,
    })

    # 타겟 파일 처리
    label_dirs = {"documents": documents_dir, "inquiries": inquiries_dir}
    convert_files(target_files, label_dirs)


if __name__ == "__main__":
    main()