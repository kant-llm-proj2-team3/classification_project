"""프로젝트 공통 설정 (project_2/baseline_bert_config.py).

TF-IDF(baseline)와 사전학습 모델(bert / roberta / koelectra) 코드가 모두 이 파일을 읽습니다.
실험 조건을 바꾸고 싶으면 이 파일만 수정합니다.

  1. DATASET_SWITCH  : 8개 데이터셋(주제 2 x 종류 4) 사용 ON/OFF
  2. LEARNING_RATES  : 모델별 Learning Rate (모델당 1개)
  3. TRAINING_CONFIG : 에폭, 배치, 입력 길이 등 공통 학습 설정
"""
from pathlib import Path

# ───────────────────────────── 경로 ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent       # project_2 (이 파일이 있는 폴더)
BASELINE_DIR = PROJECT_ROOT / "baseline"
BERT_DIR = PROJECT_ROOT / "bert"
DATA_ROOT = PROJECT_ROOT / "dataset"

# 결과 저장 위치: <artifacts>/<topic>/<dataset>/<model>/...
BASELINE_ARTIFACTS = BASELINE_DIR / "artifacts"      # tfidf 결과 + 모델 간 비교/추천 결과
BERT_ARTIFACTS = BERT_DIR / "artifacts"              # bert / roberta / koelectra 결과
ARTIFACT_ROOTS = {
    "tfidf": BASELINE_ARTIFACTS,
    "bert": BERT_ARTIFACTS,
    "roberta": BERT_ARTIFACTS,
    "koelectra": BERT_ARTIFACTS,
}

# ───────────────────────── 데이터셋 ON/OFF ─────────────────────────
# True = 사용, False = 메뉴/학습에서 제외.
# plus 데이터셋은 아직 작업 중일 수 있어 기본값을 False 로 두었습니다. 준비되면 해당 줄만 True 로 바꾸세요.
TOPICS = ("inquiries", "documents")                  # 고객 문의 / 문서
DATASET_NAMES = ("basic", "basic_plus", "advanced", "advanced_plus")
DATASET_SWITCH = {
    "inquiries": {"basic": True, "basic_plus": True, "advanced": True, "advanced_plus": True},
    "documents": {"basic": True, "basic_plus": True, "advanced": True, "advanced_plus": True},
}

# ───────────────────────── 데이터 파일 ─────────────────────────
# 모든 파일은 <root>/<topic>/ 한 폴더 안에 있습니다. (basic 과 plus 가 같은 폴더를 쓰고 파일명만 다름)
#
#   basic / advanced           : train.csv, validation.csv, test.csv
#   basic_plus / advanced_plus : 위 파일 + add_train.csv, add_validation.csv, add_test.csv
#
# add_*.csv 는 기존 파일에 "추가되는 행"만 들어 있습니다. (기존 파일과 id·문장이 겹치지 않음)
# PLUS_MERGE_WITH_ORIGINAL = True  : plus = 기존 + 추가  (예: train 360 + 300 = 660행)  <- 기본
# PLUS_MERGE_WITH_ORIGINAL = False : plus = 추가 파일만 단독 사용 (예: train 300행)
DATASET_SOURCES = {
    "basic": {"root": DATA_ROOT / "project2-data" / "data", "plus": False},
    "basic_plus": {"root": DATA_ROOT / "project2-data" / "data", "plus": True},
    "advanced": {"root": DATA_ROOT / "project2-advanced-data" / "data", "plus": False},
    "advanced_plus": {"root": DATA_ROOT / "project2-advanced-data" / "data", "plus": True},
}
DATA_FILES = {
    "labels": "labels.json",
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}
PLUS_FILES = {
    "train": "add_train.csv",
    "validation": "add_validation.csv",
    "test": "add_test.csv",
}
PLUS_MERGE_WITH_ORIGINAL = True

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
ID_COLUMN = "id"
GROUP_COLUMN = "group_id"
SOURCE_COLUMN = "source"
REQUIRED_COLUMNS = (ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN, GROUP_COLUMN, SOURCE_COLUMN)

# 라벨 순서가 곧 숫자 id(0, 1, 2)입니다. labels.json 의 순서와 같아야 하며, 다르면 실행을 멈춥니다.
LABELS = {
    "inquiries": ["shipping", "refund", "account"],
    "documents": ["announcement", "manual", "minutes"],
}
LABEL2ID = {topic: {label: i for i, label in enumerate(labels)} for topic, labels in LABELS.items()}
ID2LABEL = {topic: {i: label for i, label in enumerate(labels)} for topic, labels in LABELS.items()}
NUM_LABELS = {topic: len(labels) for topic, labels in LABELS.items()}

SEED = 42
METRICS = ("accuracy", "macro_f1")
SELECTION_METRIC = "macro_f1"   # 모델/LR 선택 기준 (validation 에서만 사용, 동점이면 val loss -> 학습시간)

# ───────────────────── TF-IDF + 로지스틱 회귀 (baseline) ─────────────────────
TFIDF_CONFIG = {
    "analyzer": "char",
    "ngram_range": (2, 5),
    "min_df": 1,
    "sublinear_tf": True,
    "max_features": None,
}
LOGISTIC_REGRESSION_CONFIG = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": SEED,
}

# ───────────────────────── 사전학습 분류 모델 ─────────────────────────
PRETRAINED_MODELS = {
    "bert": {"model_id": "klue/bert-base"},
    "roberta": {"model_id": "klue/roberta-base"},
    "koelectra": {"model_id": "monologg/koelectra-small-v3-discriminator"},
}

# 모델별 Learning Rate (모델당 1개). 결과는 lr3e5/ 같은 run 폴더에 저장됩니다.
# koelectra-small 은 작은 모델이라 더 높게 잡았습니다.
# 여러 값을 비교하고 싶으면 [2e-5, 5e-5] 처럼 리스트로 적어도 동작합니다.
LEARNING_RATES = {
    "bert": 3e-5,
    "roberta": 3e-5,
    "koelectra": 1e-4,
}

# 공통 학습 설정 (Learning Rate 는 위 LEARNING_RATES 에서 모델별로 지정)
TRAINING_CONFIG = {
    "max_length": 192,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,   # 실제 배치 = 4 x 4 = 16
    "epochs": 5,                        # 학습 행 수가 적어 실습 키트(3)보다 늘렸습니다
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "gradient_checkpointing": True,     # GPU 메모리 절약(속도는 조금 느려짐). 메모리가 충분하면 False
    "local_files_only": False,          # True 면 이미 내려받은 모델만 사용(인터넷 없이 실행)
    "seed": SEED,
}
DEVICE = "auto"                         # auto / cuda / cpu
