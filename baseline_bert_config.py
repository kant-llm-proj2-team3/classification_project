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

# 결과 저장 위치 (모든 모델 결과가 한 폴더에 모인다):
#   <ARTIFACTS_ROOT>/<topic>/<dataset>/<baseline | bert | roberta | koelectra>/
#       best_model/      지금까지 validation 점수가 가장 좋았던 결과 (가중치 파일 포함) + best_info.json
#       20261012_195030/ 학습을 한 번 할 때마다 만들어지는 날짜·시간 폴더 (점수·로그만, 가중치 없음)
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
BEST_DIR_NAME = "best_model"
RUN_TIME_FORMAT = "%Y%m%d_%H%M%S"                   # run 폴더 이름 (초까지)

# 실행 로그(화면에 찍히는 내용 전체)는 실행할 때마다 이 폴더에 새 파일로 저장됩니다. (학습/추천/test/길이점검)
# run 폴더 안의 train.log 와 달리, 초기화(2번)를 해도 지워지지 않습니다.
LOG_DIR = PROJECT_ROOT / "logs"

# ───────────────────────── 데이터셋 ON/OFF ─────────────────────────
# True = 사용, False = 메뉴/학습에서 제외. (데이터 폴더가 비어 있으면 자동으로 건너뜀)
# total_merge 는 서버에 줄 최종 모델을 만드는 데이터입니다. 학습·test 평가·시각화 표에는 포함되지만 모델 추천에서는 빠집니다. (아래 RECOMMEND_EXCLUDED)
TOPICS = ("inquiries", "documents")                  # 고객 문의 / 문서
DATASET_NAMES = ("basic", "basic_merge", "advanced", "advanced_merge", "total_merge")
DATASET_SWITCH = {
    "inquiries": {"basic": True, "basic_merge": True, "advanced": True, "advanced_merge": True, "total_merge": True},
    "documents": {"basic": True, "basic_merge": True, "advanced": True, "advanced_merge": True, "total_merge": True},
}
RECOMMEND_EXCLUDED = ("total_merge",)               # 모델 추천(메뉴 3번·8번)에서 빠지는 데이터셋. 이런 데이터셋은 test 를 4개 모델 모두 평가합니다.

# ───────────────────────── 데이터 파일 ─────────────────────────
# 모든 파일은 <DATA_ROOT>/<topic>/<폴더>/ 에 있고, 폴더마다 train.csv, validation.csv, test.csv, labels.json 이 같은 이름으로 들어 있습니다.
#
#   basic / advanced           : 원본                    (basic, advanced 폴더)
#   basic_merge / advanced_merge: 원본 + 추가본      (이미 합쳐진 파일이라 따로 이어 붙이지 않음)
#   total_merge                : basic_merge + advanced_merge (서버용 최종 모델 학습용)
#
# basic_add / advanced_add 폴더에는 "추가본만" 들어 있습니다. (원본과 id·문장이 겹치지 않음) 우리 코드는 읽지 않습니다.
# 추가본만으로 실험하려면 아래 DATASET_FOLDERS 에서 폴더 이름을 바꾸세요. (예: "basic_merge": "basic_add")
DATA_FOLDERS = ("basic", "basic_add", "basic_merge", "advanced", "advanced_add", "advanced_merge", "total_merge")   # 주제마다 있는 폴더
DATASET_FOLDERS = {                      # 데이터셋 이름 -> 실제 데이터 폴더
    "basic": "basic",
    "basic_merge": "basic_merge",
    "advanced": "advanced",
    "advanced_merge": "advanced_merge",
    "total_merge": "total_merge",
}
DATA_FILES = {
    "labels": "labels.json",
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}

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

# 모델별 Learning Rate (모델당 1개). 학습할 때마다 날짜 폴더가 새로 생기고, best_model 과 비교합니다.
# koelectra-small 은 작은 모델이라 더 높게 잡았습니다.
# 여러 값을 비교(튜닝)하고 싶으면 [2e-5, 5e-5] 처럼 리스트로 적으세요. 값마다 run 이 하나씩 돈다.
LEARNING_RATES = {
    "bert": 3e-5,
    "roberta": 3e-5,
    "koelectra": 1e-4,
}

# 공통 학습 설정 (Learning Rate 는 위 LEARNING_RATES 에서 모델별로 지정)
TRAINING_CONFIG = {
    # 입력 길이: 이 모델들의 최대는 512토큰. 넘는 문장은 truncation 방식으로 자른다.
    #   "right"     : 앞쪽만 남기고 뒤를 자름           (1차 실험: max_length 192 + right)
    #   "head_tail" : 앞 head_tokens 개 + 뒤쪽 나머지를 남기고 가운데를 자름
    # advanced 데이터는 라벨을 정하는 "새로 접수한 문의"가 글의 맨 앞에 있기도 하고 맨 뒤에 있기도 합니다.
    # 앞쪽만 남기면 정답 구간이 통째로 잘리는 행이 많아서 앞과 뒤를 모두 남깁니다.
    "max_length": 512,
    "truncation": "head_tail",
    "head_tokens": 256,                 # head_tail 일 때 앞쪽에서 남길 토큰 수 (뒤쪽 = max_length - 2 - head_tokens)

    # 배치: "모델에 한 번에 몇 문장을 넣을지" x "몇 묶음마다 가중치를 한 번 고칠지"
    #   실질 배치 = batch_size x gradient_accumulation_steps (아래 기본값: 4 x 2 = 8문장마다 가중치 수정)
    #   작을수록 가중치 업데이트 횟수가 늘어 적은 데이터(advanced)에 유리합니다.
    "batch_size": 4,
    "gradient_accumulation_steps": 2,

    # 에폭: 최대 횟수. 아래 중단 조건에 걸리면 그 전에 멈춥니다. (가중치는 항상 가장 좋았던 에폭 것을 저장)
    # 최종버전 epochs 옵션 저장."epochs": 20, "early_stopping_patience": 5,
    "epochs": 20,
    "early_stopping_patience": 5,       # validation Macro-F1 이 이 횟수만큼 연속으로 안 오르면 중단. 0 이면 끔
    "target_macro_f1": None,            # 숫자(예: 0.99)를 적으면 validation Macro-F1 이 그 점수에 닿는 순간 중단. None 이면 끔
    #   (목표 중단은 쉬운 데이터(basic)에서 1~2에폭만에 멈출 수 있어, 켠 때는 이 점을 감안하세요)
    # 점수 기준 조기 종료: 아래 조건이 "모두" 만족되는 에폭이 나오면 그 에폭에서 멈춘다. (값을 None 으로 두면 그 조건은 보지 않음)
    #   만점(정답률 100%) + Macro-F1 1.0 + validation loss 가 충분히 작으면, 더 학습해도 얻을 게 거의 없다는 판단.
    "stop_when_scores_reached": True,   # False 로 두면 이 기능을 끔 (아래 값은 무시됨)
    "stop_min_accuracy": 1.0,           # validation 정답 비율이 이 값 이상 (1.0 = 만점)
    "stop_min_macro_f1": 1.0,           # validation Macro-F1 이 이 값 이상
    "stop_max_val_loss": 0.01,          # validation loss 가 이 값 이하 (0.01 은 정답에 약 99% 확신하는 수준)
    "stop_min_epochs": 3,               # 점수 기준 종료가 설정한 에폭 이후부터만 작동하게 함 (멈추는 조건이 아니라 늦추는 안전장치)
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "gradient_checkpointing": True,     # GPU 메모리 절약(속도는 조금 느려짐). 메모리가 충분하면 False
    "local_files_only": True,          # True 면 이미 내려받은 모델만 사용(인터넷 없이 실행). 모델을 한 번도 내려받지 않은 컴퓨터에서는 False
    "seed": SEED,
}
DEVICE = "auto"                         # auto / cuda / cpu
