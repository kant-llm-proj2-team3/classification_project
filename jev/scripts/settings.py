"""
jev 디렉토리의 steps에서 사용할 경로 및 설정 파일
secret key는 .env에 저장
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# 프로젝트 루트 경로
ROOT = Path(__file__).resolve().parent.parent.parent
# 루트의 .env 로드
load_dotenv(ROOT / ".env", override=False)

# laya모델 설치 및 경로 받기
MODEL_ROOT = Path(snapshot_download("convaiinnovations/laya", allow_patterns=["multilingual/*"]))
MODEL_PATH = MODEL_ROOT / "multilingual"
# 주제 목록
TOPICS = ("documents", "inquiries")

# 데이터 디렉토리
DATA_DIR = ROOT / "dataset"
# 출력 디렉토리
OUTPUT_ROOT = ROOT / "artifacts"
# 실험용 임시 출력 디렉토리 (git 추적 제외)
TMP_DIR = ROOT / "jev" / "tmp"

# laya 학습 설정 (에폭, 배치, 조기 종료, seed는 baseline_bert_config.TRAINING_CONFIG를 함께 사용)
LAYA_MODEL_ID = "convaiinnovations/laya (multilingual)"
LAYA_CONFIG = {
    # 질문 + 선택지 + 문장 전체의 토큰 한도. 데이터에서 가장 긴 문장(약 1,230토큰)도 잘리지 않는 값
    "max_len": 1536,
    "head_max_len": None,       # None이면 체크포인트 기본값(256) 사용
    "encoder_lr": 2.5e-5,       # 인코더(mmBERT) 학습률
    "head_lr": 1e-4,            # 판단 헤드 학습률
    "loss": "rlcd",             # laya 기본 학습 목표. "soft-ce"로 바꾸면 cross-entropy만 사용
    "shuffle_options": False,   # True면 에폭마다 선택지 순서를 섞어 학습(순서 의존성 완화)
    # 평가 배치. 학습 직후 평가는 optimizer 상태가 GPU에 남아 있어 메모리가 부족해지기 쉬우므로 작게 둔다.
    "eval_batch_size": 2,
}

JEV_MODEL = os.getenv("JEV_MODEL", "jev-1.13.0")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))
TIMEOUT_SECONDS = float(os.getenv("GENERATION_TIMEOUT_SECONDS", "120"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "4000"))