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

JEV_MODEL = os.getenv("JEV_MODEL", "jev-1.13.0")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))
TIMEOUT_SECONDS = float(os.getenv("GENERATION_TIMEOUT_SECONDS", "120"))
MAX_INTPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "4000"))