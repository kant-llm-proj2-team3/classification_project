"""
jev 디렉토리의 CLM 계열(step05)에서 쓰는 경로와 설정 파일

인코더(Qwen3-8B 4비트 GGUF를 llama-server로 띄운 임베딩 서버)는 프로젝트 밖에 두고, 코드가 필요할 때 찾아서 쓰거나 직접 띄운다.
프로젝트(artifacts)에는 파인튜닝한 판단 헤드(head.safetensors)와, 그 헤드가 어떤 인코더로 학습됐는지 적은 기록(encoder.json)만 저장한다.
경로는 .env에 값이 있으면 그 값을, 없으면 아래 기본값을 쓴다. (.env.example 참고)

settings.py(laya)는 import할 때 모델을 내려받기 때문에, CLM은 이 파일을 따로 쓴다.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트 경로
ROOT = Path(__file__).resolve().parent.parent.parent
# 루트의 .env 로드 (빈 값은 아래에서 기본값으로 대체된다)
load_dotenv(ROOT / ".env", override=False)

# ───────────────────────── 프로젝트 밖의 외부 도구 ─────────────────────────
TOOLS_DIR = Path(os.getenv("CLM_TOOLS_DIR") or Path.home() / "tools")
# CLM 공식 저장소(https://github.com/Contrastive-LM/CLM) clone 위치. 헤드 구조, 토큰화 규칙, 질문->텍스트 규칙을 그대로 가져다 쓴다.
CLM_REPO = Path(os.getenv("CLM_REPO") or TOOLS_DIR / "CLM")
CLM_REPO_COMMIT = "d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6"   # 이 커밋으로 검증했다. 다르면 경고만 한다.
# llama-server 실행 파일. 비어 있으면 PATH, 그다음 TOOLS_DIR/llama.cpp 순서로 찾는다.
LLAMA_SERVER = os.getenv("CLM_LLAMA_SERVER") or ""

# 임시 출력 디렉토리 (git 추적 제외)
TMP_DIR = ROOT / "jev" / "tmp"
EMBED_CACHE_DIR = TMP_DIR / "clm_embeddings"     # 문장 임베딩 캐시. 인코더(GGUF 파일, 길이)마다 폴더가 따로 생긴다.
SERVER_LOG_DIR = TMP_DIR / "logs"                # llama-server 로그

# ───────────────────────── 인코더 (고정된 Qwen3-8B 4비트) ─────────────────────────
MAX_LEN = 2048              # 인코더 서버의 컨텍스트(-c)이자 한 문장의 토큰 한도. 넘으면 앞부분을 잘라 낸다(질문은 끝에 있어서 남는다)
ENCODER = {
    "gguf_repo": "Qwen/Qwen3-8B-GGUF",          # GGUF는 HF 캐시에 받는다 (프로젝트 안에 두지 않는다)
    "gguf_file": "Qwen3-8B-Q4_K_M.gguf",        # 4비트 양자화 고정. 바꾸면 임베딩 캐시와 헤드를 모두 다시 만들어야 한다.
    "tokenizer": "Qwen/Qwen3-8B",               # 토큰화 규칙(Recipe)용. 토크나이저 파일만 쓴다.
    "served_name": "qwen3-8b",                  # 서버 별칭. CLM 클라이언트 기본값과 같다.
    "url": os.getenv("CLM_ENCODER_URL") or "http://127.0.0.1:8090",
    "pooling": "last",                          # 공식 헤드는 마지막 토큰 임베딩에 맞춰져 있다.
    "ctx": MAX_LEN,
    "batch": 2048,
    # -ub(물리 배치)는 한 번에 계산하는 토큰 수다. 글이 이보다 길면 조각으로 나눠 처리하며, Qwen3처럼 앞에서부터 보는 모델은 결과가 거의 같다.
    # (1,215토큰 글을 2048과 비교하면 코사인 1024: 0.99986, 512: 0.99978. 속도는 같고 VRAM은 512에서 약 0.3GB 줄어든다.)
    # 기본 2048은 캐시에 쌓은 임베딩과 같은 값이 나오게 하는 값이다. VRAM이 부족하면 .env의 CLM_UBATCH로 줄일 수 있다.
    "ubatch": int(os.getenv("CLM_UBATCH") or 2048),
    "parallel": 1,
    # GPU에 올릴 레이어 수(Qwen3-8B는 36개). VRAM이 부족하면 .env의 CLM_GPU_LAYERS를 줄인다. (예: 8GB GPU에서 안 올라가면 30)
    "gpu_layers": int(os.getenv("CLM_GPU_LAYERS") or 99),
    "start_timeout": 600,                       # 서버를 직접 띄울 때 준비될 때까지 기다리는 시간(초)
    "request_chunk": 32,                        # 임베딩 요청 한 번에 보내는 문장 수. 한 묶음마다 캐시 파일에 저장한다.
}
CLM_MODEL_ID = "CLM-v0.1-8B head + Qwen3-8B Q4_K_M"

# 출발점이 되는 공식 판단 헤드 (zero-shot 평가와 파인튜닝의 초기값)
BASE_HEAD = {"repo": "Contrastive-LM/CLM-v0.1-8B", "file": "CLM_v0.1-8B.pt"}

# ───────────────────────── 헤드 학습 ─────────────────────────
# 에폭, 조기 종료, 점수 기준 종료, seed는 baseline_bert_config.TRAINING_CONFIG를 함께 사용한다. (laya와 같음)
CLM_CONFIG = {
    "loss": "infonce",          # 같은 배치 안의 선택지 텍스트를 서로 negative로 쓰는 양방향 InfoNCE (CLM 공식 choice 기본값)
    "lr": 5e-4,                 # 판단 헤드(state/action) 학습률
    # InfoNCE 배치. CLM 공식 기본값은 256이지만 우리 데이터는 train이 180~990행이라 에폭당 업데이트가 1~4번뿐이어서 덜 학습된다.
    # 32로 낮추면 validation Macro-F1 평균이 0.849 -> 0.954가 되고 10개 데이터셋 모두 좋아졌다. (선택지 텍스트가 3개뿐이라 배치를 줄여도 negative는 그대로다)
    "batch": 32,
    "weight_decay": 0.0,        # CLM 공식 기본값. 공통 설정의 0.01은 쓰지 않는다.
    "warmup_ratio": 0.1,        # OneCycleLR의 pct_start
    "grad_clip": 1.0,
    # 인코더가 GPU 메모리를 거의 다 쓰므로 헤드(약 2천만 파라미터)는 CPU에서 학습한다. 360문항 기준 학습 10초 내외.
    "head_device": "cpu",
}
