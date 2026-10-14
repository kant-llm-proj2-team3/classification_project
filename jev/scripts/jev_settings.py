"""
jev(TypeSafe의 System One 모델) 평가(step06)에서 쓰는 설정 파일

jev는 학습하지 않고 TypeSafe API(https://docs.typesafe.ai)를 불러서 분류 결과만 얻는 모델이다.
laya, CLM 같은 로컬 System One 모델의 baseline으로 쓰고, 결과는 다른 모델과 같은 형식으로 artifacts에 저장한다.
API 키는 .env의 JEV_API_KEY에 적는다. (SDK가 쓰는 TYPESAFE_API_KEY도 읽는다) 키는 로그와 결과 파일에 남기지 않는다.

settings.py(laya)는 import할 때 모델을 내려받기 때문에, jev는 이 파일을 따로 쓴다.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트 경로
ROOT = Path(__file__).resolve().parent.parent.parent
# 루트의 .env 로드 (빈 값은 아래에서 기본값으로 대체된다)
load_dotenv(ROOT / ".env", override=False)

JEV_KEY = "jev"             # artifacts/<topic>/<dataset>/<여기>/ 폴더 이름이자 모델 이름. baseline_bert_config.EXTERNAL_MODELS의 키와 같다.

# ───────────────────────── API ─────────────────────────
API_BASE_URL = (os.getenv("JEV_BASE_URL") or os.getenv("TYPESAFE_BASE_URL") or "https://api.typesafe.ai").rstrip("/")
ENDPOINT = "/v1/systemone"
# 별칭(jev-latest)이 아니라 버전 ID를 쓴다. 별칭은 새 버전이 나오면 가리키는 모델이 바뀌어서 결과를 다시 만들 수 없게 된다.
MODEL = os.getenv("JEV_MODEL") or "jev-1.13.0"


def api_key() -> str:
    """API 키. 없으면 안내와 함께 실패한다. 이 함수의 반환값은 화면, 로그, 결과 파일에 쓰지 않는다."""
    key = (os.getenv("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("API 키가 없습니다. .env의 JEV_API_KEY에 TypeSafe API 키를 적으세요. "
                           "(키는 https://console.typesafe.ai/keys 에서 만든다)")
    return key


# ───────────────────────── 경로 ─────────────────────────
TMP_DIR = ROOT / "jev" / "tmp"                      # git 추적 제외 (.gitignore)
RESPONSE_CACHE_DIR = TMP_DIR / "jev_responses"      # API 응답 캐시. 같은 문장은 데이터셋이 달라도 한 번만 보낸다.

# ───────────────────────── 호출 방식 ─────────────────────────
JEV_CONFIG = {
    "concurrency": int(os.getenv("JEV_CONCURRENCY") or 8),    # 동시에 보내는 요청 수. 문서의 한도는 초당 80요청이다.
    "timeout": 60.0,                # 요청 하나의 응답을 기다리는 시간(초)
    # 재시도: 문서의 SDK 기본값(408, 429, 5xx를 재시도, 0.5초부터 두 배씩 늘려 최대 5초, 지터 25%, Retry-After 존중)을 따르되
    # 몇백 건을 한 번에 보내는 작업이라 횟수와 최대 대기만 더 넉넉하게 잡았다.
    "max_retries": 5,
    "backoff_initial": 0.5,
    "backoff_max": 8.0,
    "backoff_jitter": 0.25,
    "retry_after_cap": 60.0,        # 서버가 Retry-After로 요청한 대기가 이보다 길어도 이만큼만 기다린다
    # API가 알려 주는 확률은 반올림되어 0.0이 나올 수 있다. 정답 확률이 이보다 작으면 이 값으로 바꿔서 loss(교차 엔트로피)를 계산한다.
    "loss_floor": 1e-6,
    # 문서(models)의 가격: 입력 토큰 100만 개에 $0.042, 출력 토큰은 무료. 실제 사용량은 응답의 usage로 센다.
    "price_per_million_input_tokens_usd": 0.042,
    "dry_run_tokens_per_char": 1.0,  # --dry-run이 토큰 수를 어림잡는 비율. 한국어는 보통 이보다 적게 나와서 비용을 넉넉하게 어림잡는다.
}
