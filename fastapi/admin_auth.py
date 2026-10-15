"""조회용 /admin API의 인증. 암호를 보내지 않고 서명만 보내는 방식이다.

서버 .env의 TRACKING_ADMIN_PASSWORD가 비어 있으면 인증 없이 통과하고, 값이 있으면 모든 /admin 요청이 서명을 갖춰야 한다.
클라이언트는 요청마다 아래 세 헤더를 보낸다. 암호 자체는 네트워크로 나가지 않는다.
  X-Admin-Timestamp : 요청을 만든 시각(UNIX 초)
  X-Admin-Nonce     : 요청마다 새로 만든 무작위 문자열
  X-Admin-Signature : HMAC-SHA256(암호, "시각\\nnonce\\n메서드\\n경로\\n쿼리\\n본문 SHA-256") 의 16진수
서명에 메서드·경로·쿼리·본문이 들어가므로 중간에서 요청 내용을 바꿀 수 없고, 시각 허용 오차(60초)와 nonce 1회용 검사로
같은 요청을 다시 보내는 것도 막는다. (응답 내용까지 숨기려면 HTTPS가 필요하다.)
클라이언트 쪽 계산은 chatbot/admin_client.py의 sign()과 같은 식이다. 한쪽을 바꾸면 다른 쪽도 맞춰야 한다.
"""

import hashlib
import hmac
import os
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

from dotenv import load_dotenv
from fastapi import HTTPException, Request

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

PASSWORD_ENV = "TRACKING_ADMIN_PASSWORD"
MAX_SKEW_SECONDS = 60
MAX_NONCES = 10_000
_seen_nonces: dict[str, float] = {}      # nonce -> 만료 시각. 허용 오차 안에서 한 번 쓴 nonce는 다시 받지 않는다.


def admin_password() -> str:
    return os.getenv(PASSWORD_ENV) or ""


def auth_required() -> bool:
    return bool(admin_password())


def canonical_query(query: str) -> str:
    """쿼리 문자열을 인코딩 방식과 순서에 상관없이 같은 값이 되도록 정리한다. (클라이언트도 같은 식으로 계산한다)"""
    return urlencode(sorted(parse_qsl(query, keep_blank_values=True)))


def sign(password: str, timestamp: str, nonce: str, method: str, path: str, query: str, body: bytes) -> str:
    message = "\n".join([timestamp, nonce, method.upper(), path, query, hashlib.sha256(body).hexdigest()])
    return hmac.new(password.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def _remember_nonce(nonce: str, now: float) -> bool:
    """처음 보는 nonce면 기억하고 True. 이미 썼으면 False."""
    for key in [key for key, expires in _seen_nonces.items() if expires < now]:
        del _seen_nonces[key]
    if nonce in _seen_nonces or len(_seen_nonces) >= MAX_NONCES:
        return False
    _seen_nonces[nonce] = now + 2 * MAX_SKEW_SECONDS
    return True


async def require_admin(request: Request) -> None:
    """FastAPI 의존성. 암호가 설정돼 있지 않으면 아무것도 확인하지 않는다."""
    password = admin_password()
    if not password:
        return
    headers = request.headers
    timestamp, nonce, signature = (headers.get("x-admin-timestamp"), headers.get("x-admin-nonce"),
                                   headers.get("x-admin-signature"))
    if not (timestamp and nonce and signature):
        raise HTTPException(status_code=401, detail="인증이 필요합니다.")
    now = time.time()
    try:
        fresh = abs(now - float(timestamp)) <= MAX_SKEW_SECONDS
    except ValueError:
        fresh = False
    query = canonical_query(request.scope.get("query_string", b"").decode("latin-1"))
    expected = sign(password, timestamp, nonce, request.method, request.scope["path"], query, await request.body())
    # 서명을 먼저 확인하고(상수 시간 비교), 맞는 요청의 nonce만 기억한다. 틀린 요청이 nonce 저장소를 채우지 못하게 한다.
    if not (hmac.compare_digest(expected, signature) and fresh and len(nonce) <= 128 and _remember_nonce(nonce, now)):
        raise HTTPException(status_code=401, detail="인증에 실패했습니다.")
