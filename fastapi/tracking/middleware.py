"""모든 HTTP 요청의 시작·종료·상태코드를 기록하는 ASGI 미들웨어.

  - 요청마다 서버가 request_id를 발급하고 응답 헤더 X-Request-ID로 돌려준다.
    클라이언트가 보낸 X-Request-ID는 형식을 확인한 뒤 client_request_id로 따로 남긴다. (같은 값이 반복돼도 기록이 섞이지 않는다)
  - 요청 본문은 읽지 않는다. 입력은 /chat이 검증을 통과한 뒤에 RequestRecord에 채운다.
  - 422 같은 검증 실패, 4xx/5xx 응답, 처리되지 않은 예외, 클라이언트 연결 끊김(취소)도 기록한다.
BaseHTTPMiddleware 대신 순수 ASGI로 만들어 ContextVar가 안전하게 전달되고 응답을 감싸지 않는다.
"""

import asyncio
import logging
import re
import time
from collections.abc import Callable

from starlette.datastructures import MutableHeaders

from tracking import context
from tracking.context import RequestRecord, bind_request, unbind_request

logger = logging.getLogger("uvicorn.error")

REQUEST_ID_HEADER = "X-Request-ID"
CLIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
# 연결 확인 등으로 자주 불리는 경로는 기록하지 않는다. (request_id 헤더는 그대로 붙인다)
SKIP_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json"})
MAX_DETAIL_BYTES = 1000


class RequestTrackingMiddleware:
    def __init__(self, app, sink: Callable[[dict], None], skip_paths=SKIP_PATHS):
        self.app, self.sink, self.skip_paths = app, sink, skip_paths

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope["headers"]}
        client_id = headers.get("x-request-id")
        client = scope.get("client")
        record = RequestRecord(
            client_request_id=client_id if client_id and CLIENT_ID_PATTERN.match(client_id) else None,
            method=scope["method"], path=scope["path"], client_host=client[0] if client else None,
            user_agent=(headers.get("user-agent") or "")[:200] or None)
        token = bind_request(record)
        started = time.perf_counter()
        detail = bytearray()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                record.status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = record.request_id
            elif message["type"] == "http.response.body" and (record.status_code or 0) >= 400:
                if len(detail) < MAX_DETAIL_BYTES:
                    detail.extend(message.get("body", b"")[:MAX_DETAIL_BYTES - len(detail)])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except asyncio.CancelledError:
            record.outcome = "cancelled"
            raise
        except Exception as error:
            record.set_error(error)
            if record.status_code is None:      # 응답이 나가기 전에 터졌다 = 서버가 500을 돌려준다
                record.status_code = 500
            raise
        finally:
            record.duration_ms = (time.perf_counter() - started) * 1000
            record.error_detail = detail.decode("utf-8", errors="replace") or None
            if record.error_detail and not context.STORE_TEXT:
                # 422 응답 본문에는 FastAPI가 입력값(원문)을 그대로 넣어 주므로, 원문을 저장하지 않을 때는 본문을 남기지 않는다.
                record.error_detail = "(원문 저장이 꺼져 있어 응답 본문을 생략했습니다)"
            record.finish_outcome()
            unbind_request(token)
            if record.path not in self.skip_paths:
                self._emit(record)

    def _emit(self, record: RequestRecord) -> None:
        try:
            self.sink(record.to_dict())
        except Exception:
            logger.exception("[tracking] 요청 기록을 내보내지 못했습니다. request_id=%s", record.request_id)


def log_request(record: dict) -> None:
    """요청 하나를 로그 한 줄로 요약한다. (저장소 sink가 붙기 전까지의 확인용)"""
    logger.info("[request] id=%s client_id=%s %s %s -> %s %.0fms outcome=%s dataset=%s%s", record["request_id"],
                record["client_request_id"], record["method"], record["path"], record["status_code"],
                record["duration_ms"], record["outcome"], record["dataset"],
                f" error={record['error_type']}" if record["error_type"] else "")
