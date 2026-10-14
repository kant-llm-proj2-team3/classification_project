# -*- coding: utf-8 -*-
"""Streamlit 트레이싱 페이지가 FastAPI /admin을 부르는 클라이언트.

암호는 네트워크로 보내지 않고, 요청마다 HMAC-SHA256 서명만 헤더로 보낸다. (서버 쪽 검증은 fastapi/admin_auth.py)
서명하는 내용: 시각, 1회용 nonce, 메서드, 경로, 정리한 쿼리, 본문의 SHA-256.
서버가 암호를 요구하지 않으면(auth-info의 required가 false) 서명 없이 보낸다.
"""

import hashlib
import hmac
import json
import secrets
import time
from urllib.parse import parse_qsl, urlencode

import httpx

TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)


class AdminAuthError(Exception):
    """암호가 없거나 틀렸다 (HTTP 401)."""


def canonical_query(query):
    # 서버(admin_auth.canonical_query)와 같은 식이어야 서명이 맞는다.
    return urlencode(sorted(parse_qsl(query, keep_blank_values=True)))


def sign(password, timestamp, nonce, method, path, query, body):
    message = "\n".join([
        timestamp, nonce, method.upper(), path, query,
        hashlib.sha256(body).hexdigest(),
    ])
    return hmac.new(
        password.encode("utf-8"), message.encode("utf-8"), hashlib.sha256,
    ).hexdigest()


class AdminClient:
    def __init__(self, api_url, password=None):
        self.base = api_url.rstrip("/")
        self.password = password or ""

    def request(self, method, path, params=None, json_body=None, signed=True):
        params = {key: value for key, value in (params or {}).items() if value is not None}
        query = urlencode(params)
        body = b""

        if json_body is not None:
            body = json.dumps(json_body, ensure_ascii=False).encode("utf-8")

        headers = {"Content-Type": "application/json"} if json_body is not None else {}

        if signed and self.password:
            timestamp = str(time.time())
            nonce = secrets.token_hex(16)
            headers.update({
                "X-Admin-Timestamp": timestamp,
                "X-Admin-Nonce": nonce,
                "X-Admin-Signature": sign(
                    self.password, timestamp, nonce, method, path,
                    canonical_query(query), body,
                ),
            })

        url = f"{self.base}{path}" + (f"?{query}" if query else "")
        response = httpx.request(
            method, url, content=body or None, headers=headers, timeout=TIMEOUT,
        )

        if response.status_code == 401:
            raise AdminAuthError("암호가 필요하거나 올바르지 않습니다.")

        response.raise_for_status()
        return response

    # ── 조회 ──
    def auth_info(self):
        return self.request("GET", "/admin/auth-info", signed=False).json()

    def check(self):
        return self.request("GET", "/admin/auth-check").json()

    def list_requests(self, filters, limit=200, offset=0, sort="started_at", direction="desc"):
        params = {**filters, "limit": limit, "offset": offset, "sort": sort, "direction": direction}
        return self.request("GET", "/admin/requests", params).json()

    def stats(self, filters):
        return self.request("GET", "/admin/stats", filters).json()

    def detail(self, request_id):
        return self.request("GET", f"/admin/requests/{request_id}").json()

    def export(self, body):
        """-> (파일 이름, 파일 내용(bytes), 내보낸 행 수)"""
        response = self.request("POST", "/admin/export", json_body=body)
        disposition = response.headers.get("content-disposition", "")
        filename = disposition.split('filename="')[-1].rstrip('"') or "tracing_export"
        return filename, response.content, int(response.headers.get("x-export-rows", 0))
