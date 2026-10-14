"""
jev(TypeSafe System One API) 클라이언트

  JevClient.run(items) : 질문 요청 여러 건을 동시에 보내고 결과를 돌려준다. 재시도, 응답 검증, 사용량 집계를 한다.
  ResponseCache        : 응답을 디스크에 쌓아 같은 요청은 다시 보내지 않는다. 작업이 중간에 끊겨도 끝난 요청은 남는다.

요청 형식은 문서(https://docs.typesafe.ai/api)의 POST /v1/systemone 그대로다.
  {"state": 문장, "model": 모델 ID, "questions": {"label": {"type": "choice", "instructions": 질문, "criteria": {라벨: 설명}}}}
응답의 answers[label]에서 choice, probabilities, confidence를 읽고, usage에서 입력/출력 토큰 수를 읽는다.

API 키는 이 파일 밖으로 나가지 않는다. 오류 문장, 로그, 캐시 어디에도 키나 요청 헤더를 쓰지 않는다.
"""

import concurrent.futures
import hashlib
import json
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

import jev_settings as S


# ═════════════════════════ 오류 ═════════════════════════
class JevApiError(RuntimeError):
    """API 호출이 끝내 실패했다. status는 HTTP 상태 코드(연결 실패면 None)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class JevAuthError(JevApiError):
    """키가 없거나 잘못됨(401, 403). 다른 요청도 모두 실패하므로 바로 멈춘다."""


class JevRequestError(JevApiError):
    """요청이 거부됨(400, 422 등). 재시도해도 같은 결과라서 재시도하지 않는다."""


class JevAborted(JevApiError):
    """다른 요청이 인증 오류로 실패해서 보내지 않고 건너뛴 요청. 진짜 오류(JevAuthError)는 따로 올라오므로 무시해도 된다."""


# ═════════════════════════ 결과 · 캐시 ═════════════════════════
@dataclass
class Answer:
    choice: str
    probabilities: dict          # {라벨: 확률}. API가 돌려준 값 그대로
    confidence: float            # API가 계산한 확신도 (확률이 한 곳에 몰릴수록 1에 가깝다)
    input_tokens: int
    output_tokens: int
    latency: float               # 성공한 요청 하나의 응답 시간(초). 재시도로 기다린 시간은 넣지 않는다.
    model: str                   # 응답이 알려 준 모델 버전 (예: jev-1.13.0)
    cached: bool = False


class ResponseCache:
    """요청(sha1) -> Answer. 끝에 덧붙이기만 하는 JSONL이라 중간에 끊겨도 앞부분은 남는다. 문장 원문은 저장하지 않는다."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: dict[str, Answer] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self._rows[row["key"]] = Answer(
                        row["choice"], row["probabilities"], row["confidence"], row["input_tokens"],
                        row["output_tokens"], row["latency"], row["model"], cached=True)
                except (ValueError, KeyError):
                    continue         # 쓰다 만 줄은 버린다. 같은 키가 또 나오면 뒤의 것이 이긴다.

    def __len__(self) -> int:
        return len(self._rows)

    def get(self, key: str) -> Answer | None:
        return self._rows.get(key)

    def put(self, key: str, answer: Answer) -> None:
        row = {"key": key, "model": answer.model, "choice": answer.choice, "probabilities": answer.probabilities,
               "confidence": answer.confidence, "input_tokens": answer.input_tokens,
               "output_tokens": answer.output_tokens, "latency": answer.latency,
               "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._rows[key] = Answer(answer.choice, answer.probabilities, answer.confidence, answer.input_tokens,
                                     answer.output_tokens, answer.latency, answer.model, cached=True)


def cache_path(model: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in model)
    return S.RESPONSE_CACHE_DIR / f"{safe}.jsonl"


def build_payload(model: str, state: str, questions: dict) -> dict:
    """POST /v1/systemone의 요청 본문. API 키 없이도 만들 수 있다. (--dry-run과 캐시 키 계산에 쓴다)"""
    return {"state": state, "model": model, "questions": questions}


def request_key(payload: dict) -> str:
    """요청 내용의 해시. 선택지 순서도 들어가서, 순서를 바꾸면 다른 요청으로 센다. (모델은 첫 번째 선택지로 쏠리는 경향이 있다)"""
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ═════════════════════════ 클라이언트 ═════════════════════════
class JevClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None,
                 concurrency: int | None = None, cache: ResponseCache | None = None):
        cfg = S.JEV_CONFIG
        self.model = model or S.MODEL
        self.base_url = (base_url or S.API_BASE_URL).rstrip("/")
        self.concurrency = concurrency or cfg["concurrency"]
        self.cache = cache if cache is not None else ResponseCache(cache_path(self.model))
        self._sleep = time.sleep                      # 시험에서 기다림을 건너뛰려고 바꿀 수 있다
        self._abort = threading.Event()               # 인증 오류가 나면 켜서, 대기 중인 요청이 더 나가지 않게 한다
        timeout = httpx.Timeout(cfg["timeout"], connect=10.0)
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout,
                                  headers={"Authorization": f"Bearer {api_key or S.api_key()}",
                                           "Content-Type": "application/json"},
                                  limits=httpx.Limits(max_connections=self.concurrency * 2))

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "JevClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── 호출 하나 (재시도 포함) ──
    def _delay(self, attempt: int, response: httpx.Response | None) -> float:
        cfg = S.JEV_CONFIG
        if response is not None:                     # 서버가 기다리라고 알려 주면 그 시간을 따른다
            try:
                if "retry-after-ms" in response.headers:
                    return min(float(response.headers["retry-after-ms"]) / 1000, cfg["retry_after_cap"])
                if "retry-after" in response.headers:
                    return min(float(response.headers["retry-after"]), cfg["retry_after_cap"])
            except ValueError:
                pass
        delay = min(cfg["backoff_initial"] * 2 ** (attempt - 1), cfg["backoff_max"])
        return delay * (1 - random.random() * cfg["backoff_jitter"])

    @staticmethod
    def _error_text(response: httpx.Response) -> str:
        """오류 응답의 본문 앞부분. 요청 헤더(키)는 절대 넣지 않는다."""
        return response.text.strip().replace("\n", " ")[:300]

    def ask(self, payload: dict) -> Answer:
        """요청 한 건을 보내고 Answer를 돌려준다. 일시적인 오류(408, 429, 5xx, 연결 실패)는 재시도한다."""
        cfg = S.JEV_CONFIG
        qid = next(iter(payload["questions"]))        # 질문은 하나뿐이다
        labels = list(payload["questions"][qid]["criteria"])
        attempt, parse_errors = 0, 0
        while True:
            if self._abort.is_set():
                raise JevAborted("다른 요청의 인증 오류로 보내지 않았습니다.")
            response, problem = None, None
            started = time.perf_counter()
            try:
                response = self._http.post(S.ENDPOINT, json=payload)
                latency = time.perf_counter() - started
            except httpx.HTTPError as error:         # 시간 초과, 연결 끊김 등
                problem = f"연결 오류 ({type(error).__name__})"
            else:
                status = response.status_code
                if status == 200:
                    try:
                        return self._parse(response.json(), labels, qid, latency)
                    except ValueError as error:      # JSON이 아니거나 문서와 다른 모양의 응답. 프록시 오류일 수 있어 한 번만 재시도한다.
                        parse_errors += 1
                        if parse_errors > 1:         # 계속 이러면 형식이 바뀐 것이다. 요청(과금)을 더 반복하지 않는다.
                            raise JevApiError(f"응답 형식이 문서와 다릅니다: {error}", status) from error
                        problem = f"응답을 해석하지 못했습니다 ({error})"
                elif status in (401, 403):
                    self._abort.set()                # 키가 틀리면 모든 요청이 실패한다. 아직 안 보낸 요청은 보내지 않는다.
                    raise JevAuthError(f"API 키가 없거나 올바르지 않습니다 (HTTP {status}). .env의 JEV_API_KEY를 확인하세요.", status)
                elif status == 408 or status == 429 or status >= 500:
                    problem = f"HTTP {status}"
                else:                                # 422 등: 요청 자체가 잘못됐다. 재시도해도 같다.
                    raise JevRequestError(f"요청이 거부됐습니다 (HTTP {status}): {self._error_text(response)}", status)
            attempt += 1
            if attempt > cfg["max_retries"]:
                raise JevApiError(f"{problem}. 재시도 {cfg['max_retries']}번을 모두 실패했습니다.",
                                  response.status_code if response is not None else None)
            self._sleep(self._delay(attempt, response))

    def _parse(self, data: dict, labels: list, qid: str, latency: float) -> Answer:
        try:
            answer = data["answers"][qid]
            probabilities = {label: float(answer["probabilities"][label]) for label in labels}
            choice, confidence = answer["choice"], float(answer["confidence"])
            usage = data.get("usage") or {}
            model = data.get("model") or self.model
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"필요한 항목이 없습니다: {error!r}") from error
        if answer.get("type") != "choice" or choice not in labels:
            raise ValueError(f"선택지가 올바르지 않습니다: {answer.get('type')}, {choice!r}")
        if abs(sum(probabilities.values()) - 1) > 0.05:
            raise ValueError(f"확률의 합이 1이 아닙니다: {sum(probabilities.values()):.3f}")
        return Answer(choice, probabilities, confidence, int(usage.get("input_tokens") or 0),
                      int(usage.get("output_tokens") or 0), latency, model)

    # ── 여러 건을 동시에 ──
    def run(self, items: list[tuple[str, dict]], refresh: bool = False, label: str = "") -> dict[str, Answer]:
        """items: [(요청 키, 요청 본문)]. 같은 키는 한 번만 보내고, 캐시에 있으면 보내지 않는다. -> {키: Answer}

        하나라도 끝내 실패하면 나머지 요청을 취소하고 그 오류를 올린다. 이미 끝난 응답은 캐시에 남아서 다시 실행하면 이어서 한다.
        """
        results: dict[str, Answer] = {}
        todo: dict[str, dict] = {}
        for key, payload in items:
            cached = None if refresh else self.cache.get(key)
            if cached is not None:
                results[key] = cached
            elif key not in todo:
                todo[key] = payload
        if not todo:
            return results
        print(f"  [jev] {label} 새 요청 {len(todo)}건을 보냅니다. (캐시에서 가져온 것 {len(results)}건, 동시 {self.concurrency}건)", flush=True)
        started, last_print = time.perf_counter(), 0.0
        self._abort.clear()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency)
        futures = {executor.submit(self.ask, payload): key for key, payload in todo.items()}
        processed = set()
        try:
            for future in concurrent.futures.as_completed(futures):
                try:
                    answer = future.result()         # 실패하면 여기서 오류가 올라온다
                except JevAborted:
                    continue                         # 건너뛴 요청. 원인이 된 인증 오류가 이 반복문에서 곧 올라온다.
                processed.add(future)
                self.cache.put(futures[future], answer)
                answer.cached = False                # 이번에 보낸 요청이라는 표시
                results[futures[future]] = answer
                now = time.perf_counter() - started
                if now - last_print >= 5 or len(processed) == len(todo):
                    print(f"  [jev] {len(processed)}/{len(todo)} ({now:.0f}초, 초당 {len(processed) / max(now, 1e-9):.1f}건)", flush=True)
                    last_print = now
        finally:
            executor.shutdown(wait=True, cancel_futures=True)     # 멈추게 한 오류 때문에 취소되는 요청 외에는 끝까지 기다린다
            for future, key in futures.items():                   # 이미 응답을 받은 요청(과금됨)은 버리지 않고 캐시에 남긴다
                if future not in processed and future.done() and not future.cancelled() and future.exception() is None:
                    self.cache.put(key, future.result())
        return results

    # ── 연결 확인 (문장은 보내지 않는다) ──
    def list_models(self):
        """GET /v1/models의 응답(JSON)을 그대로 돌려준다. 계정이 쓸 수 있는 모델 이름이 들어 있다."""
        response = self._http.get("/v1/models")
        if response.status_code in (401, 403):
            raise JevAuthError(f"API 키가 없거나 올바르지 않습니다 (HTTP {response.status_code}). .env의 JEV_API_KEY를 확인하세요.", response.status_code)
        if response.status_code != 200:
            raise JevApiError(f"모델 목록을 가져오지 못했습니다 (HTTP {response.status_code}): {self._error_text(response)}", response.status_code)
        return response.json()


def model_names(body) -> list[str]:
    """list_models()의 응답에서 모델 이름을 뽑는다. 응답 모양을 문서에서 확인하지 못해서 흔한 모양을 모두 받아 준다."""
    items = body
    if isinstance(body, dict):
        items = next((body[key] for key in ("data", "models", "items") if isinstance(body.get(key), list)), [])
    names = []
    for item in items if isinstance(items, list) else []:
        name = item if isinstance(item, str) else (item.get("name") or item.get("id") or item.get("model")) if isinstance(item, dict) else None
        if name:
            names.append(str(name))
    return names
