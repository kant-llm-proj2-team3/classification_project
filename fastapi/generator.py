"""OpenAI로 답변(answer, next_steps)을 생성한다.

  OpenAIGenerator : OpenAI. 설정은 .env의 OPENAI_API_KEY, OPENAI_MODEL, OPENAI_REASONING_EFFORT를 쓴다.
                    추론(reasoning) 모델은 추론 토큰도 MAX_OUTPUT_TOKENS에 들어가서 기본 강도에서는 답변이 잘리므로 강도를 낮춘다(기본 low).
  OllamaGenerator : 로컬 Ollama(비교용). 설정은 .env의 OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_NUM_CTX, OLLAMA_KEEP_ALIVE를 쓴다.
공통 설정: MAX_OUTPUT_TOKENS, GENERATION_TIMEOUT_SECONDS, GENERATION_TEMPERATURE(비우면 각 제공자의 기본값).
두 생성기는 같은 형식으로 답한다. generate()는 예외를 던지지 않고 status(success / error / timeout)를 담은 결과를 돌려준다.
metrics.provider가 어느 모델이 만든 결과인지 알려 준다. API 키는 결과·로그에 넣지 않는다.
"""

import json
import logging
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import APITimeoutError, AsyncOpenAI, OpenAIError
from policy import output_field

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

logger = logging.getLogger("uvicorn.error")

PROVIDERS = ("openai", "ollama")


def parse_answer(raw: str, field: str = "next_steps") -> tuple[str, list[str]]:
    """모델 출력에서 answer와 목록 필드(문의는 next_steps, 문서는 summary_items)를 읽는다. 형식이 맞지 않으면 ValueError."""
    text = raw.strip()
    if text.startswith("```"):          # Markdown 없이 달라고 했지만 코드 블록으로 감싸 오는 경우가 있다.
        text = text.strip("`").removeprefix("json").strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("JSON 객체가 아닙니다.")
    answer, steps = data.get("answer"), data.get(field, [] if field == "next_steps" else None)
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer가 비어 있습니다.")
    if not isinstance(steps, list) or any(not isinstance(step, str) for step in steps):
        raise ValueError(f"{field}가 문자열 배열이 아닙니다.")
    if field == "summary_items" and not any(step.strip() for step in steps):
        raise ValueError("summary_items가 비어 있습니다.")       # 요약 문서의 핵심이라 비어 있으면 실패로 본다.
    return answer.strip(), steps


def success_result(answer: str, items: list[str], field: str, metrics: dict) -> dict:
    """성공 결과. 문의는 {answer, next_steps}로 예전 그대로이고, 문서는 next_steps를 비우고 summary_items를 더한다."""
    result = {"status": "success", "answer": answer, "next_steps": items if field == "next_steps" else [], "metrics": metrics}
    if field == "summary_items":
        result["summary_items"] = items
    return result


def with_raw(result: dict, raw: str | None) -> dict:
    """평가 도구(evaluation)가 요청했을 때만 모델이 돌려준 원문을 결과 맨 위에 붙인다. 서비스는 요청하지 않으므로 응답과 추적에는 남지 않는다."""
    if raw is not None:
        result["raw_text"] = raw
    return result


def generation_temperature() -> float | None:
    """모든 생성기가 같은 온도로 비교되도록 .env의 GENERATION_TEMPERATURE를 읽는다. 비어 있으면 None(제공자 기본값)."""
    value = (os.getenv("GENERATION_TEMPERATURE") or "").strip()
    return float(value) if value else None


class OpenAIGenerator:
    provider = "openai"

    def __init__(self):
        self.model = os.getenv("OPENAI_MODEL") or ""
        self.max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS") or 512)
        self.timeout = float(os.getenv("GENERATION_TIMEOUT_SECONDS") or 60)
        # none/low/medium/high 등. "off"로 적으면 reasoning 옵션을 보내지 않는다. (추론을 지원하지 않는 모델용)
        effort = (os.getenv("OPENAI_REASONING_EFFORT") or "low").strip().lower()
        self.reasoning_effort = None if effort == "off" else effort
        self.temperature = generation_temperature()
        self._client = None

    def describe(self) -> dict:
        return {"model": self.model or None, "configured": bool(os.getenv("OPENAI_API_KEY") and self.model)}

    def _get_client(self) -> AsyncOpenAI:
        # 키가 없어도 서버는 떠야 하므로 첫 호출 때 만든다. 재시도는 끄고 timeout이 전체 대기 시간의 한도가 되게 한다.
        if self._client is None:
            if not os.getenv("OPENAI_API_KEY") or not self.model:
                raise RuntimeError("OPENAI_API_KEY 또는 OPENAI_MODEL이 .env에 없습니다.")
            self._client = AsyncOpenAI(timeout=self.timeout, max_retries=0)
        return self._client

    async def generate(self, prompt: str, keep_raw: bool = False, topic: str = "inquiries") -> dict:
        metrics = {"provider": "openai", "model": self.model, "max_output_tokens": self.max_output_tokens,
                   "timeout_s": self.timeout, "reasoning_effort": self.reasoning_effort,
                   "temperature": self.temperature, "prompt_chars": len(prompt)}
        options = {"reasoning": {"effort": self.reasoning_effort}} if self.reasoning_effort else {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        started = time.perf_counter()
        try:
            response = await self._get_client().responses.create(
                model=self.model, input=prompt, max_output_tokens=self.max_output_tokens,
                text={"format": {"type": "json_object"}}, **options)
        except APITimeoutError:
            return self._failure("timeout", "생성 모델 응답 시간이 초과되었습니다.", metrics, started)
        except (OpenAIError, RuntimeError) as error:
            logger.error("[generator] 호출 실패: %s", type(error).__name__)
            return self._failure("error", f"{type(error).__name__}: {str(error)[:200]}", metrics, started)

        usage = response.usage
        metrics.update(latency_ms=(time.perf_counter() - started) * 1000, response_model=response.model,
                       response_id=response.id, response_status=response.status,
                       input_tokens=usage.input_tokens if usage else None,
                       output_tokens=usage.output_tokens if usage else None,
                       reasoning_tokens=usage.output_tokens_details.reasoning_tokens if usage else None)
        raw = response.output_text if keep_raw else None
        if response.status != "completed":
            reason = response.incomplete_details.reason if response.incomplete_details else response.status
            return with_raw(self._failure("error", f"생성이 끝나지 않았습니다: {reason}", metrics, started), raw)
        field = output_field(topic)
        try:
            answer, items = parse_answer(response.output_text, field)
        except ValueError as error:           # json.JSONDecodeError도 ValueError다.
            return with_raw(self._failure("error", f"생성 결과의 형식이 올바르지 않습니다: {error}", metrics, started), raw)
        return with_raw(success_result(answer, items, field, metrics), raw)

    @staticmethod
    def _failure(status: str, message: str, metrics: dict, started: float) -> dict:
        metrics.setdefault("latency_ms", (time.perf_counter() - started) * 1000)
        return {"status": status, "error": message, "metrics": metrics}

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


class OllamaGenerator:
    """로컬 Ollama(/api/generate)로 같은 프롬프트의 답변을 만든다. 결과 형식은 OpenAIGenerator와 같다."""
    provider = "ollama"

    def __init__(self):
        self.base_url = (os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL") or ""
        self.max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS") or 512)
        self.timeout = float(os.getenv("GENERATION_TIMEOUT_SECONDS") or 60)
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX") or 4096)       # 프롬프트와 답변이 함께 들어가는 문맥 길이
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE") or "5m"      # 호출 뒤 모델을 메모리에 두는 시간
        self.temperature = generation_temperature()
        self._client = None

    def describe(self) -> dict:
        return {"model": self.model or None, "configured": bool(self.model), "base_url": self.base_url}

    def _get_client(self) -> httpx.AsyncClient:
        # 환경의 프록시 설정을 쓰지 않는다. (로컬 서버 호출이 프록시로 빠지지 않게)
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=5.0), trust_env=False)
        return self._client

    async def preload(self) -> bool:
        """서버 시작 때 모델을 메모리에 올려 둔다. 첫 요청이 모델 로딩 시간까지 떠안아서 비교가 왜곡되는 것을 막는다.
        Ollama가 꺼져 있거나 모델이 없어도 서버 시작은 계속하므로 예외를 던지지 않는다."""
        if not self.model:
            return False
        try:
            response = await self._get_client().post(f"{self.base_url}/api/generate",
                                                     json={"model": self.model, "keep_alive": self.keep_alive})
            response.raise_for_status()
            return True
        except httpx.HTTPError as error:
            logger.warning("[generator] Ollama 모델을 미리 올리지 못했습니다(%s). 첫 호출이 느릴 수 있습니다.", type(error).__name__)
            return False

    async def generate(self, prompt: str, keep_raw: bool = False, topic: str = "inquiries") -> dict:
        metrics = {"provider": "ollama", "model": self.model, "max_output_tokens": self.max_output_tokens,
                   "timeout_s": self.timeout, "temperature": self.temperature, "num_ctx": self.num_ctx,
                   "prompt_chars": len(prompt)}
        options = {"num_predict": self.max_output_tokens, "num_ctx": self.num_ctx}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        # format=json은 Ollama가 JSON만 내도록 강제한다. (OpenAI의 json_object와 같은 역할)
        body = {"model": self.model, "prompt": prompt, "stream": False, "format": "json",
                "keep_alive": self.keep_alive, "options": options}
        started = time.perf_counter()
        if not self.model:
            return OpenAIGenerator._failure("error", "OLLAMA_MODEL이 .env에 없습니다.", metrics, started)
        try:
            response = await self._get_client().post(f"{self.base_url}/api/generate", json=body)
        except httpx.TimeoutException:
            return OpenAIGenerator._failure("timeout", "생성 모델 응답 시간이 초과되었습니다.", metrics, started)
        except httpx.HTTPError as error:
            logger.error("[generator] Ollama 호출 실패: %s", type(error).__name__)
            return OpenAIGenerator._failure(
                "error", f"{type(error).__name__}: Ollama({self.base_url})에 연결하지 못했습니다.", metrics, started)

        metrics["latency_ms"] = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            try:
                detail = response.json().get("error", "")
            except ValueError:
                detail = ""
            return OpenAIGenerator._failure("error", f"HTTP {response.status_code}: {detail[:200]}", metrics, started)

        data = response.json()
        raw = data.get("response", "") if keep_raw else None

        def ms(key):      # Ollama의 시간 단위는 나노초다.
            return data[key] / 1e6 if data.get(key) is not None else None

        eval_count, eval_ns = data.get("eval_count"), data.get("eval_duration")
        metrics.update(
            response_model=data.get("model"), response_status="completed" if data.get("done") else "incomplete",
            done_reason=data.get("done_reason"), input_tokens=data.get("prompt_eval_count"), output_tokens=eval_count,
            reasoning_tokens=None, total_ms=ms("total_duration"), load_ms=ms("load_duration"),
            prompt_eval_ms=ms("prompt_eval_duration"), eval_ms=ms("eval_duration"),
            tokens_per_sec=(eval_count / (eval_ns / 1e9)) if eval_count and eval_ns else None,
            # 프롬프트+답변이 문맥 길이에 닿으면 Ollama가 프롬프트 앞부분을 조용히 잘라 낼 수 있다.
            context_full=bool(data.get("prompt_eval_count") and eval_count
                              and data["prompt_eval_count"] + eval_count >= self.num_ctx))
        if not data.get("done") or data.get("done_reason") == "length":
            return with_raw(OpenAIGenerator._failure(
                "error", f"생성이 끝나지 않았습니다: {data.get('done_reason') or 'incomplete'}", metrics, started), raw)
        field = output_field(topic)
        try:
            answer, items = parse_answer(data.get("response", ""), field)
        except ValueError as error:
            return with_raw(OpenAIGenerator._failure(
                "error", f"생성 결과의 형식이 올바르지 않습니다: {error}", metrics, started), raw)
        return with_raw(success_result(answer, items, field, metrics), raw)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def create_generators() -> dict:
    """제공자 이름 -> 생성기. 새 제공자를 더할 때는 PROVIDERS와 여기만 고치면 된다."""
    return {"openai": OpenAIGenerator(), "ollama": OllamaGenerator()}
