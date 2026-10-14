"""OpenAI로 답변(answer, next_steps)을 생성한다.

설정은 .env의 OPENAI_API_KEY, OPENAI_MODEL, MAX_OUTPUT_TOKENS, GENERATION_TIMEOUT_SECONDS를 쓴다.
추론(reasoning) 모델은 추론 토큰도 MAX_OUTPUT_TOKENS에 들어가서 기본 강도에서는 답변이 잘리므로, OPENAI_REASONING_EFFORT(기본 low)로 낮춘다.
generate()는 예외를 던지지 않고 status(success / error / timeout)를 담은 결과를 돌려준다. API 키는 결과·로그에 넣지 않는다.
"""

import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APITimeoutError, AsyncOpenAI, OpenAIError

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

logger = logging.getLogger("uvicorn.error")


def parse_answer(raw: str) -> tuple[str, list[str]]:
    """모델 출력에서 answer와 next_steps를 읽는다. 형식이 맞지 않으면 ValueError."""
    text = raw.strip()
    if text.startswith("```"):          # Markdown 없이 달라고 했지만 코드 블록으로 감싸 오는 경우가 있다.
        text = text.strip("`").removeprefix("json").strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("JSON 객체가 아닙니다.")
    answer, steps = data.get("answer"), data.get("next_steps", [])
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer가 비어 있습니다.")
    if not isinstance(steps, list) or any(not isinstance(step, str) for step in steps):
        raise ValueError("next_steps가 문자열 배열이 아닙니다.")
    return answer.strip(), steps


class OpenAIGenerator:
    def __init__(self):
        self.model = os.getenv("OPENAI_MODEL") or ""
        self.max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS") or 512)
        self.timeout = float(os.getenv("GENERATION_TIMEOUT_SECONDS") or 60)
        # none/low/medium/high 등. "off"로 적으면 reasoning 옵션을 보내지 않는다. (추론을 지원하지 않는 모델용)
        effort = (os.getenv("OPENAI_REASONING_EFFORT") or "low").strip().lower()
        self.reasoning_effort = None if effort == "off" else effort
        self._client = None

    def _get_client(self) -> AsyncOpenAI:
        # 키가 없어도 서버는 떠야 하므로 첫 호출 때 만든다. 재시도는 끄고 timeout이 전체 대기 시간의 한도가 되게 한다.
        if self._client is None:
            if not os.getenv("OPENAI_API_KEY") or not self.model:
                raise RuntimeError("OPENAI_API_KEY 또는 OPENAI_MODEL이 .env에 없습니다.")
            self._client = AsyncOpenAI(timeout=self.timeout, max_retries=0)
        return self._client

    async def generate(self, prompt: str) -> dict:
        metrics = {"provider": "openai", "model": self.model, "max_output_tokens": self.max_output_tokens,
                   "timeout_s": self.timeout, "reasoning_effort": self.reasoning_effort,
                   "prompt_chars": len(prompt)}
        options = {"reasoning": {"effort": self.reasoning_effort}} if self.reasoning_effort else {}
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
        if response.status != "completed":
            reason = response.incomplete_details.reason if response.incomplete_details else response.status
            return self._failure("error", f"생성이 끝나지 않았습니다: {reason}", metrics, started)
        try:
            answer, next_steps = parse_answer(response.output_text)
        except ValueError as error:           # json.JSONDecodeError도 ValueError다.
            return self._failure("error", f"생성 결과의 형식이 올바르지 않습니다: {error}", metrics, started)
        return {"status": "success", "answer": answer, "next_steps": next_steps, "metrics": metrics}

    @staticmethod
    def _failure(status: str, message: str, metrics: dict, started: float) -> dict:
        metrics.setdefault("latency_ms", (time.perf_counter() - started) * 1000)
        return {"status": status, "error": message, "metrics": metrics}

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
