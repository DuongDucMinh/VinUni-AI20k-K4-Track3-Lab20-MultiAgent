"""LLM client abstraction.

Supports OpenAI, Groq, and custom OpenAI-compatible providers.
Features:
- Auto-detection of Groq endpoints from `gsk_` API keys or `OPENAI_BASE_URL`
- Model fallback rotation on rate limits (429) or provider errors
- Pacing delay to avoid exceeding tight RPM/TPM thresholds
- Robust retry with exponential backoff
- Token usage tracking and cost calculation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)

# Standard model pricing (USD per 1M tokens)
_PRICING_TABLE = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    model_used: str | None = None


class LLMClient:
    """Provider-agnostic LLM client with Groq and multi-model fallback support."""

    def __init__(self) -> None:
        from multi_agent_research_lab.core.config import get_settings

        settings = get_settings()
        api_key = settings.effective_api_key
        if not api_key:
            raise AgentExecutionError(
                "Neither GROQ_API_KEY nor OPENAI_API_KEY is set. Add it to your .env file."
            )

        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise AgentExecutionError(
                "openai package not installed. Run: pip install openai"
            ) from exc

        base_url = settings.effective_base_url
        provider_name = "Groq" if settings.is_groq else "OpenAI"
        logger.info(
            "Initializing LLMClient [Provider: %s | Base URL: %s | Model: %s]",
            provider_name,
            base_url or "Default OpenAI",
            settings.effective_model,
        )

        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=settings.timeout_seconds,
        )

        # Build candidate model list with fallbacks
        primary = settings.effective_model
        fallback_list = [m for m in settings.fallback_models if m != primary]
        self._model_candidates = [primary] + fallback_list
        self._inter_call_delay = settings.inter_call_delay_seconds
        self._last_call_time = 0.0


    def _pace(self) -> None:
        """Ensure minimum pause between calls to respect rate limits (e.g. 30 RPM / 8K TPM)."""
        if self._inter_call_delay <= 0:
            return
        now = time.perf_counter()
        elapsed = now - self._last_call_time
        if elapsed < self._inter_call_delay:
            sleep_time = self._inter_call_delay - elapsed
            logger.debug("Pacing LLM call: sleeping for %.2fs", sleep_time)
            time.sleep(sleep_time)
        self._last_call_time = time.perf_counter()

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Call LLM with fallback models and rate limit handling."""
        self._pace()

        last_exc: Exception | None = None

        for model in self._model_candidates:
            try:
                logger.info("Attempting LLM completion with model=%s", model)
                response = self._call_model(model, system_prompt, user_prompt)
                return response
            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                is_rate_limit = "429" in err_str or "rate limit" in err_str or "quota" in err_str
                logger.warning(
                    "Model %s failed (rate_limit=%s): %s. Trying next candidate...",
                    model, is_rate_limit, exc
                )
                if is_rate_limit:
                    time.sleep(3.0)  # brief cool-off before next candidate

        raise AgentExecutionError(
            f"All candidate models exhausted {self._model_candidates}. Last error: {last_exc}"
        ) from last_exc

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=6),
        reraise=True,
    )
    def _call_model(self, model: str, system_prompt: str, user_prompt: str) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        choice = response.choices[0]
        content = choice.message.content or ""

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None

        cost_usd: float | None = None
        if input_tokens is not None and output_tokens is not None:
            pricing = _PRICING_TABLE.get(model, {"input": 0.15, "output": 0.60})
            cost_usd = (
                input_tokens * pricing["input"] / 1_000_000
                + output_tokens * pricing["output"] / 1_000_000
            )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            model_used=model,
        )


