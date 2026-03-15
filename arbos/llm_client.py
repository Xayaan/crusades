"""Multi-provider LLM client for generating train.py improvements.

Supports Chutes (default), OpenRouter, and Anthropic as LLM backends.
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger("arbos.llm")

PROMPT_FILE = Path(__file__).parent / "PROMPT.md"


def _load_system_prompt() -> str:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")
    return PROMPT_FILE.read_text().strip()


@dataclass
class LLMResponse:
    reasoning: str
    code: str


def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning models (DeepSeek-R1 etc.)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_think_summary(raw_text: str, max_chars: int = 3000) -> str:
    """Extract key reasoning from <think> tags.

    Takes the last several paragraphs (conclusion/decision section) which typically
    contains the most actionable reasoning from DeepSeek-R1's chain-of-thought.
    Falls back to single-paragraph or line-based splitting for short think blocks.
    """
    match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
    if not match:
        return ""
    thinking = match.group(1).strip()
    if not thinking:
        return ""

    paragraphs = [p.strip() for p in thinking.split("\n\n") if p.strip()]
    if not paragraphs:
        lines = [ln.strip() for ln in thinking.split("\n") if ln.strip()]
        if lines:
            tail = "\n".join(lines[-20:])
            return tail[-max_chars:] if len(tail) > max_chars else tail
        return thinking[-max_chars:]

    tail = "\n\n".join(paragraphs[-7:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def _parse_response(raw_text: str) -> LLMResponse:
    """Extract reasoning and code from LLM response."""
    think_summary = _extract_think_summary(raw_text)
    text = _strip_thinking_tags(raw_text)
    reasoning = ""
    code = ""

    reasoning_patterns = [
        r"(?:\*\*)?REASONING:?(?:\*\*)?:?\s*\n?(.*?)(?=\n(?:\*\*)?CODE|\n```)",
        r"#+\s*REASONING:?\s*\n?(.*?)(?=\n#+\s*CODE|\nCODE:|\n```)",
        r"^(.*?)(?=\nCODE:|\n```python)",
    ]
    for pattern in reasoning_patterns:
        m = re.search(pattern, text, re.DOTALL)
        if m and m.group(1).strip():
            reasoning = re.sub(r"^\*\*\s*", "", m.group(1).strip())
            break

    if think_summary:
        if reasoning:
            reasoning = f"{reasoning}\n\n[From chain-of-thought]: {think_summary}"
        else:
            reasoning = think_summary

    if not reasoning:
        pre_code = re.split(r"```python", text, maxsplit=1)[0].strip()
        if pre_code and len(pre_code) > 20:
            reasoning = pre_code[-2000:]
        else:
            reasoning = "No reasoning provided."

    # Find all python code blocks; pick the one after CODE: marker, or the largest
    all_blocks = list(re.finditer(r"```python\s*\n(.*?)```", text, re.DOTALL))
    if all_blocks:
        code_section = re.search(r"CODE:\s*\n(.*)", text, re.DOTALL)
        if code_section:
            # Find the first code block that starts within or after the CODE: section
            code_start = code_section.start()
            for block in all_blocks:
                if block.start() >= code_start:
                    code = block.group(1).strip()
                    break
        if not code:
            # Fall back to the largest block (most likely the full file)
            code = max((b.group(1).strip() for b in all_blocks), key=len)
    elif "def inner_steps" in text:
        lines = text.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if line.startswith(("import ", "from ", "def ", "class ", "@")):
                in_code = True
            if in_code:
                code_lines.append(line)
        code = "\n".join(code_lines).strip()

    if not code:
        raise ValueError("Could not extract code from LLM response")

    return LLMResponse(reasoning=reasoning or "No reasoning provided", code=code)


class LLMClient:
    """Multi-provider LLM client."""

    def __init__(self):
        self.provider = os.environ.get("LLM_PROVIDER", "chutes")
        if self.provider not in ("chutes", "openrouter", "anthropic"):
            raise ValueError(
                f"Unknown LLM_PROVIDER '{self.provider}'. "
                f"Must be one of: chutes, openrouter, anthropic"
            )
        self.timeout = int(os.environ.get("LLM_TIMEOUT", "600"))

        if self.provider == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            self.base_url = "https://api.anthropic.com/v1"
            self.model = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")
        elif self.provider == "openrouter":
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
            self.base_url = "https://openrouter.ai/api/v1"
            self.model = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")
        else:
            self.api_key = os.environ.get("CHUTES_API_KEY", "")
            self.base_url = os.environ.get("CHUTES_BASE_URL", "https://llm.chutes.ai/v1")
            self.model = os.environ.get(
                "LLM_MODEL",
                "deepseek-ai/DeepSeek-R1-0528-TEE",
            )

        if not self.api_key:
            key_name = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
            }.get(self.provider, "CHUTES_API_KEY")
            raise ValueError(f"{key_name} not set. Set it in .env or environment.")

        logger.info(f"LLM provider: {self.provider} ({self.model})")

    def _call_anthropic(self, messages: list[dict]) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": 16384,
            "temperature": 0.6,
            "system": _load_system_prompt(),
            "messages": messages,
        }
        if "extended" in self.model.lower() or "thinking" in self.model.lower():
            body.pop("temperature", None)
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/messages", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

    def _model_traits(self) -> dict:
        """Detect model family traits for optimal API usage."""
        m = self.model.lower()
        is_openai_reasoning = bool(re.search(r"\bo[13][\b\-]", m))
        is_deepseek_reasoning = "deepseek" in m and "r1" in m
        is_qwq = "qwq" in m
        is_reasoning = is_openai_reasoning or is_deepseek_reasoning or is_qwq

        return {
            "merge_system_into_user": is_reasoning,
            "supports_temperature": not is_openai_reasoning,
            "system_role": "developer" if is_openai_reasoning else "system",
            "has_think_tags": is_deepseek_reasoning or is_qwq,
        }

    def _call_openai_compat(self, messages: list[dict]) -> str:
        """Call OpenAI-compatible API (Chutes, OpenRouter)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        system_prompt = _load_system_prompt()
        traits = self._model_traits()

        if traits["merge_system_into_user"]:
            api_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    api_messages.append(
                        {
                            "role": "user",
                            "content": f"{system_prompt}\n\n---\n\n{msg['content']}",
                        }
                    )
                else:
                    api_messages.append(msg)
        else:
            api_messages = [
                {"role": traits["system_role"], "content": system_prompt},
                *messages,
            ]

        body = {
            "model": self.model,
            "max_tokens": 16384,
            "messages": api_messages,
        }
        if traits["supports_temperature"]:
            body["temperature"] = float(os.environ.get("LLM_TEMPERATURE", "0.6"))

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def _call(self, messages: list[dict]) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(messages)
        return self._call_openai_compat(messages)

    def suggest_improvement(
        self,
        code: str,
        mfu: float,
        history: str,
        hparams: dict,
    ) -> LLMResponse:
        """Ask the LLM to suggest one improvement to the train.py code."""
        num_gpus = (hparams.get("docker") or {}).get("num_gpus", 2)
        user_msg = f"""## Current best train.py (MFU: {mfu:.2f}%)

```python
{code}
```

## Hardware
- {num_gpus}x A100 80GB SXM (NVLink), 312 TFLOPS bf16 peak each
- Model: {hparams["benchmark_model_name"]}
- Batch: {hparams["benchmark_batch_size"]}, Seq len: {hparams["benchmark_sequence_length"]}, Steps: {hparams["eval_steps"]}

## Previous attempts
{history}

Current best MFU: {mfu:.2f}%. Beat it. Return the COMPLETE updated train.py."""

        messages = [{"role": "user", "content": user_msg}]

        start = time.time()
        logger.info(f"Requesting improvement from {self.provider} ({self.model})...")

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                text = self._call(messages)
                elapsed = time.time() - start
                logger.info(f"LLM responded in {elapsed:.1f}s")
                return _parse_response(text)
            except httpx.ReadTimeout:
                logger.warning(
                    f"LLM read timeout (attempt {attempt}/{max_retries}, timeout={self.timeout}s)"
                )
                if attempt == max_retries:
                    raise
                time.sleep(10)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                detail = ""
                try:
                    body = e.response.json()
                    detail = body.get("detail", {})
                    if isinstance(detail, dict):
                        detail = detail.get("message", str(detail))
                except Exception:
                    detail = e.response.text[:200]

                if status == 402:
                    raise RuntimeError(
                        f"LLM API payment required (HTTP 402): {detail}. "
                        f"Your {self.provider} account has no balance. "
                        f"Add funds or switch LLM_PROVIDER in arbos/.env"
                    ) from e

                if status == 404:
                    raise RuntimeError(
                        f"Model '{self.model}' not found on {self.provider} (HTTP 404). "
                        f"Check available models or set LLM_MODEL in arbos/.env"
                    ) from e

                logger.warning(
                    f"LLM API error (attempt {attempt}/{max_retries}): HTTP {status} — {detail}"
                )
                if status not in (429, 500, 502, 503, 504):
                    raise
                if attempt == max_retries:
                    raise
                wait = 30 * attempt if status == 429 else 2**attempt
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
            except ValueError as e:
                logger.warning(
                    f"Failed to parse LLM response (attempt {attempt}/{max_retries}): {e}"
                )
                if attempt == max_retries:
                    raise
                time.sleep(2)
