from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from src.dataset_utils import ensure_dir
from src.json_utils import parse_and_validate_json_output


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    raise RuntimeError("Missing SILICONFLOW_API_KEY in .env")

SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
TEACHER_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"


class SiliconFlowClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = SILICONFLOW_API_URL,
        model: str = TEACHER_MODEL,
        timeout: int = 120,
        max_retries: int = 3,
        backoff_seconds: tuple[int, ...] = (2, 4, 8),
        error_log_path: str | Path = ROOT / "outputs" / "logs" / "datagen_errors.jsonl",
    ) -> None:
        self.api_key = api_key or SILICONFLOW_API_KEY
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.error_log_path = Path(error_log_path)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _log_error(self, payload: dict[str, Any]) -> None:
        ensure_dir(self.error_log_path.parent)
        with self.error_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 800,
        extra_payload: dict[str, Any] | None = None,
    ) -> str:
        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "text"},
        }
        if extra_payload:
            request_payload.update(extra_payload)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self._headers(),
                    json=request_payload,
                    timeout=self.timeout,
                )
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                payload = response.json()
                choices = payload.get("choices") or []
                if not choices:
                    raise RuntimeError("No choices returned from SiliconFlow.")
                message = choices[0].get("message") or {}
                content = message.get("content", "")
                if isinstance(content, list):
                    text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                    content = "\n".join(part for part in text_parts if part)
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("Empty content returned from SiliconFlow.")
                return content
            except Exception as exc:  # noqa: BLE001
                self._log_error(
                    {
                        "attempt": attempt,
                        "model": self.model,
                        "error": str(exc),
                        "request_preview": {
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "message_roles": [message.get("role") for message in messages],
                        },
                    }
                )
                if attempt >= self.max_retries:
                    raise
                backoff = self.backoff_seconds[min(attempt - 1, len(self.backoff_seconds) - 1)]
                time.sleep(backoff)
        raise RuntimeError("SiliconFlow request failed after retries.")

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw_text = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
                return parse_and_validate_json_output(raw_text)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._log_error(
                    {
                        "attempt": attempt,
                        "model": self.model,
                        "error": f"json_validation_failed: {exc}",
                    }
                )
                if attempt >= self.max_retries:
                    break
                backoff = self.backoff_seconds[min(attempt - 1, len(self.backoff_seconds) - 1)]
                time.sleep(backoff)
        raise RuntimeError(f"SiliconFlow JSON completion failed: {last_error}")
