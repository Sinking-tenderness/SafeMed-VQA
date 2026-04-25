from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from src.dataset_utils import ensure_dir, read_json, write_json
from src.json_utils import build_exception_failure_payload, safe_parse_and_validate_json_output


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    raise RuntimeError("Missing SILICONFLOW_API_KEY in .env")

SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
TEACHER_MODEL = "Qwen/Qwen3.5-122B-A10B"


class SiliconFlowClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = SILICONFLOW_API_URL,
        model: str = TEACHER_MODEL,
        timeout: int = 120,
        connectivity_timeout: int = 30,
        max_retries: int = 3,
        backoff_seconds: tuple[int, ...] = (2, 4, 8),
        error_log_path: str | Path = ROOT / "outputs" / "logs" / "datagen_errors.jsonl",
        enable_error_logging: bool = True,
    ) -> None:
        self.api_key = api_key or SILICONFLOW_API_KEY
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.connectivity_timeout = connectivity_timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.error_log_path = Path(error_log_path)
        self.enable_error_logging = enable_error_logging
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies = {}
        self.session.headers.update(self._headers())

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _log_error(self, payload: dict[str, Any]) -> None:
        if not self.enable_error_logging:
            return
        ensure_dir(self.error_log_path.parent)
        with self.error_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.session.close()

    def _post(self, payload: dict[str, Any], timeout: int | None = None) -> requests.Response:
        return self.session.post(
            self.base_url,
            json=payload,
            timeout=timeout or self.timeout,
            proxies={},
        )

    def load_status_state(self, status_path: str | Path) -> dict[str, Any]:
        target = Path(status_path)
        if not target.exists():
            return {}
        payload = read_json(target)
        if not isinstance(payload, dict):
            raise ValueError("Status file must contain a JSON object keyed by sample_id.")
        return payload

    def save_status_state(self, status_path: str | Path, status_state: dict[str, Any]) -> None:
        write_json(status_path, status_state)

    def update_status_entry(
        self,
        status_state: dict[str, Any],
        *,
        status_key: str,
        status: str,
        increment_attempt: bool = True,
        request_context: dict[str, Any] | None = None,
        error_payload: dict[str, Any] | None = None,
        output_record_path: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        previous = status_state.get(status_key, {})
        attempt_count = int(previous.get("attempt_count", 0))
        if increment_attempt and status in {"success", "failure", "failed"}:
            attempt_count += 1
        entry: dict[str, Any] = {
            "sample_id": status_key,
            "status": status,
            "attempt_count": attempt_count,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if request_context:
            entry.update(request_context)
        if error_payload:
            entry["last_failure"] = error_payload
        if output_record_path:
            entry["output_record_path"] = output_record_path
        if note:
            entry["note"] = note
        status_state[status_key] = entry
        return entry

    def log_error(
        self,
        *,
        event: str,
        error: str,
        request_context: dict[str, Any] | None = None,
        attempt: int | None = None,
        response_preview: str | None = None,
        request_preview: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "event": event,
            "model": self.model,
            "error": error,
        }
        if attempt is not None:
            payload["attempt"] = attempt
        if request_context:
            payload.update(request_context)
        if response_preview:
            payload["response_preview"] = response_preview
        if request_preview:
            payload["request_preview"] = request_preview
        self._log_error(payload)

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

        response = self._post(request_payload)
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

    def connectivity_self_check(self, timeout: int | None = None) -> dict[str, Any]:
        resolved_timeout = self.connectivity_timeout if timeout is None else timeout
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Reply with OK."}],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 8,
            "response_format": {"type": "text"},
        }
        response = self._post(payload, timeout=resolved_timeout)
        return {
            "ok": response.status_code == 200,
            "status_code": response.status_code,
            "base_url": self.base_url,
            "trust_env": self.session.trust_env,
            "session_proxies": dict(self.session.proxies),
            "timeout": resolved_timeout,
            "response_preview": response.text[:200],
        }

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 800,
        request_context: dict[str, Any] | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            raw_text: str | None = None
            try:
                raw_text = self.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_payload=extra_payload,
                )
                parsed_output, failure_payload = safe_parse_and_validate_json_output(raw_text)
                if parsed_output is not None:
                    return parsed_output
                raise RuntimeError(json.dumps(failure_payload, ensure_ascii=False))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                failure_payload = build_exception_failure_payload(exc)
                self.log_error(
                    event="teacher_completion_failed",
                    error=failure_payload["error"],
                    request_context=request_context,
                    attempt=attempt,
                    response_preview=raw_text[:500] if raw_text else failure_payload.get("response_preview"),
                    request_preview={
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "extra_payload": extra_payload,
                        "message_roles": [message.get("role") for message in messages],
                    },
                )
                if attempt >= self.max_retries:
                    break
                backoff = self.backoff_seconds[min(attempt - 1, len(self.backoff_seconds) - 1)]
                time.sleep(backoff)
        raise RuntimeError(f"SiliconFlow JSON completion failed: {last_error}")
