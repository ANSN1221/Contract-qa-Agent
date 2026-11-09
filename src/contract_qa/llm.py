"""OpenAI 兼容接口的最小客户端：向量化与对话。

只用标准库 urllib，避免为一个 POST 请求引入 SDK。所有失败都抛出带上下文的异常，
不做静默重试到成功——重试次数显式可见，方便在评测里统计失败率。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from contract_qa.config import Settings, get_settings

TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0


class ApiError(RuntimeError):
    pass


def _post(url: str, payload: dict, api_key: str) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise ApiError(f"HTTP {exc.code} {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ApiError(f"连接失败 {url}: {exc.reason}") from exc


def _post_with_retry(url: str, payload: dict, api_key: str) -> dict:
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _post(url, payload, api_key)
        except ApiError as exc:
            last = exc
            # 4xx 说明请求本身有问题，重试没有意义
            if "HTTP 4" in str(exc):
                raise
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    raise last  # type: ignore[misc]


def embed_texts(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    """批量向量化。按 16 条一批，避免单次请求体过大。"""
    settings = settings or get_settings()
    if not settings.has_embedding:
        raise ApiError("缺少 EMBEDDING_API_KEY")

    vectors: list[list[float]] = []
    batch_size = 16
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        data = _post_with_retry(
            f"{settings.embedding_base_url}/embeddings",
            {"model": settings.embedding_model, "input": batch, "encoding_format": "float"},
            settings.embedding_api_key,
        )
        items = sorted(data.get("data", []), key=lambda item: item["index"])
        if len(items) != len(batch):
            raise ApiError(f"向量条数不匹配：请求 {len(batch)} 条，返回 {len(items)} 条")
        vectors.extend(item["embedding"] for item in items)
    return vectors


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def chat(
    messages: list[dict[str, str]],
    settings: Settings | None = None,
    *,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> ChatResult:
    settings = settings or get_settings()
    if not settings.has_llm:
        raise ApiError("缺少 LLM_API_KEY")

    payload: dict[str, object] = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    data = _post_with_retry(
        f"{settings.llm_base_url}/chat/completions", payload, settings.llm_api_key
    )
    usage = data.get("usage", {})
    return ChatResult(
        content=data["choices"][0]["message"]["content"],
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
    )
