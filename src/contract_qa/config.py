"""配置加载：读取 .env，不引入额外依赖。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class ConfigError(RuntimeError):
    """缺少必要配置时抛出，提示要补哪个变量。"""


def load_env(path: Path | None = None) -> dict[str, str]:
    """把 .env 读进 os.environ（已存在的环境变量优先，不覆盖）。"""
    env_path = path or ENV_FILE
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        values[key] = value
        os.environ.setdefault(key, value)
    return values


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    chroma_path: Path
    processed_dir: Path

    @property
    def has_llm(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def has_embedding(self) -> bool:
        return bool(self.embedding_api_key)


def get_settings() -> Settings:
    load_env()
    return Settings(
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        llm_base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/"),
        llm_model=os.environ.get("LLM_MODEL", "deepseek-chat"),
        embedding_api_key=os.environ.get("EMBEDDING_API_KEY", ""),
        embedding_base_url=os.environ.get("EMBEDDING_BASE_URL", "").rstrip("/"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3"),
        chroma_path=PROJECT_ROOT / os.environ.get("CHROMA_PATH", "./chroma"),
        processed_dir=PROJECT_ROOT / "data" / "processed",
    )


def require(settings: Settings, *, embedding: bool = False) -> None:
    if embedding and not settings.has_embedding:
        raise ConfigError(
            "缺少 EMBEDDING_API_KEY。DeepSeek 不提供 Embedding 接口，"
            "请在 .env 里填入硅基流动等 OpenAI 兼容服务的密钥。"
        )
    if not embedding and not settings.has_llm:
        raise ConfigError("缺少 LLM_API_KEY，请在 .env 里填入对话模型密钥。")
