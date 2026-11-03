from dataclasses import dataclass
import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


MLX_MODEL = str(Path.home() / "Documents/stillroom-pr12-home/.models/assistant-qwen3-1.7b")
MLX_BASE_URL = "http://127.0.0.1:18768/v1"
MLX_4B_MODEL = str(Path.home() / "Documents/Stillroom/.models/writer-qwen3-4b")
MLX_4B_BASE_URL = "http://127.0.0.1:18771/v1"
MLX_PROVIDERS = {
    "mlx": (MLX_MODEL, MLX_BASE_URL, "Local MLX · Qwen3 1.7B 4-bit"),
    "mlx_4b": (MLX_4B_MODEL, MLX_4B_BASE_URL, "Local MLX · Qwen3 4B 4-bit"),
}


def provider_options() -> list[dict]:
    """Check configuration and the fixed local server without running inference."""
    options = [
        {"id": "openai", "label": "OpenAI", "model": os.getenv("NESYFS_MODEL", "gpt-4.1-mini"),
         "ready": bool(os.getenv("OPENAI_API_KEY")),
         "detail": "Remote API calls may incur charges."},
    ]
    opener = build_opener(ProxyHandler({}))
    for provider, (model, base_url, label) in MLX_PROVIDERS.items():
        local_ready = False
        try:
            with opener.open(f"{base_url}/models", timeout=1) as response:
                payload = json.loads(response.read(1_000_001))
            local_ready = any(item.get("id") == model for item in payload.get("data", []))
        except (URLError, OSError, ValueError, TypeError, AttributeError):
            pass
        options.append({"id": provider, "label": label, "model": model, "ready": local_ready,
                        "detail": ("Local server has the expected model; structured output is not guaranteed."
                                   if local_ready else "Start the local MLX server with the expected model; no cloud fallback.")})
    return options


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    model: str
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    reflection_limit: int = 3
    reflection_samples: int = 5
    particle_count: int = 6
    planning_horizon: int = 5
    retrieval_limit: int = 100
    provider: str = "openai"

    @classmethod
    def from_env(cls, provider: str = "openai") -> "Settings":
        if provider != "openai" and provider not in MLX_PROVIDERS:
            raise ValueError("Unsupported model provider")
        def required(name: str) -> str:
            value = os.getenv(name)
            if not value:
                raise ValueError(f"Missing required environment variable: {name}")
            return value

        settings = cls(
            openai_api_key=required("OPENAI_API_KEY") if provider == "openai" else "",
            model=os.getenv("NESYFS_MODEL", "gpt-4.1-mini") if provider == "openai" else MLX_PROVIDERS[provider][0],
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=required("NEO4J_PASSWORD"),
            reflection_limit=int(os.getenv("NESYFS_REFLECTION_LIMIT", "3")),
            reflection_samples=int(os.getenv("NESYFS_REFLECTION_SAMPLES", "5")),
            particle_count=int(os.getenv("NESYFS_PARTICLE_COUNT", "6")),
            planning_horizon=int(os.getenv("NESYFS_PLANNING_HORIZON", "5")),
            retrieval_limit=int(os.getenv("NESYFS_RETRIEVAL_LIMIT", "100")),
            provider=provider,
        )
        if min(settings.reflection_limit, settings.reflection_samples, settings.particle_count,
               settings.planning_horizon, settings.retrieval_limit) < 1:
            raise ValueError("NeSyFS limits must all be positive integers")
        return settings
