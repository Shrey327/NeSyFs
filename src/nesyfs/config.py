from dataclasses import dataclass
import os


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

    @classmethod
    def from_env(cls) -> "Settings":
        def required(name: str) -> str:
            value = os.getenv(name)
            if not value:
                raise ValueError(f"Missing required environment variable: {name}")
            return value

        settings = cls(
            openai_api_key=required("OPENAI_API_KEY"),
            model=os.getenv("NESYFS_MODEL", "gpt-4.1-mini"),
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=required("NEO4J_PASSWORD"),
            reflection_limit=int(os.getenv("NESYFS_REFLECTION_LIMIT", "3")),
            reflection_samples=int(os.getenv("NESYFS_REFLECTION_SAMPLES", "5")),
            particle_count=int(os.getenv("NESYFS_PARTICLE_COUNT", "6")),
            planning_horizon=int(os.getenv("NESYFS_PLANNING_HORIZON", "5")),
            retrieval_limit=int(os.getenv("NESYFS_RETRIEVAL_LIMIT", "100")),
        )
        if min(settings.reflection_limit, settings.reflection_samples, settings.particle_count,
               settings.planning_horizon, settings.retrieval_limit) < 1:
            raise ValueError("NeSyFS limits must all be positive integers")
        return settings
