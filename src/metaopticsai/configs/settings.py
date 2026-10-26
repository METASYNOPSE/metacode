from __future__ import annotations
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[3]
_ENV = str(_ROOT / ".env")


class RCWAConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RCWA_", env_file=_ENV, extra="ignore")

    harmonics: int = 3
    resolution: int = 64
    minibatch_size: int = 20


class PathsConfig:
    """Not a Pydantic model — paths are deterministic, derived from project root."""

    def __init__(self):
        self.project_root: Path = _ROOT
        self.output_dir: Path = _ROOT / "outputs"
        self.papers_dir: Path = _ROOT / "papers"
        self.material_dir: Path = (
            _ROOT / "src" / "metaopticsai" / "physics" / "rcwa_engine" / "material_data"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)


class Settings:
    def __init__(self):
        self.rcwa = RCWAConfig()
        self.paths = PathsConfig()
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


settings = Settings()
