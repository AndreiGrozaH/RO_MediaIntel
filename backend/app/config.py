from functools import lru_cache
from typing import List

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "RO-MediaIntel API"
    default_sources: List[AnyHttpUrl] = [
        "https://www.digi24.ro/stiri/actualitate/politica",
        "https://spotmedia.ro/stiri/politica",
        "https://www.antena3.ro/politica",
        "https://www.stiripesurse.ro/politica",
        "https://www.dcnews.ro/politica",
        "https://www.digi24.ro/stiri/extern",
        "https://adevarul.ro/politica",
        "https://www.libertatea.ro/politica",
        "https://www.g4media.ro/category/politica",
    ]
    model_name: str = "dumitrescustefan/bert-base-romanian-ner"

    model_config = {
        "env_file": ".env",
        "protected_namespaces": ("settings_",),
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
