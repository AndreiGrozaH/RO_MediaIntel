from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from ..constants import LABEL_ID2LABEL
from ..config import get_settings


@lru_cache(maxsize=1)
def get_ner_pipeline():
    settings = get_settings()
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    model = AutoModelForTokenClassification.from_pretrained(settings.model_name)
    model.config.id2label = LABEL_ID2LABEL
    model.config.label2id = {v: k for k, v in LABEL_ID2LABEL.items()}
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
