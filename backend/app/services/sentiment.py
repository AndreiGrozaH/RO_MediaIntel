import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

class SentimentService:
    _instance = None
    _model = None
    _tokenizer = None

    def __new__(cls):
        # Singleton: Asigură încărcarea modelului o singură dată
        if cls._instance is None:
            cls._instance = super(SentimentService, cls).__new__(cls)
            print(" Loading Sentiment Model (readerbench/ro-sentiment)...")
            try:
                model_name = "readerbench/ro-sentiment"
                cls._tokenizer = AutoTokenizer.from_pretrained(model_name)
                cls._model = AutoModelForSequenceClassification.from_pretrained(model_name)
                print(" Sentiment Model Loaded!")
            except Exception as e:
                print(f" Failed to load Sentiment Model: {e}")
                cls._instance = None
        return cls._instance

    def _get_score(self, text: str) -> float:
        """Helper intern: Returnează un scor între -1.0 (Negativ) și 1.0 (Pozitiv)"""
        if not self._model or not self._tokenizer:
            return 0.0

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        probs = softmax(outputs.logits, dim=1).numpy()[0]
        # Label 0 = Negativ, Label 1 = Pozitiv
        return float(probs[1] - probs[0])

    def get_entity_sentiment(self, text: str, entity_name: str) -> float:
        if not self._model or not entity_name:
            return 0.0
        
        # 1. Normalizăm textul (litere mici)
        text_lower = text.lower()
        entity_lower = entity_name.lower()

        # 2. Verificare Inteligentă:
        # Spargem numele în bucăți (ex: "Ion Marcel Ciolacu" -> ["ion", "marcel", "ciolacu"])
        name_parts = entity_lower.split()
        
        # Considerăm că entitatea este prezentă dacă MĂCAR UN nume (mai lung de 2 litere) apare în text
        # Ex: Dacă găsește "Ciolacu" în text, e suficient.
        is_present = any(part in text_lower for part in name_parts if len(part) > 2)
        
        if not is_present:
            return 0.0

        # 3. Calculăm sentimentul
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        probs = softmax(outputs.logits, dim=1).numpy()[0]
        # Label 1 = Pozitiv, Label 0 = Negativ
        score = float(probs[1] - probs[0])
        
        return score