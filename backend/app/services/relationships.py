import spacy
from typing import List, Dict, Tuple

class RelationshipExtractor:
    _instance = None
    _nlp = None

    # "Nobel-level" Vocabulary Lists
    CONFLICT_VERBS = {
        "ataca", "critică", "acuza", "respinge", "condamnă", "contestă", 
        "ironizează", "demite", "suspenda", "exclude", "cearta"
    }
    ALLIANCE_VERBS = {
        "susține", "aprobă", "votează", "propune", "felicită", "negociază", 
        "aliază", "semnează", "numește", "promovează", "validează"
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RelationshipExtractor, cls).__new__(cls)
            print("⏳ Loading spaCy (ro_core_news_lg) for Relationships...")
            try:
                # We only need the parser and tagger, strictly speaking
                cls._nlp = spacy.load("ro_core_news_lg", disable=["ner"])
                print(" spaCy Loaded!")
            except OSError:
                print(" spaCy model not found. Run: python -m spacy download ro_core_news_lg")
        return cls._instance

    def analyze_interactions(self, text: str, entities: Dict[str, List[str]]) -> List[Dict]:
        """
        Input: Text and the entities found by BERT.
        Output: List of relationships e.g., {'source': 'Ciolacu', 'target': 'Simion', 'type': 'conflict', 'verb': 'atacat'}
        """
        if not self._nlp or not text:
            return []

        doc = self._nlp(text)
        relationships = []
        
        # Flatten BERT entities into a set for fast lookup
        # We focus on PERSON and ORG for alliances/conflicts
        target_entities = set(entities.get("person", []) + entities.get("org", []))
        
        # Normalize entities for matching (lowercase)
        target_map = {e.lower(): e for e in target_entities}

        # Iterate through sentences to find interactions
        for sent in doc.sents:
            # 1. Identify chunks in this sentence that match our BERT entities
            found_in_sent = []
            for token in sent:
                if token.text.lower() in target_map:
                    found_in_sent.append(token)

            # We need at least 2 entities to have a relationship
            if len(found_in_sent) < 2:
                continue

            # 2. Dependency Parsing Magic
            # We look for two entities that share a common "HEAD" (Verb)
            for i in range(len(found_in_sent)):
                for j in range(i + 1, len(found_in_sent)):
                    token_a = found_in_sent[i]
                    token_b = found_in_sent[j]

                    # Trace up the dependency tree to find the main verb
                    verb = self._find_common_verb(token_a, token_b)
                    
                    if verb:
                        rel_type = self._classify_verb(verb.lemma_)
                        if rel_type != "neutral":
                            relationships.append({
                                "source": target_map[token_a.text.lower()],
                                "target": target_map[token_b.text.lower()],
                                "type": rel_type,
                                "verb": verb.text,
                                "snippet": sent.text
                            })

        return relationships

    def _find_common_verb(self, token1, token2):
        """
        Finds the first common ancestor in the dependency tree. 
        If it's a verb, we have a winner.
        """
        # Simplistic approach: checking if they share the same head or if one is head of another
        # A more robust 'Nobel' approach climbs the tree.
        
        ancestors1 = list(token1.ancestors) + [token1]
        ancestors2 = list(token2.ancestors) + [token2]
        
        common = [node for node in ancestors1 if node in ancestors2]
        
        if common:
            # The lowest common ancestor
            lca = common[0] 
            if lca.pos_ == "VERB":
                return lca
        return None

    def _classify_verb(self, lemma: str) -> str:
        # Check lemma against our dictionaries
        lemma = lemma.lower()
        if any(v in lemma for v in self.CONFLICT_VERBS):
            return "conflict"
        if any(v in lemma for v in self.ALLIANCE_VERBS):
            return "alliance"
        return "neutral"