from google import genai
import numpy as np
from config import settings
from models import NodeData
from typing import List, Tuple

class SemanticMapper:
    THRESHOLD = 0.40  # Lowered for better discovery (Bug 4.3)
    TOP_K     = 3

    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_id = "text-embedding-004"
        self._cache: dict[str, np.ndarray] = {}

    def find_best_node(self, intent: str,
                        nodes: List[NodeData]) -> NodeData | None:
        if not nodes:
            return None

        # Embed intent
        intent_emb = self._embed(intent)

        # Build text representations and embed all nodes
        node_texts = [n.to_text_repr() for n in nodes]
        
        # Batch embed all nodes
        response = self.client.models.embed_content(
            model=self.model_id,
            contents=node_texts,
            config={"task_type": "RETRIEVAL_DOCUMENT"}
        )
        node_embs = np.array([e.values for e in response.embeddings])
        
        # Normalize node embeddings
        norms = np.linalg.norm(node_embs, axis=1, keepdims=True) + 1e-8
        node_embs_norm = node_embs / norms
        
        # Normalize intent embedding
        intent_emb_norm = intent_emb / (np.linalg.norm(intent_emb) + 1e-8)

        # Cosine similarity
        scores = node_embs_norm @ intent_emb_norm

        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= self.THRESHOLD:
            return nodes[best_idx]

        # Below threshold — fallback to highest score if it's reasonable
        if best_score > 0.30:
            return nodes[best_idx]
            
        return None

    def _embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            response = self.client.models.embed_content(
                model=self.model_id,
                contents=text,
                config={"task_type": "RETRIEVAL_QUERY"}
            )
            self._cache[text] = np.array(response.embeddings[0].values)
            
            if len(self._cache) > 512:   # LRU eviction
                self._cache.pop(next(iter(self._cache)))
        return self._cache[text]
