from sentence_transformers import SentenceTransformer
import numpy as np
from models import NodeData
from typing import List, Tuple

class SemanticMapper:
    THRESHOLD = 0.55
    TOP_K     = 3

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self._cache: dict[str, np.ndarray] = {}

    def find_best_node(self, intent: str,
                        nodes: List[NodeData]) -> NodeData | None:
        if not nodes:
            return None

        # Embed intent
        intent_emb = self._embed(intent)

        # Build text representations and embed all nodes
        node_texts = [n.to_text_repr() for n in nodes]
        node_embs  = self.model.encode(node_texts,
                                        batch_size=64,
                                        normalize_embeddings=True)
        intent_emb_norm = intent_emb / (np.linalg.norm(intent_emb) + 1e-8)

        # Cosine similarity (embeddings already normalised)
        scores = node_embs @ intent_emb_norm

        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= self.THRESHOLD:
            return nodes[best_idx]

        # Below threshold — ask LLM to pick from top-3 candidates
        top_indices = np.argsort(scores)[::-1][:self.TOP_K]
        candidates  = [(nodes[i], float(scores[i])) for i in top_indices]
        return self._llm_fallback(intent, candidates)

    def _embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = self.model.encode(
                text, normalize_embeddings=True)
            if len(self._cache) > 512:   # LRU eviction
                self._cache.pop(next(iter(self._cache)))
        return self._cache[text]

    def _llm_fallback(self, intent: str,
                       candidates: List[Tuple[NodeData, float]]
                       ) -> NodeData | None:
        if not candidates:
            return None
        # Simple heuristic fallback: return highest scoring candidate
        # even below threshold rather than making an extra LLM call
        # (LLM call costs 3 tokens — only use if confidence very low)
        best_node, best_score = candidates[0]
        if best_score > 0.30:
            return best_node
        return None   # truly no match
