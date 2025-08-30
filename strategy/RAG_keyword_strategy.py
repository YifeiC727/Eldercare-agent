from typing import List, Dict, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import os
import json
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not available, RAG functionality will be limited")

class RAGKeywordStrategy:
    def __init__(self, corpus_path: str = "elder_topic_corpus.json"):
        self.corpus_path = os.path.join(os.path.dirname(__file__), "elder_topic_corpus.json")
        
        # Initialize embed_model, set to None if failed
        try:
            self.embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            self.dimension = self.embed_model.get_sentence_embedding_dimension()
            if FAISS_AVAILABLE:
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = None
        except Exception as e:
            print(f"Warning: Failed to initialize SentenceTransformer in RAGKeywordStrategy: {e}")
            self.embed_model = None
            self.dimension = 384  # Default dimension
            self.index = None
        
        self.texts = []
        self.metadata = []

        self._load_corpus()

    def _load_corpus(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus file {self.corpus_path} not found.")

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        self.texts = [entry["text"] for entry in entries]
        self.metadata = [entry for entry in entries]
        
        # Only compute embeddings when embed_model is available
        if self.embed_model is not None and self.index is not None:
            try:
                embeddings = self.embed_model.encode(self.texts, convert_to_numpy=True, normalize_embeddings=True)
                self.index.add(embeddings)
            except Exception as e:
                print(f"Warning: Failed to encode corpus embeddings: {e}")

    def retrieve_relevant_snippets(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.embed_model is None or self.index is None:
            # If embed_model is not available, return empty list
            return []
        
        try:
            query_embedding = self.embed_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
            D, I = self.index.search(np.array([query_embedding]), top_k)
            results = []
            for idx in I[0]:
                if idx < len(self.metadata):
                    results.append(self.metadata[idx])
            return results
        except Exception as e:
            print(f"Warning: Failed to retrieve relevant snippets: {e}")
            return []

    def generate_support_prompt(self, keywords: List[str]) -> Optional[str]:
        if not keywords:
            return None

        prompts = []
        for kw in keywords:
            # First try direct matching with keywords field
            relevant_by_keyword = self._find_by_keyword(kw)
            if relevant_by_keyword:
                prompts.append(relevant_by_keyword["text"])
            else:
                # If not found, use semantic retrieval
                relevant = self.retrieve_relevant_snippets(kw, top_k=1)
                if relevant:
                    prompts.append(relevant[0]["text"])

        if not prompts:
            return None

        return "这些内容可能对你有帮助：" + " / ".join(prompts[:3])
    
    def _find_by_keyword(self, keyword: str) -> Optional[Dict]:
        """Direct matching through keywords field"""
        for metadata in self.metadata:
            if "keywords" in metadata:
                keywords_list = metadata["keywords"]
                if keyword in keywords_list:
                    return metadata
        return None

    def refresh_corpus(self, new_entries: List[Dict]):
        # Dynamically add new corpus content (optional)
        for entry in new_entries:
            self.texts.append(entry["text"])
            self.metadata.append(entry)
            emb = self.embed_model.encode(entry["text"], convert_to_numpy=True, normalize_embeddings=True)
            self.index.add(np.array([emb]))

    def get_random_knowledge_snippet(self) -> str:
        import random
        return random.choice(self.texts) if self.texts else ""

    def get_corpus_size(self) -> int:
        return len(self.texts)

    def get_supported_keywords(self) -> List[str]:
        tags = []
        for m in self.metadata:
            if "keywords" in m:
                tags.extend(m["keywords"])
        return list(set(tags))