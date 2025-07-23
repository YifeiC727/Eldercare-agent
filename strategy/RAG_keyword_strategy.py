from typing import List, Dict, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import os
import json
import faiss
import numpy as np

class RAGKeywordStrategy:
    def __init__(self, corpus_path: str = "elder_topic_corpus.json"):
        self.corpus_path = os.path.join(os.path.dirname(__file__), "elder_topic_corpus.json")
        self.embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.dimension = self.embed_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
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
        embeddings = self.embed_model.encode(self.texts, convert_to_numpy=True, normalize_embeddings=True)
        self.index.add(embeddings)

    def retrieve_relevant_snippets(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.embed_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(np.array([query_embedding]), top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

    def generate_support_prompt(self, keywords: List[str]) -> Optional[str]:
        if not keywords:
            return None

        prompts = []
        for kw in keywords:
            # 首先尝试直接匹配keywords字段
            relevant_by_keyword = self._find_by_keyword(kw)
            if relevant_by_keyword:
                prompts.append(relevant_by_keyword["text"])
            else:
                # 如果没有找到，使用语义检索
                relevant = self.retrieve_relevant_snippets(kw, top_k=1)
                if relevant:
                    prompts.append(relevant[0]["text"])

        if not prompts:
            return None

        return "这些内容可能对你有帮助：" + " / ".join(prompts[:3])
    
    def _find_by_keyword(self, keyword: str) -> Optional[Dict]:
        """通过keywords字段直接匹配"""
        for metadata in self.metadata:
            if "keywords" in metadata:
                keywords_list = metadata["keywords"]
                if keyword in keywords_list:
                    return metadata
        return None

    def refresh_corpus(self, new_entries: List[Dict]):
        # 动态添加新语料内容（可选）
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