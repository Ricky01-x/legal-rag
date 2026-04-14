"""
retriever.py — Hybrid Retrieval Engine

兩路檢索 + RRF 合併：
  ① FAISS Dense（bge-m3 語意向量）   → 捕捉語意相似性
  ② BM25 Sparse（jieba 關鍵字匹配）  → 捕捉精確術語
  ③ Reciprocal Rank Fusion（RRF）   → 合併兩路排名
"""

import json
import pickle
import numpy as np
import faiss
import jieba
from sentence_transformers import SentenceTransformer

from config import (
    FAISS_INDEX_PATH, METADATA_PATH, BM25_PATH,
    EMBED_MODEL, NORMALIZE_EMBEDDINGS,
    TOP_K_RETRIEVAL, RRF_K,
)

# 與 ingest.py 一致的自定義詞典
_LEGAL_TERMS = [
    "性騷擾", "性侵害", "權勢性騷擾", "被害人", "行為人",
    "申訴", "再申訴", "調解", "調查小組", "審議會",
    "主管機關", "直轄市", "縣市政府", "政府機關",
    "僱用人", "罰鍰", "損害賠償", "懲罰性賠償金",
    "告訴乃論", "性別平等", "調查報告", "處理建議",
]
for _t in _LEGAL_TERMS:
    jieba.add_word(_t)


class HybridRetriever:
    """
    混合檢索器。
    載入一次後可重複使用，所有索引常駐記憶體。
    """

    def __init__(self):
        # 確認索引檔存在
        for path in (FAISS_INDEX_PATH, METADATA_PATH, BM25_PATH):
            if not path.exists():
                raise FileNotFoundError(
                    f"找不到索引檔：{path}\n"
                    "請先執行：python ingest.py"
                )

        # 載入 metadata（chunk 原文 + 章條資訊）
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.chunks: list[dict] = json.load(f)

        # 載入 FAISS 索引
        self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

        # 載入 BM25 索引
        with open(BM25_PATH, "rb") as f:
            self.bm25 = pickle.load(f)

        # 載入 Embedding 模型（已在本地 cache，不需網路）
        self.embedder = SentenceTransformer(EMBED_MODEL)

        print(
            f"[Retriever] 就緒 — "
            f"{len(self.chunks)} 個 chunks，"
            f"FAISS {self.faiss_index.ntotal} 個向量"
        )

    # ── ① Dense 語意檢索 ───────────────────────────────────────────────────────

    def _dense_search(
        self, query: str, top_k: int
    ) -> tuple[list[int], list[float]]:
        """
        將問題 embed 後在 FAISS 中做 Inner Product 搜尋。
        回傳 (chunk_ids, cosine_scores)
        """
        q_vec = self.embedder.encode(
            [query],
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
        ).astype(np.float32)

        scores, indices = self.faiss_index.search(q_vec, top_k)
        return indices[0].tolist(), scores[0].tolist()

    # ── ② BM25 關鍵字檢索 ─────────────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int) -> list[int]:
        """
        jieba 斷詞後用 BM25 評分，回傳排序後的 chunk_ids。
        """
        tokens = jieba.lcut(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return ranked[:top_k]

    # ── ③ RRF 合併排序 ────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_merge(
        dense_ids: list[int],
        bm25_ids: list[int],
        k: int = RRF_K,
    ) -> list[int]:
        """
        Reciprocal Rank Fusion（RRF）：

            score(doc) = Σ_{每個排名列表} 1 / (rank + k)

        k=60 是學界標準預設值，可防止 rank=1 的文件得分過度膨脹。
        兩路排名來源的分數直接相加，不需要額外的分數正規化。
        """
        rrf: dict[int, float] = {}

        for rank, idx in enumerate(dense_ids):
            if idx < 0:          # FAISS 可能回傳 -1（padding）
                continue
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rank + k)

        for rank, idx in enumerate(bm25_ids):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rank + k)

        return sorted(rrf.keys(), key=lambda x: -rrf[x])

    # ── 主要介面 ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
    ) -> tuple[list[dict], float]:
        """
        混合檢索，回傳 (排序後的 chunks, 最佳 dense cosine 分數)。

        best_score 用於 RAG pipeline 的 threshold 判斷：
        若 best_score < SIMILARITY_THRESHOLD → 知識庫中無相關資訊。
        """
        # ① Dense
        dense_ids, dense_scores = self._dense_search(query, top_k * 2)
        best_score = float(max(dense_scores)) if dense_scores else 0.0

        # ② BM25
        bm25_ids = self._bm25_search(query, top_k * 2)

        # ③ RRF
        merged_ids = self._rrf_merge(dense_ids, bm25_ids)

        # 取前 top_k，過濾無效 id
        results = [
            self.chunks[i]
            for i in merged_ids[:top_k]
            if 0 <= i < len(self.chunks)
        ]

        return results, best_score
