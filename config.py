"""
config.py — 所有設定集中在此，改一個地方全系統生效
"""

from pathlib import Path

# ── 路徑 ───────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
DATA_DIR         = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
METADATA_PATH    = VECTOR_STORE_DIR / "metadata.json"
BM25_PATH        = VECTOR_STORE_DIR / "bm25.pkl"

# ── 模型 ───────────────────────────────────────────────────────────────────────
EMBED_MODEL      = "BAAI/bge-m3"        # 1024 維，繁中 SOTA，首次執行自動下載 ~570MB
LLM_MODEL        = "qwen2.5:14b"        # 主力回答模型
REWRITE_MODEL    = "qwen2.5:3b"         # 專用於 Query Rewriting（輕量快速，無 thinking mode）
OLLAMA_BASE_URL  = "http://localhost:11434"

# ── 向量資料庫參數 ──────────────────────────────────────────────────────────────
EMBED_DIM            = 1024   # bge-m3 輸出維度
HNSW_M               = 32     # HNSW 圖中每個節點的連接數（越大越準但建構慢）
HNSW_EF_CONSTRUCTION = 200    # 建構時搜尋深度
HNSW_EF_SEARCH       = 128    # 查詢時搜尋深度
NORMALIZE_EMBEDDINGS = True   # L2 正規化後，Inner Product == Cosine Similarity

# ── 檢索參數 ───────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL      = 6     # 每個子查詢各自取幾個候選（BM25/FAISS 各自）
TOP_K_LLM            = 8     # 最終送進 LLM 的 unique chunk 數（v3.1 從 6 提升為 8）
SIMILARITY_THRESHOLD = 0.40  # cosine 低於此值 → 回傳「無法回覆」（可調整 0.3~0.6）
RRF_K                = 60    # RRF 平滑常數（學界標準值）

# ── Adaptive Query Decomposition ───────────────────────────────────────────────
MAX_DECOMPOSED_QUERIES = 6   # 最多拆解成幾個子查詢（v3.1 從4提升至6，覆蓋刑/行/民/程序各面向）

# ── LLM 參數 ───────────────────────────────────────────────────────────────────
LLM_TIMEOUT      = 180       # 等待 Ollama 回應的秒數（v3.1 context 加大，延長至180s）
LLM_TEMPERATURE  = 0.1       # 低 temperature 讓答案更穩定
MAX_CONTEXT_CHARS = 8000     # 送進 LLM 的最大 context 字元數（8 chunks 需要更多空間）
