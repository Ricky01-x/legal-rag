#!/usr/bin/env python3
"""
ingest.py — 資料處理流程（執行一次即可）

流程：PDF 解析 → 文字清洗 → 法條切分（章+條兩層） → Embedding → 寫入 FAISS + BM25

用法：
    python ingest.py                          # 自動掃描 data/ 目錄
    python ingest.py data/性騷擾防治法.pdf    # 指定路徑
"""

import sys
import re
import json
import pickle
from pathlib import Path

import fitz                          # pymupdf
import httpx
import numpy as np
import faiss
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    DATA_DIR, VECTOR_STORE_DIR,
    FAISS_INDEX_PATH, METADATA_PATH, BM25_PATH,
    EMBED_MODEL, EMBED_DIM,
    HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH,
    NORMALIZE_EMBEDDINGS,
    OLLAMA_BASE_URL, LLM_MODEL,
)

# ── jieba 法律術語自定義詞典 ────────────────────────────────────────────────────
# 提升 BM25 的斷詞精準度，讓「性騷擾」不會被拆成「性」+「騷擾」
_LEGAL_TERMS = [
    "性騷擾", "性侵害", "權勢性騷擾", "被害人", "行為人",
    "申訴", "再申訴", "調解", "調查小組", "審議會",
    "主管機關", "直轄市", "縣市政府", "政府機關",
    "僱用人", "罰鍰", "損害賠償", "懲罰性賠償金",
    "告訴乃論", "性別平等", "調查報告", "處理建議",
    "申訴管道", "防治措施", "教育訓練",
]
for _term in _LEGAL_TERMS:
    jieba.add_word(_term)


# ── 中文數字轉換 ────────────────────────────────────────────────────────────────
_ZH_NUM = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

def _zh_to_int(zh: str) -> int:
    s = zh.strip()
    if len(s) == 1:
        return _ZH_NUM.get(s, 0)
    if s[0] == "十":
        return 10 + (_ZH_NUM.get(s[1], 0) if len(s) > 1 else 0)
    if len(s) == 2 and s[1] == "十":
        return _ZH_NUM.get(s[0], 0) * 10
    if len(s) >= 2:
        return _ZH_NUM.get(s[0], 0) * 10 + _ZH_NUM.get(s[-1], 0)
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — PDF 解析
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: Path) -> str:
    """用 PyMuPDF (fitz) 逐頁提取文字，保留換行結構。"""
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    raw = "\n".join(pages)
    print(f"  提取完成：{len(raw):,} 字元，共 {len(pages)} 頁")
    return raw


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — 文字清洗
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(raw: str) -> str:
    """
    清洗策略：
    1. 移除全國法規資料庫頁眉（中英文 logo 文字）
    2. 移除法規名稱、修正日期標頭列
    3. 正規化多餘空白與連續空行
    """
    text = raw

    # 移除頁眉 logo
    text = re.sub(r"全國法規資料庫[^\n]*\n?", "", text)
    text = re.sub(r"Laws\s*&\s*Regulations[^\n]*\n?", "", text, flags=re.IGNORECASE)

    # 移除標頭列
    text = re.sub(r"法規名稱[：:][^\n]+\n?", "", text)
    text = re.sub(r"修正日期[：:][^\n]+\n?", "", text)

    # 正規化空白
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    print(f"  清洗完成：{len(text):,} 字元（移除 {len(raw)-len(text):,} 字元）")
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — 法條切分（章 + 條 兩層 Metadata）
# ══════════════════════════════════════════════════════════════════════════════

# 章標題：「第 一 章 總則」或「第一章 總則」（有無空格皆支援）
_CHAPTER_RE = re.compile(
    r"第\s*([一二三四五六七八九十]+)\s*章\s+(.+?)$"
)
# 條號：「第 1 條」或「第1條」
_ARTICLE_RE = re.compile(r"^第\s*(\d+)\s*條\s*$")


def parse_law_chunks(text: str) -> list[dict]:
    """
    以法條（第 N 條）為基本切分單位，保留所有章節 metadata。

    Metadata 兩層結構：
      chapter_num  : int   章的序號（1, 2, 3...）
      chapter      : str   「第一章 總則」
      article_num  : int   條的序號（1, 2, 3...）
      article      : str   「第 1 條」

    設計決策：每條內容不再細拆（實測每條均短於 500 字），
    保持法條語意完整，避免截斷關鍵定義。
    """
    chunks: list[dict] = []
    cur_chapter_num  = 0
    cur_chapter      = "（前言）"
    cur_article_num  = None
    cur_article_str  = None
    cur_lines: list[str] = []

    def _flush():
        nonlocal cur_article_num, cur_article_str, cur_lines
        if cur_article_num is None:
            return
        content = "\n".join(l for l in cur_lines if l.strip())
        if content.strip():
            chunks.append({
                "chunk_id"   : len(chunks),
                "chapter_num": cur_chapter_num,
                "chapter"    : cur_chapter,
                "article_num": cur_article_num,
                "article"    : cur_article_str,
                "text"       : content.strip(),
            })
        cur_article_num = None
        cur_article_str = None
        cur_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # 偵測章標題
        ch_m = _CHAPTER_RE.match(line)
        if ch_m:
            _flush()
            cur_chapter_num = _zh_to_int(ch_m.group(1))
            # 正規化章名（移除字元間多餘空格）
            zh_num  = ch_m.group(1).replace(" ", "")
            ch_name = ch_m.group(2).strip()
            cur_chapter = f"第{zh_num}章 {ch_name}"
            continue

        # 偵測條號
        art_m = _ARTICLE_RE.match(line)
        if art_m:
            _flush()
            cur_article_num = int(art_m.group(1))
            cur_article_str = f"第 {cur_article_num} 條"
            cur_lines = []
            continue

        # 條文內容（含項目編號如「1 本法所稱...」「一、...」）
        if cur_article_num is not None:
            cur_lines.append(line)

    _flush()  # 最後一條

    print(f"  切分完成：{len(chunks)} 個 chunks")
    for c in chunks:
        print(f"    [{c['chapter']}] {c['article']} — {len(c['text'])} 字")

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — LLM 語意標籤生成（Semantic Tag Enrichment）
# ══════════════════════════════════════════════════════════════════════════════

_TAG_PROMPT = """\
你是法律文件語意標記模組。為以下法條內容生成 6-8 個語意關鍵詞標籤，用於增強向量檢索精準度。

要求：
- 每個標籤 2-8 字，使用法律正式用語
- 必須涵蓋：法律責任類型（如刑事責任、民事賠償、行政罰鍰）、行為主體（行為人、機構、僱用人）、核心法律效果（時效、程序、罰則）
- 只輸出標籤，每行一個，不加任何解釋、標點或編號

法條內容：
{text}

關鍵詞標籤（每行一個）："""


def _generate_tags_for_chunk(text: str) -> list[str]:
    """
    呼叫 qwen2.5:14b 為單一 chunk 生成語意標籤。
    Fallback：若 LLM 失敗，回傳空列表（不中斷 ingest 流程）。
    """
    prompt = _TAG_PROMPT.format(text=text[:800])  # 截短避免超出 context
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model"   : LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream"  : False,
                    "options" : {"temperature": 0.05, "num_predict": 100},
                },
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"].strip()
            # 移除 <think> 區塊（qwen 思考模式）
            raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
            tags = [
                line.strip().lstrip("•-[]「」【】")
                for line in raw.splitlines()
                if line.strip() and len(line.strip()) <= 20
            ]
            return tags[:8]  # 最多保留 8 個
    except Exception:
        return []


def enrich_chunks_with_tags(chunks: list[dict]) -> None:
    """
    為每個 chunk 生成語意標籤並寫入 chunk dict：
      - chunk["tags"]       : list[str]  — 標籤列表（存入 metadata 供除錯）
      - chunk["embed_text"] : str        — 「[標籤1][標籤2]... 原始法條文字」
                                           用於 FAISS + BM25 索引，不送給 LLM 生成答案

    設計原則：
      原始 chunk["text"] 保持不變（LLM 回答時用此欄位，避免標籤汙染答案）
      標籤只用在檢索側（embed + BM25），查詢側完全不需要做關鍵詞匹配
    """
    print(f"  正在為 {len(chunks)} 個法條生成語意標籤（{LLM_MODEL}）...")
    failed = 0
    for i, chunk in enumerate(chunks):
        tags = _generate_tags_for_chunk(chunk["text"])
        chunk["tags"] = tags
        if tags:
            tag_prefix = "".join(f"[{t}]" for t in tags)
            chunk["embed_text"] = f"{tag_prefix} {chunk['text']}"
        else:
            chunk["embed_text"] = chunk["text"]
            failed += 1
        print(f"    [{i+1:02d}/{len(chunks)}] {chunk['article']} "
              f"→ {tags if tags else '⚠️  標籤生成失敗，使用原始文字'}")
    if failed:
        print(f"  ⚠️  {failed} 個 chunk 標籤生成失敗（已 fallback 至原始文字）")
    else:
        print(f"  ✅ 所有 chunk 標籤生成完成")


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Embedding（BAAI/bge-m3）
# ══════════════════════════════════════════════════════════════════════════════

def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
) -> np.ndarray:
    """
    將所有 chunk 的 embed_text（標籤 + 原文）轉為向量。
    - embed_text 包含 LLM 生成的語意標籤，增強向量空間的語意覆蓋
    - 查詢側不做任何標籤處理，由向量相似度自動跨越語彙差距
    - normalize_embeddings=True → L2 正規化後 Inner Product = Cosine Similarity
    """
    texts = [c.get("embed_text", c["text"]) for c in chunks]
    print(f"  正在生成 {len(texts)} 個向量（模型：{EMBED_MODEL}）...")
    vectors = model.encode(
        texts,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        batch_size=16,
        show_progress_bar=True,
    )
    return vectors.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — FAISS 索引
# ══════════════════════════════════════════════════════════════════════════════

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """
    建立 FAISS HNSW 索引。

    索引規格：
    - 類型：IndexHNSWFlat — 無需訓練的精確 HNSW 圖索引
    - 距離度量：METRIC_INNER_PRODUCT（L2 正規化後等價 cosine）
    - M=32：每節點連接數，平衡準確率與記憶體
    - efConstruction=200：建構時搜尋深度（越大越準，建構越慢）
    - efSearch=128：查詢時搜尋深度（越大越準，查詢越慢）
    - 向量維度：1024（bge-m3）
    - 向量型態：float32
    """
    dim   = vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch       = HNSW_EF_SEARCH
    index.add(vectors)

    print(f"  FAISS HNSW 索引建立完成")
    print(f"    向量數：{index.ntotal}，維度：{dim}，M：{HNSW_M}")
    print(f"    度量：METRIC_INNER_PRODUCT（cosine after L2 norm）")
    return index


# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — BM25 索引（混合檢索用）
# ══════════════════════════════════════════════════════════════════════════════

def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    建立 BM25 稀疏索引（關鍵字檢索）。
    使用 embed_text（標籤 + 原文）建立，讓標籤關鍵字也能被 BM25 命中。
    例如：查詢「刑事責任」可透過 BM25 直接命中含有 [刑事責任] 標籤的 chunk。
    """
    print("  正在建立 BM25 索引（jieba 斷詞，含語意標籤）...")
    tokenized = [jieba.lcut(c.get("embed_text", c["text"])) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"  BM25 索引建立完成，文件數：{len(tokenized)}")
    return bm25


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def ingest(pdf_path: Path) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"  法律文件 RAG — 資料建置流程")
    print(f"  來源：{pdf_path.name}")
    print(f"{'='*60}")

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: 解析 PDF
    print("\n[Step 1/6] PDF 解析")
    raw = extract_text_from_pdf(pdf_path)

    # Step 2: 清洗
    print("\n[Step 2/6] 文字清洗")
    clean = clean_text(raw)

    # Step 3: 法條切分
    print("\n[Step 3/7] 法條切分（章 + 條 兩層 Metadata）")
    chunks = parse_law_chunks(clean)
    if not chunks:
        print("  ❌ 未切出任何 chunk，請檢查 PDF 格式")
        sys.exit(1)

    # Step 4: LLM 語意標籤生成（Tag Enrichment）
    print(f"\n[Step 4/7] 語意標籤生成（{LLM_MODEL}）")
    print("  為每個法條生成 6-8 個語意標籤，融入 embed_text 供向量與 BM25 索引使用")
    print("  （原始 text 欄位不變，LLM 回答時仍使用乾淨的法條原文）")
    enrich_chunks_with_tags(chunks)

    # Step 5: 儲存 metadata（含標籤）
    print("\n[Step 5/7] 儲存 Metadata（含語意標籤）")
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  → {METADATA_PATH}")

    # Step 6: 載入 Embedding 模型
    print(f"\n[Step 6/7] 載入 Embedding 模型：{EMBED_MODEL}")
    print("  （首次執行會自動下載 ~570MB，之後完全離線）")
    embedder = SentenceTransformer(EMBED_MODEL)

    # Step 7a: 生成向量 + 建立 FAISS（使用 embed_text）
    print("\n[Step 7/7-a] 生成向量 & 建立 FAISS 索引")
    vectors = embed_chunks(chunks, embedder)
    faiss_index = build_faiss_index(vectors)
    faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
    print(f"  → {FAISS_INDEX_PATH}")

    # Step 7b: 建立 BM25（使用 embed_text）
    print("\n[Step 7/7-b] 建立 BM25 索引")
    bm25_index = build_bm25_index(chunks)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"  → {BM25_PATH}")

    # 完成摘要
    print(f"\n{'='*60}")
    print(f"  ✅ 建置完成！")
    print(f"  法條數量   : {len(chunks)}")
    print(f"  語意標籤   : {LLM_MODEL} 生成，平均 6-8 個/條，融入向量與 BM25")
    print(f"  Embedding  : {EMBED_MODEL}（{EMBED_DIM} 維，float32，L2 正規化）")
    print(f"  FAISS 索引  : HNSW M={HNSW_M}，度量 = Cosine via IP")
    print(f"  BM25 索引   : jieba 斷詞 + 語意標籤 + 法律術語自定義詞典")
    print(f"{'='*60}\n")

    return chunks


if __name__ == "__main__":
    # 決定 PDF 路徑
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        pdfs = sorted(DATA_DIR.glob("*.pdf"))
        if not pdfs:
            print(f"請將 PDF 放入 {DATA_DIR}/ 目錄，或指定路徑：")
            print(f"  python ingest.py <pdf路徑>")
            sys.exit(1)
        target = pdfs[0]

    if not target.exists():
        print(f"找不到檔案：{target}")
        sys.exit(1)

    ingest(target)
