"""
rag.py — Adaptive RAG Pipeline v3.1（事實提取 + 問題拆解 + Cross-Query RRF）

流程：
  1. Fact Extraction + Query Decomposition
       — LLM 同步完成：(a) 從問題提取結構化事實條件，(b) 拆解 1-6 個子查詢
  2. Per-Sub-Query Hybrid Retrieval
       — 每個子查詢各自執行 BM25 + Dense + RRF
  3. Cross-Query RRF Merge
       — 跨子查詢合併去重，取 Top-K unique chunks
  4. Threshold Guard
       — 最佳 cosine < 閾值 → 直接拒答
  5. LLM Answer Generation
       — 事實條件 + 法條 context 同時送入，要求逐步比對後選擇正確款項
"""

from __future__ import annotations
import re
import json
import asyncio
import httpx

from retriever import HybridRetriever
from config import (
    OLLAMA_BASE_URL, LLM_MODEL,
    LLM_TIMEOUT, LLM_TEMPERATURE,
    TOP_K_RETRIEVAL, TOP_K_LLM, MAX_CONTEXT_CHARS,
    SIMILARITY_THRESHOLD, RRF_K, MAX_DECOMPOSED_QUERIES,
)

# ── Prompts ────────────────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """\
你是法律文件檢索系統的問題分析模組。請對以下用戶問題執行兩個任務，\
並嚴格按照格式輸出，不得輸出任何其他內容。

━━ 任務一：提取事實條件 ━━
從問題中提取可用於法條款項比對的關鍵事實，每條以「-」開頭。
若問題沒有具體事實（例如純粹詢問程序），此區塊可空白。

━━ 任務二：拆解子查詢 ━━
將問題拆解成 1-{max_q} 個子查詢，每個子查詢聚焦一個法律面向。
若問題涉及「法律責任」，刑事、行政、民事三類**必須各自獨立**為子查詢：
  ‣ 刑事示例：「行為人遭受何種刑事罰則（有期徒刑或罰金）」
  ‣ 行政示例：「機構或個人面臨之行政罰鍰金額及裁罰機關」
  ‣ 民事示例：「被害人得請求之損害賠償及懲罰性賠償金」
若涉及降職/差別待遇等附帶效果，亦獨立為子查詢。
每個子查詢使用正式法律術語，前面加數字編號（1. 2. 3.）。

━━ 嚴格輸出格式（不得偏離）━━
FACTS:
- [事實1]
- [事實2]

QUERIES:
1. [子查詢1]
2. [子查詢2]

━━ 用戶問題 ━━
{question}"""


_SYSTEM_PROMPT = """\
你是一個專業的法律知識助理，只負責解答「性騷擾防治法」相關問題。

【絕對規則】
1. 只能根據「參考法條」區塊中的內容回答，禁止使用任何外部知識或自行推論。
2. 回答時必須明確引用條號，例如：「根據第14條第1項」。
3. 若參考法條不足以完整回答問題，請如實說明哪些部分無法回答。
4. 若完全找不到相關資訊，直接回覆「無法回覆該問題」，不要猜測或編造。
5. 回答使用繁體中文，語氣專業但清晰易懂。
6. 不得引用「參考法條」以外的任何法律，包括刑法、民法、行政訴訟法等。"""

_USER_TEMPLATE = """\
【事實條件（供條款適用比對）】
{facts_block}

【參考法條】
{context}

【問題】
{question}

請依照以下步驟回答：
1. 逐步比對「事實條件」與各法條的適用情境，選擇正確的款項（例如第幾項第幾款）
2. 對問題的每個面向，分別引用對應條號作答
3. 在回答末尾，若有任何面向因參考法條資料不足而無法回答，請加上：
   ⚠️ 以下面向因參考法條資料不足，無法回答：[列出]"""

_NO_ANSWER = (
    "無法回覆該問題。\n"
    "（原因：知識庫中的相關法條與問題的語意相似度過低，"
    "建議換個問法或確認問題是否屬於性騷擾防治法範疇。）"
)


# ── 工具函數 ───────────────────────────────────────────────────────────────────

def _strip_think_tags(text: str) -> str:
    """移除 Qwen 系列模型的 <think>...</think> 內部推理區塊。"""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def _parse_decompose_output(raw: str, original: str) -> tuple[list[str], list[str]]:
    """
    解析 FACTS + QUERIES 格式的 LLM 輸出。

    回傳 (facts: list[str], sub_queries: list[str])

    容錯設計：
    - FACTS 區塊空白 → 回傳 []
    - QUERIES 解析失敗 → fallback 回 [original]
    - 任何格式異常 → 不中斷，盡量提取可用部分
    """
    facts: list[str] = []
    sub_queries: list[str] = []

    # 分割 FACTS / QUERIES 兩個區塊
    facts_match   = re.search(r"FACTS:\s*(.*?)(?=QUERIES:|$)", raw, re.DOTALL | re.IGNORECASE)
    queries_match = re.search(r"QUERIES:\s*(.*?)$",             raw, re.DOTALL | re.IGNORECASE)

    # 解析 FACTS
    if facts_match:
        for line in facts_match.group(1).splitlines():
            line = line.strip().lstrip("-•").strip()
            if line and len(line) <= 80:
                facts.append(line)

    # 解析 QUERIES
    if queries_match:
        for line in queries_match.group(1).splitlines():
            line = line.strip()
            if not line:
                continue
            # 去除數字編號前綴
            line = re.sub(r"^[\d]+[.）)、]\s*", "", line).strip()
            if not line or len(line) > 120:
                continue
            if line.endswith("：") or line.endswith(":"):
                continue
            sub_queries.append(line)

    sub_queries = sub_queries[:MAX_DECOMPOSED_QUERIES]

    if not sub_queries:
        sub_queries = [original]

    return facts, sub_queries


def _build_context(chunks: list[dict]) -> str:
    """
    將 chunks 格式化成 LLM 可讀的 context，控制總長度。
    使用原始 text 欄位（不含 ingest 時加的標籤），保持法條原文乾淨。
    """
    parts: list[str] = []
    total = 0

    for chunk in chunks[:TOP_K_LLM]:
        entry = (
            f"[{chunk['chapter']}｜{chunk['article']}]\n"
            f"{chunk['text']}"
        )
        if total + len(entry) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total
            if remaining > 100:
                entry = entry[:remaining] + "…（截斷）"
                parts.append(entry)
            break
        parts.append(entry)
        total += len(entry)

    return "\n\n---\n\n".join(parts)


# ── RAG Pipeline ───────────────────────────────────────────────────────────────

class RAGPipeline:

    def __init__(self):
        self.retriever = HybridRetriever()

    # ── Step 1: Fact Extraction + Query Decomposition ─────────────────────────

    async def _analyze_and_decompose(
        self, question: str
    ) -> tuple[list[str], list[str]]:
        """
        一次 LLM 呼叫完成兩件事：
          (a) 提取問題中的關鍵事實條件（用於答案生成時的條款比對）
          (b) 拆解 1-MAX_DECOMPOSED_QUERIES 個子查詢

        回傳 (facts, sub_queries)

        Fallback：失敗時回傳 ([], [original_question])
        """
        prompt = _DECOMPOSE_PROMPT.format(
            question=question,
            max_q=MAX_DECOMPOSED_QUERIES,
        )
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model"   : LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream"  : False,
                        "options" : {"temperature": 0.1, "num_predict": 300},
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["message"]["content"]
                raw = _strip_think_tags(raw)
                return _parse_decompose_output(raw, question)
        except httpx.ConnectError:
            print("  [WARN] 無法連接 Ollama，Decomposition 跳過，使用原始問題")
            return [], [question]
        except Exception as e:
            print(f"  [WARN] Decomposition 失敗（{type(e).__name__}），使用原始問題")
            return [], [question]

    # ── Step 5: LLM Answer Generation ────────────────────────────────────────

    async def _generate_answer(
        self, question: str, context: str, facts: list[str]
    ) -> str:
        """
        呼叫 Ollama 生成答案。
        facts 若非空，加入【事實條件】區塊幫助 LLM 比對正確款項。
        """
        if facts:
            facts_block = "\n".join(f"- {f}" for f in facts)
        else:
            facts_block = "（本問題無需特別事實比對）"

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_TEMPLATE.format(
                facts_block=facts_block,
                context=context,
                question=question,
            )},
        ]

        try:
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model"   : LLM_MODEL,
                        "messages": messages,
                        "stream"  : False,
                        "options" : {"temperature": LLM_TEMPERATURE},
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["message"]["content"]
                return _strip_think_tags(raw)
        except httpx.ConnectError:
            raise RuntimeError(
                "無法連接 Ollama。\n"
                "請執行：ollama serve\n"
                f"並確認模型已下載：ollama pull {LLM_MODEL}"
            )

    # ── 主要介面 ───────────────────────────────────────────────────────────────

    async def ask(self, question: str, verbose: bool = False) -> dict:
        """
        v3.1 完整 RAG 問答流程。

        回傳格式：
        {
            "question"   : str,
            "facts"      : list[str],   # 提取的事實條件
            "sub_queries": list[str],   # 拆解的子查詢
            "answer"     : str,
            "sources"    : list[dict],
            "best_score" : float,
            "answered"   : bool,
        }
        """
        if verbose:
            print(f"\n{'─'*50}")

        # ── Step 1: Fact Extraction + Query Decomposition ──────────────────────
        facts, sub_queries = await self._analyze_and_decompose(question)

        if verbose:
            print(f"[1] 原始問題  ：{question}")
            if facts:
                print(f"    事實條件  ：")
                for f in facts:
                    print(f"      · {f}")
            print(f"    子查詢數量：{len(sub_queries)}")
            for i, sq in enumerate(sub_queries, 1):
                print(f"    子查詢 {i}  ：{sq}")

        # ── Step 2: Per-Sub-Query Hybrid Retrieval（並行執行）─────────────────
        # retriever.search() 是同步 CPU 運算，用 run_in_executor 在 thread pool
        # 中並行執行，避免 6 次順序等待（省 ~2s，不影響品質）
        loop = asyncio.get_event_loop()

        async def _search_one(sq: str) -> tuple[str, list[dict], float]:
            chunks, score = await loop.run_in_executor(
                None, self.retriever.search, sq, TOP_K_RETRIEVAL
            )
            return sq, chunks, score

        sub_results: list[tuple[str, list[dict], float]] = await asyncio.gather(
            *[_search_one(sq) for sq in sub_queries]
        )

        if verbose:
            for sq, chunks, score in sub_results:
                print(f"[2] 子查詢檢索：「{sq[:28]}」→ {len(chunks)} chunks, cosine={score:.4f}")

        # ── Step 3: Cross-Query RRF Merge ──────────────────────────────────────
        chunk_rrf: dict[int, float] = {}
        chunk_map: dict[int, dict]  = {}

        for _, chunks, _ in sub_results:
            for rank, chunk in enumerate(chunks):
                cid = chunk["chunk_id"]
                chunk_map[cid] = chunk
                chunk_rrf[cid] = chunk_rrf.get(cid, 0.0) + 1.0 / (rank + RRF_K)

        sorted_ids   = sorted(chunk_rrf.keys(), key=lambda x: -chunk_rrf[x])
        final_chunks = [chunk_map[cid] for cid in sorted_ids[:TOP_K_LLM]]
        best_score   = max(score for _, _, score in sub_results)

        if verbose:
            print(f"[3] Cross-Query RRF：{len(chunk_rrf)} unique chunks → Top {TOP_K_LLM}")
            for c in final_chunks:
                print(f"    → [{c['chapter']}｜{c['article']}] RRF={chunk_rrf[c['chunk_id']]:.5f}")

        # ── Step 4: Threshold Guard ────────────────────────────────────────────
        if best_score < SIMILARITY_THRESHOLD:
            if verbose:
                print(f"[4] 拒答：best_score={best_score:.4f} < {SIMILARITY_THRESHOLD}")
            return {
                "question"   : question,
                "facts"      : facts,
                "sub_queries": sub_queries,
                "answer"     : _NO_ANSWER,
                "sources"    : [],
                "best_score" : round(best_score, 4),
                "answered"   : False,
            }

        # ── Step 5: Context 組裝 + LLM 生成答案 ───────────────────────────────
        context = _build_context(final_chunks)

        if verbose:
            print(f"[5] 送入 LLM  ：{LLM_MODEL}（context {len(context)} 字，{len(final_chunks)} 條）")
            if facts:
                print(f"    事實輔助  ：{len(facts)} 條事實條件一併送入")

        answer = await self._generate_answer(question, context, facts)

        return {
            "question"   : question,
            "facts"      : facts,
            "sub_queries": sub_queries,
            "answer"     : answer,
            "sources"    : [
                {"chapter": c["chapter"], "article": c["article"], "text": c["text"]}
                for c in final_chunks
            ],
            "best_score" : round(best_score, 4),
            "answered"   : True,
        }

    # ── 串流版介面 ─────────────────────────────────────────────────────────────────

    async def _generate_answer_stream(
        self, question: str, context: str, facts: list[str]
    ):
        """
        串流版答案生成。以 async generator 逐 token yield 內容字串。
        同時過濾 qwen 系列的 <think>...</think> 區塊。
        """
        if facts:
            facts_block = "\n".join(f"- {f}" for f in facts)
        else:
            facts_block = "（本問題無需特別事實比對）"

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_TEMPLATE.format(
                facts_block=facts_block,
                context=context,
                question=question,
            )},
        ]

        buf      = ""   # 跨 chunk 緩衝（用於過濾 <think> 標籤）
        in_think = False

        try:
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model"   : LLM_MODEL,
                        "messages": messages,
                        "stream"  : True,
                        "options" : {"temperature": LLM_TEMPERATURE},
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data    = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                        except json.JSONDecodeError:
                            continue
                        if not content:
                            continue

                        # 過濾 <think>...</think>
                        buf += content
                        while True:
                            if in_think:
                                end = buf.find("</think>")
                                if end == -1:
                                    buf = ""
                                    break
                                buf      = buf[end + 8:]
                                in_think = False
                            else:
                                start = buf.find("<think>")
                                if start == -1:
                                    if buf:
                                        yield buf
                                        buf = ""
                                    break
                                if start > 0:
                                    yield buf[:start]
                                buf      = buf[start + 7:]
                                in_think = True
        except httpx.ConnectError:
            raise RuntimeError(
                "無法連接 Ollama。\n"
                f"請執行：ollama serve\n"
                f"並確認模型已下載：ollama pull {LLM_MODEL}"
            )

    async def ask_stream(self, question: str):
        """
        串流版 RAG 問答（async generator）。

        yield 的事件格式：
          {"type": "meta",     "facts": [...], "sub_queries": [...]}
          {"type": "sources",  "sources": [...], "best_score": float}
          {"type": "token",    "content": str}
          {"type": "rejected", "message": str}
          {"type": "error",    "message": str}
          {"type": "done"}
        """
        # Step 1: Decompose
        facts, sub_queries = await self._analyze_and_decompose(question)
        yield {"type": "meta", "facts": facts, "sub_queries": sub_queries}

        # Step 2: Per-Sub-Query Retrieval（並行）
        loop = asyncio.get_event_loop()

        async def _search_one(sq: str) -> tuple[str, list[dict], float]:
            chunks, score = await loop.run_in_executor(
                None, self.retriever.search, sq, TOP_K_RETRIEVAL
            )
            return sq, chunks, score

        sub_results = await asyncio.gather(*[_search_one(sq) for sq in sub_queries])

        # Step 3: Cross-Query RRF
        chunk_rrf: dict[int, float] = {}
        chunk_map: dict[int, dict]  = {}
        for _, chunks, _ in sub_results:
            for rank, chunk in enumerate(chunks):
                cid = chunk["chunk_id"]
                chunk_map[cid] = chunk
                chunk_rrf[cid] = chunk_rrf.get(cid, 0.0) + 1.0 / (rank + RRF_K)

        sorted_ids   = sorted(chunk_rrf.keys(), key=lambda x: -chunk_rrf[x])
        final_chunks = [chunk_map[cid] for cid in sorted_ids[:TOP_K_LLM]]
        best_score   = max(score for _, _, score in sub_results)

        # Step 4: Threshold Guard
        if best_score < SIMILARITY_THRESHOLD:
            yield {"type": "rejected", "message": _NO_ANSWER}
            return

        # Step 5: 先送法條來源，再串流答案
        sources = [
            {"chapter": c["chapter"], "article": c["article"], "text": c["text"]}
            for c in final_chunks
        ]
        yield {"type": "sources", "sources": sources, "best_score": round(best_score, 4)}

        context = _build_context(final_chunks)
        async for token in self._generate_answer_stream(question, context, facts):
            yield {"type": "token", "content": token}

        yield {"type": "done"}
