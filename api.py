#!/usr/bin/env python3
"""
api.py — FastAPI REST API（加分項）

啟動：
    uvicorn api:app --reload --port 8000

端點：
    GET  /health       健康檢查
    POST /ask          問答（主要端點）
    GET  /stats        知識庫統計
"""

from __future__ import annotations
import json as _json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

from rag import RAGPipeline
from config import LLM_MODEL, EMBED_MODEL, SIMILARITY_THRESHOLD

# ── App 生命週期 ────────────────────────────────────────────────────────────────

_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """啟動時初始化 RAG pipeline（載入索引），關閉時釋放資源。"""
    global _pipeline
    print("[API] 正在初始化 RAG pipeline...")
    _pipeline = RAGPipeline()
    print("[API] 就緒，開始接收請求")
    yield
    _pipeline = None


app = FastAPI(
    title="性騷擾防治法 RAG API",
    description=(
        "本地端 LLM + Hybrid Retrieval（BM25 + Dense Vector + RRF）\n"
        "完全離線運行，不呼叫任何雲端 API。\n\n"
        f"LLM 模型：{LLM_MODEL} | Embedding：{EMBED_MODEL}"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS（為前端預留，接上 UI 後即可使用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="要查詢的自然語言問題",
        examples=["申訴性騷擾事件的期限是多久？"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {"question": "雇主對員工實施性騷擾，罰則為何？"}
        }
    }


class SourceItem(BaseModel):
    chapter: str = Field(description="章標題，如「第四章 申訴及調查程序」")
    article: str = Field(description="條號，如「第 14 條」")
    text   : str = Field(description="條文原文")


class AskResponse(BaseModel):
    question   : str              = Field(description="原始問題")
    facts      : list[str]        = Field(description="從問題提取的關鍵事實條件（用於條款比對）")
    sub_queries: list[str]        = Field(description="Adaptive Decomposition 拆解的子查詢列表（1-6個）")
    answer     : str              = Field(description="LLM 根據法條生成的答案")
    sources    : list[SourceItem] = Field(description="引用的法條來源列表")
    best_score : float            = Field(description="最佳 cosine 相似度分數 (0~1)")
    answered   : bool             = Field(description="是否成功回答（False 表示知識庫無相關資訊）")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question"   : "申訴期限是多久？",
                "sub_queries": ["性騷擾事件被害人提出申訴之法定期限"],
                "answer"     : "根據第14條第1項，一般性騷擾事件於知悉後2年內提出申訴...",
                "sources"    : [{"chapter": "第四章 申訴及調查程序", "article": "第 14 條", "text": "..."}],
                "best_score" : 0.7823,
                "answered"   : True,
            }
        }
    }


class StatsResponse(BaseModel):
    total_chunks     : int
    llm_model        : str
    embed_model      : str
    similarity_threshold: float
    status           : str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_index():
    """回傳展示網站首頁。"""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    try:
        with open(html_path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>找不到 index.html</h1><p>請先建立展示頁面。</p>", status_code=404)


@app.get("/health", summary="健康檢查")
async def health():
    """確認服務是否正常運行。"""
    return {"status": "ok", "pipeline_ready": _pipeline is not None}


@app.get("/stats", response_model=StatsResponse, summary="知識庫統計")
async def stats():
    """回傳知識庫與模型配置資訊。"""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")
    return StatsResponse(
        total_chunks        =len(_pipeline.retriever.chunks),
        llm_model           =LLM_MODEL,
        embed_model         =EMBED_MODEL,
        similarity_threshold=SIMILARITY_THRESHOLD,
        status              ="running",
    )


@app.post("/ask", response_model=AskResponse, summary="法律問答")
async def ask(req: AskRequest):
    """
    主要問答端點。

    - 自動進行 Query Rewriting
    - 使用 Hybrid Retrieval（BM25 + Dense Vector + RRF）
    - 相似度低於閾值時回傳拒答訊息，不產生無根據的答案
    - 回傳答案與引用法條原文
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")

    try:
        result = await _pipeline.ask(req.question)
        return AskResponse(
            question    =result["question"],
            facts       =result["facts"],
            sub_queries =result["sub_queries"],
            answer      =result["answer"],
            sources     =[SourceItem(**s) for s in result["sources"]],
            best_score  =result["best_score"],
            answered    =result["answered"],
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"內部錯誤：{str(e)}")


@app.post("/ask/stream", summary="法律問答（串流）")
async def ask_stream(req: AskRequest):
    """
    串流版問答端點，使用 Server-Sent Events（SSE）格式逐步回傳結果。

    事件類型（data 欄位 JSON）：
    - `meta`     — Decomposition 完成：facts + sub_queries
    - `sources`  — 檢索完成：法條來源列表 + best_score
    - `token`    — LLM 逐 token 輸出
    - `rejected` — 相似度過低，拒答
    - `error`    — 發生錯誤
    - `done`     — 串流結束
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")

    async def generate():
        try:
            async for event in _pipeline.ask_stream(req.question):
                yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
        except RuntimeError as e:
            yield f"data: {_json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {_json.dumps({'type': 'error', 'message': f'內部錯誤：{str(e)}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
