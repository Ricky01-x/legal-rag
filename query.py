#!/usr/bin/env python3
"""
query.py — CLI 問答介面

用法：
    python query.py "申訴期限是多久？"           # 單次問答
    python query.py -i                           # 互動模式
    python query.py "問題" --json                # JSON 輸出（接前端用）
    python query.py "問題" --verbose             # 顯示完整 pipeline 過程
"""

from __future__ import annotations
import asyncio
import argparse
import json
import sys

from rag import RAGPipeline


# ── 輸出格式化 ─────────────────────────────────────────────────────────────────

SEP  = "=" * 60
LINE = "-" * 60


def print_result(result: dict) -> None:
    print(f"\n{SEP}")

    # 問題、事實條件、子查詢
    print(f"問題     ：{result['question']}")
    facts = result.get("facts", [])
    if facts:
        print(f"事實條件  ：")
        for f in facts:
            print(f"  · {f}")
    sub_qs = result.get("sub_queries", [])
    if len(sub_qs) == 1 and sub_qs[0] != result["question"]:
        print(f"查詢改寫  ：{sub_qs[0]}")
    elif len(sub_qs) > 1:
        print(f"子查詢拆解（{len(sub_qs)} 個）：")
        for i, sq in enumerate(sub_qs, 1):
            print(f"  {i}. {sq}")
    print(f"相似度分數：{result['best_score']:.4f}", end="")
    print(f"  {'✅ 回答' if result['answered'] else '❌ 拒答'}")
    print(LINE)

    # 答案
    print(f"\n{result['answer']}\n")

    # 來源法條
    if result["sources"]:
        print(LINE)
        print("來源法條：")
        for s in result["sources"]:
            print(f"\n  [{s['chapter']}｜{s['article']}]")
            preview = s["text"]
            if len(preview) > 250:
                preview = preview[:250] + "…"
            # 縮排顯示
            for line in preview.splitlines():
                print(f"  {line}")

    print(f"\n{SEP}")


# ── 模式：單次問答 ─────────────────────────────────────────────────────────────

async def run_once(question: str, pipeline: RAGPipeline, as_json: bool, verbose: bool):
    result = await pipeline.ask(question, verbose=verbose)
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_result(result)


# ── 模式：互動 ─────────────────────────────────────────────────────────────────

async def run_interactive(pipeline: RAGPipeline, verbose: bool):
    print(f"\n{'='*60}")
    print("  性騷擾防治法 — 知識檢索系統")
    print(f"{'='*60}")
    print("  輸入問題後按 Enter，輸入 quit / exit 離開\n")

    while True:
        try:
            question = input("問題：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q", "離開", "退出"):
            print("再見！")
            break

        try:
            result = await pipeline.ask(question, verbose=verbose)
            print_result(result)
        except RuntimeError as e:
            print(f"\n❌ 錯誤：{e}")
            break
        except Exception as e:
            print(f"\n⚠️  發生問題：{e}")

    print()


# ── 入口 ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="性騷擾防治法 RAG 問答系統（本地端，完全離線）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python query.py "雇主對員工性騷擾的罰則是什麼？"
  python query.py -i
  python query.py "申訴期限" --verbose
  python query.py "申訴期限" --json
        """,
    )
    parser.add_argument("question", nargs="?", help="要查詢的問題")
    parser.add_argument("-i", "--interactive", action="store_true", help="進入互動模式")
    parser.add_argument("--json",    action="store_true", help="輸出純 JSON（供前端使用）")
    parser.add_argument("--verbose", action="store_true", help="顯示完整 pipeline 過程")
    args = parser.parse_args()

    # 初始化 pipeline（載入所有索引）
    try:
        pipeline = RAGPipeline()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    # 決定執行模式
    if args.interactive or not args.question:
        asyncio.run(run_interactive(pipeline, verbose=args.verbose))
    else:
        asyncio.run(run_once(args.question, pipeline, as_json=args.json, verbose=args.verbose))


if __name__ == "__main__":
    main()
