#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from llm import call_llm, call_llm_with_tools, load_config, get_model_config
from tools import (
    HARNESS, ORACLE_HARNESS, TOOL_SCHEMAS_CHAT,
    dispatch, make_ctx, get_oracle_articles, parse_citations,
)

DATASET = ROOT / "data" / "dataset.tsv"
RESULTS = ROOT / "results"
PROMPTS = ROOT / "prompts"
CONFIG = ROOT / "llms_config.jsonl"

DEFAULT_SYSTEM_PROMPT = (PROMPTS / "system.txt").read_text(encoding="utf-8").strip()
ORACLE_SYSTEM_PROMPT = (PROMPTS / "system_oracle.txt").read_text(encoding="utf-8").strip()
LOCAL_DOCS_SYSTEM_PROMPT = (PROMPTS / "system_local-docs.txt").read_text(encoding="utf-8").strip()


def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def get_system_prompt(config: dict, harness: str) -> str:
    override = config.get("system_prompt_override")
    if override:
        return Path(override).read_text(encoding="utf-8").strip()
    if harness == "oracle":
        return ORACLE_SYSTEM_PROMPT
    if harness == "local-docs":
        return LOCAL_DOCS_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT


def run_closedbook(config: dict, system: str, row: dict) -> dict:
    """Closed-book: model receives only the question."""
    question_id = row["id"]
    question = row["question"]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    try:
        resp = call_llm(config=config, messages=messages)
        return {
            "id": question_id,
            "model": config["name"],
            "question": question,
            "answer": resp.content,
            "elapsed_s": resp.elapsed_s,
            "usage": resp.usage,
            "error": None,
            "harness": "closedbook",
        }
    except Exception as e:
        return {
            "id": question_id,
            "model": config["name"],
            "question": question,
            "answer": None,
            "elapsed_s": 0,
            "usage": {},
            "error": str(e),
            "harness": "closedbook",
        }


def run_oracle(config: dict, system: str, row: dict) -> dict:
    """Oracle: gold article text pasted into the prompt.

    Only runs on questions with resolved citations. Questions without
    oracle evidence are skipped (not fallen back to closed-book).
    """
    question_id = row["id"]
    question = row["question"]
    expert_answer = row.get("expert_answer", "")

    # Parse citations from expert answer and look up article text
    ctx = make_ctx(ORACLE_HARNESS)
    oracle_articles = get_oracle_articles(expert_answer, ctx)
    citations = parse_citations(expert_answer)

    # Skip questions without oracle evidence — no fallback to closed-book
    if not oracle_articles:
        return {
            "id": question_id,
            "model": config["name"],
            "question": question,
            "answer": None,
            "elapsed_s": 0,
            "usage": {},
            "error": None,
            "harness": "oracle",
            "oracle": {
                "citations": citations,
                "articles_found": 0,
                "skipped": True,
                "skip_reason": "no_oracle_evidence",
                "article_codes": [],
            },
        }

    oracle_text = ""
    parts = []
    for art in oracle_articles:
        parts.append(f"--- {art['code']}, {art['title']} ---\n{art['text']}")
    oracle_text = "\n\n".join(parts)

    user_content = f"Вопрос: {question}\n\n--- Релевантные нормативно-правовые акты ---\n{oracle_text}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    try:
        resp = call_llm(config=config, messages=messages)
        return {
            "id": question_id,
            "model": config["name"],
            "question": question,
            "answer": resp.content,
            "elapsed_s": resp.elapsed_s,
            "usage": resp.usage,
            "error": None,
            "harness": "oracle",
            "oracle": {
                "citations": citations,
                "articles_found": len(oracle_articles),
                "skipped": False,
                "article_codes": [a["code"] for a in oracle_articles],
            },
        }
    except Exception as e:
        return {
            "id": question_id,
            "model": config["name"],
            "question": question,
            "answer": None,
            "elapsed_s": 0,
            "usage": {},
            "error": str(e),
            "harness": "oracle",
            "oracle": {
                "citations": citations,
                "articles_found": len(oracle_articles),
                "skipped": False,
                "article_codes": [a["code"] for a in oracle_articles],
            },
        }


def run_local_docs(config: dict, system: str, row: dict) -> dict:
    """Local-docs: model uses tools to search the corpus."""
    question_id = row["id"]
    question = row["question"]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    ctx = make_ctx(HARNESS)
    max_tool_calls = HARNESS.get("max_tool_calls", 8)

    try:
        resp = call_llm_with_tools(
            config=config,
            messages=messages,
            tools=TOOL_SCHEMAS_CHAT,
            dispatch_fn=dispatch,
            ctx=ctx,
            max_tool_calls=max_tool_calls,
        )
        return {
            "id": question_id,
            "model": config["name"],
            "question": question,
            "answer": resp.content,
            "elapsed_s": resp.elapsed_s,
            "usage": resp.usage,
            "error": None,
            "harness": "local-docs",
            "trajectory": resp.trajectory,
            "tool_metrics": resp.tool_metrics,
        }
    except Exception as e:
        return {
            "id": question_id,
            "model": config["name"],
            "question": question,
            "answer": None,
            "elapsed_s": 0,
            "usage": {},
            "error": str(e),
            "harness": "local-docs",
            "trajectory": [],
            "tool_metrics": {"calls": 0, "errors": 0, "latency_s": 0, "documents_touched": []},
        }


def main():
    parser = argparse.ArgumentParser(description="Run LLM models on the dataset")
    parser.add_argument("models", nargs="*", help="Model names from llms_config.jsonl. Runs on all models if not specified")
    parser.add_argument("--harness", default="closedbook",
                        choices=["closedbook", "oracle", "local-docs"],
                        help="Evaluation condition: closedbook (default), oracle, or local-docs")
    args = parser.parse_args()

    all_configs = load_config(CONFIG)
    if args.models:
        configs = [get_model_config(name, all_configs) for name in args.models]
    else:
        configs = all_configs

    concurrency = int(os.environ.get("CONCURRENCY", "5"))
    limit = os.environ.get("LIMIT")
    limit = int(limit) if limit else None

    rows = load_dataset(DATASET)
    if limit:
        rows = rows[:limit]

    RESULTS.mkdir(exist_ok=True)

    # Determine output file suffix based on harness
    suffix = "" if args.harness == "closedbook" else f"__{args.harness}"

    # Select run function based on harness
    run_fn = {
        "closedbook": run_closedbook,
        "oracle": run_oracle,
        "local-docs": run_local_docs,
    }[args.harness]

    for config in configs:
        name = config["name"]
        out_path = RESULTS / f"{name}{suffix}_answers.jsonl"
        system = get_system_prompt(config, args.harness)
        print(f"\n[{name}] harness={args.harness} {len(rows)} questions → {out_path}")

        done_ids: set[str] = set()
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        done_ids.add(json.loads(line)["id"])
            print(f"  resuming: {len(done_ids)} already done")

        pending = [r for r in rows if r["id"] not in done_ids]
        if not pending:
            print(f"  all {len(rows)} questions already answered, skipping")
            continue

        with open(out_path, "a", encoding="utf-8") as out:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {
                    pool.submit(run_fn, config, system, r): r
                    for r in pending
                }
                for i, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    out.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                    out.flush()
                    status = "✓" if result["error"] is None else "✗"
                    extra = ""
                    if args.harness == "local-docs" and result.get("tool_metrics"):
                        tm = result["tool_metrics"]
                        extra = f" tools={tm['calls']}"
                    print(f"  {status} [{i}/{len(pending)}] id={result['id']} {result['elapsed_s']}s{extra}", flush=True)

        print(f"  done: {out_path}")


if __name__ == "__main__":
    main()
