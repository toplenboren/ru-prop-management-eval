#!/usr/bin/env python3

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from config import parse_config, load_dataset
from providers import get_llm_provider

DATASET = ROOT / "data" / "dataset.tsv"
RESULTS = ROOT / "results"
PROMPTS = ROOT / "prompts"
CONFIG = ROOT / "llms_config.jsonl"

DEFAULT_SYSTEM_PROMPT = (PROMPTS / "system.txt").read_text(encoding="utf-8").strip()


def get_system_prompt(config: dict) -> str:
    override = config.get("system_prompt_override")
    if override:
        return Path(override).read_text(encoding="utf-8").strip()
    return DEFAULT_SYSTEM_PROMPT


def run_closedbook(config, provider, system: str, row: dict) -> dict:
    """Closed-book: model receives only the question."""
    question_id = row["id"]
    question = row["question"]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    try:
        resp = provider.call_llm(messages=messages, **config.get("params", {}))
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


def main():
    parser = argparse.ArgumentParser(description="Run LLM models on the dataset")
    parser.add_argument("models", nargs="*", help="Model names from llms_config.jsonl. Runs on all models if not specified")
    args = parser.parse_args()

    configs = parse_config(CONFIG, args.models)

    limit = os.environ.get("LIMIT")
    limit = int(limit) if limit else None

    rows = list(load_dataset(DATASET).values())
    if limit:
        rows = rows[:limit]

    RESULTS.mkdir(exist_ok=True)

    for config in configs:
        name = config["name"]
        provider = get_llm_provider(config["api"], **config["provider"])
        out_path = RESULTS / f"{name}_answers.jsonl"
        system = get_system_prompt(config)
        print(f"\n[{name}] {len(rows)} questions → {out_path}")

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

        workers = int(config.get("concurrency") or os.environ.get("CONCURRENCY") or 5)
        with open(out_path, "a", encoding="utf-8") as out:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(run_closedbook, config, provider, system, r): r
                    for r in pending
                }
                for i, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    out.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                    out.flush()
                    status = "✓" if result["error"] is None else "✗"
                    print(f"  {status} [{i}/{len(pending)}] id={result['id']} {result['elapsed_s']}s", flush=True)

        print(f"  done: {out_path}")


if __name__ == "__main__":
    main()
