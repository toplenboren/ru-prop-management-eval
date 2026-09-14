#!/usr/bin/env python3
"""Judge model answers against rubrics.

Reads:  results/<model>_answers.jsonl
Writes: results/<model>_<judge>_grades.jsonl

CLI:  python src/judge.py [judge_name ...] [--runs N]
       No judge names → run all judges from judges.jsonl
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from llm import call_llm, load_config, get_model_config

DATASET = ROOT / "data" / "dataset.tsv"
RESULTS = ROOT / "results"
PROMPTS = ROOT / "prompts"
CONFIG = ROOT / "judges.jsonl"

DEFAULT_JUDGE_PROMPT = (PROMPTS / "judge.txt").read_text(encoding="utf-8").strip()


def load_dataset(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}


def load_answers(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_judge_prompt(config: dict) -> str:
    override = config.get("judge_prompt_override")
    if override:
        return Path(override).read_text(encoding="utf-8").strip()
    return DEFAULT_JUDGE_PROMPT


def judge_one(judge_config: dict, judge_prompt: str, answer: dict, question: dict, run_idx: int) -> dict:
    rubric = json.loads(question["rubric"])
    
    # Two-layer rubric: fixed dimensions + question-specific criteria
    fixed = rubric.get("fixed", [])
    question_specific = rubric.get("question_specific", rubric if isinstance(rubric, list) else [])
    factual_max = rubric.get("factual_max", 5)
    
    specific_text = "\n".join(
        f"- {i+1}. {c['criterion']} ({c['points']} балл(ов))"
        for i, c in enumerate(question_specific)
    )
    fixed_text = "\n".join(
        f"- {d['dimension']} ({d['max']} балл(ов)): {d['description']}"
        for d in fixed
    )
    
    user_msg = f"""Вопрос:
{answer["question"]}

Эталонный ответ эксперта:
{question["expert_answer"]}

Ответ модели:
{answer["answer"]}

Специфичные критерии:
{specific_text}

Фиксированные критерии:
{fixed_text}

Оцени ответ модели по каждому критерию."""
    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": user_msg},
    ]
    try:
        resp = call_llm(config=judge_config, messages=messages)
        content = resp.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content
            content = content.rsplit("```", 1)[0]
        grade = json.loads(content)
        
        # Compute normalized factual correctness score
        scores_specific = grade.get("scores_specific", [])
        scores_fixed = grade.get("scores_fixed", [])
        
        specific_awarded = sum(s.get("awarded", 0) for s in scores_specific)
        specific_max = sum(s.get("max", 0) for s in scores_specific)
        if specific_max > 0:
            factual_awarded = (specific_awarded / specific_max) * factual_max
        else:
            factual_awarded = 0
        
        fixed_awarded = sum(s.get("awarded", 0) for s in scores_fixed)
        fixed_max = sum(s.get("max", 0) for s in scores_fixed)
        
        total_awarded = factual_awarded + fixed_awarded
        total_max = factual_max + fixed_max
        
        return {
            "id": answer["id"],
            "model": answer["model"],
            "judge": judge_config["name"],
            "run": run_idx,
            "scores_specific": scores_specific,
            "scores_fixed": scores_fixed,
            "factual_awarded": round(factual_awarded, 2),
            "factual_max": factual_max,
            "total_awarded": round(total_awarded, 2),
            "total_max": total_max,
            "hallucination_flag": grade.get("hallucination_flag", False),
            "overall_comment": grade.get("overall_comment", ""),
            "elapsed_s": resp.elapsed_s,
            "error": None,
        }
    except Exception as e:
        return {
            "id": answer["id"],
            "model": answer["model"],
            "judge": judge_config["name"],
            "run": run_idx,
            "scores_specific": [],
            "scores_fixed": [],
            "factual_awarded": 0,
            "factual_max": factual_max,
            "total_awarded": 0,
            "total_max": factual_max + sum(d.get("max", 0) for d in fixed),
            "hallucination_flag": False,
            "overall_comment": "",
            "elapsed_s": 0,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Judge model answers against rubrics")
    parser.add_argument("judges", nargs="*", help="Judge names from judges.jsonl (default: all)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per question (default: 3)")
    args = parser.parse_args()

    all_configs = load_config(CONFIG)
    if args.judges:
        judge_configs = [get_model_config(name, all_configs) for name in args.judges]
    else:
        judge_configs = all_configs

    concurrency = int(os.environ.get("CONCURRENCY", "5"))
    dataset = load_dataset(DATASET)

    answer_files = sorted(RESULTS.glob("*_answers.jsonl"))
    if not answer_files:
        print(f"No *_answers.jsonl in {RESULTS}", file=sys.stderr)
        sys.exit(1)

    for judge_config in judge_configs:
        judge_name = judge_config["name"]
        judge_prompt = get_judge_prompt(judge_config)
        print(f"\n=== Judge: {judge_name} ===")

        for answers_path in answer_files:
            model_slug = answers_path.stem.replace("_answers", "")
            grades_path = RESULTS / f"{model_slug}_{judge_name}_grades.jsonl"
            print(f"\n[judging {model_slug}] → {grades_path}")

            answers = load_answers(answers_path)

            # Resume: skip (id, run) pairs already done
            done_pairs: set[tuple[str, int]] = set()
            if grades_path.exists():
                with open(grades_path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            g = json.loads(line)
                            done_pairs.add((g["id"], g["run"]))
                print(f"  resuming: {len(done_pairs)} already graded")

            # Build pending tasks: (answer, run_idx) for each answer × run
            pending = []
            for a in answers:
                if not a.get("answer"):
                    continue
                if a["id"] not in dataset:
                    continue
                for run_idx in range(args.runs):
                    if (a["id"], run_idx) not in done_pairs:
                        pending.append((a, run_idx))

            if not pending:
                print(f"  all {len(answers)} answers already graded ({args.runs} runs each), skipping")
                continue

            with open(grades_path, "a", encoding="utf-8") as out:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = {
                        pool.submit(judge_one, judge_config, judge_prompt, a, dataset[a["id"]], run_idx): (a, run_idx)
                        for a, run_idx in pending
                    }
                    for i, future in enumerate(as_completed(futures), 1):
                        result = future.result()
                        out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out.flush()
                        status = "✓" if result["error"] is None else "✗"
                        score = f"{result['total_awarded']}/{result['total_max']}"
                        print(f"  {status} [{i}/{len(pending)}] id={result['id']} run={result['run']} {score} {result['elapsed_s']}s", flush=True)

            print(f"  done: {grades_path}")


if __name__ == "__main__":
    main()
