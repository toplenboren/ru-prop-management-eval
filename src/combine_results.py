#!/usr/bin/env python3
"""Combine grades JSONL files into TSV files.

Reads:  results/<model>_<judge>_grades.jsonl
        results/<model>_answers.jsonl
        data/dataset.tsv (for categories, question text)
        llms_config.jsonl (for pricing)
Writes: results/grades_raw.tsv  (per-run, one row per model×question×run)
        results/grades.tsv     (aggregated, one row per model×question)
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "dataset.tsv"
RESULTS = ROOT / "results"
CONFIG = ROOT / "llms_config.jsonl"

RAW_OUT = RESULTS / "grades_raw.tsv"
AGG_OUT = RESULTS / "grades.tsv"

# Columns for grades_raw.tsv
RAW_COLUMNS = [
    "model", "judge", "harness", "question_id", "run",
    "total_awarded", "total_max", "hallucination_flag",
    "factual_awarded", "factual_max",
    "legal_grounding_awarded", "legal_grounding_max",
    "completeness_awarded", "completeness_max",
    "actionability_awarded", "actionability_max",
    "confidence_awarded", "confidence_max",
    "criteria_scores", "fixed_scores", "overall_comment", "elapsed_s",
    "prompt_tokens", "completion_tokens", "total_tokens", "cost",
    "tool_calls", "tool_errors", "tool_latency_s", "tool_documents_touched",
]

# Columns for grades.tsv (aggregated)
AGG_COLUMNS = [
    "model", "judge", "harness", "question_id", "category", "question", "n_runs",
    "total_awarded_avg", "total_awarded_min", "total_awarded_max",
    "total_max_avg", "total_max_min", "total_max_max",
    "score_avg", "score_min", "score_max",
    "pass_flag_avg", "hallucination_flag_avg",
    "factual_avg", "factual_min", "factual_max",
    "legal_grounding_avg", "legal_grounding_min", "legal_grounding_max",
    "completeness_avg", "completeness_min", "completeness_max",
    "actionability_avg", "actionability_min", "actionability_max",
    "confidence_avg", "confidence_min", "confidence_max",
    "criteria_scores",
    "prompt_tokens_avg", "prompt_tokens_min", "prompt_tokens_max",
    "completion_tokens_avg", "completion_tokens_min", "completion_tokens_max",
    "total_tokens_avg", "total_tokens_min", "total_tokens_max",
    "cost_avg", "cost_min", "cost_max",
    "elapsed_s_avg", "elapsed_s_min", "elapsed_s_max",
    "tool_calls_avg", "tool_errors_avg", "tool_latency_s_avg",
]


def load_config(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["name"]: json.loads(line) for line in f if line.strip()}


def load_dataset(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def calc_cost(config: dict, usage: dict) -> float:
    in_price = config.get("price_1m_input", 0) or 0
    out_price = config.get("price_1m_output", 0) or 0
    pt = usage.get("prompt_tokens") or 0
    ct = usage.get("completion_tokens") or 0
    return (pt / 1_000_000 * in_price) + (ct / 1_000_000 * out_price)


def match_model_name(stem: str, configs: dict[str, dict]) -> str | None:
    """Match a grades file stem to a model name from config.

    Handles harness suffixes: 'model__local-docs_judge' → 'model'
    """
    # Split harness suffix if present (model__harness_judge)
    if "__" in stem:
        base = stem.split("__", 1)[0]
        for cfg_name in configs:
            if base == cfg_name:
                return cfg_name
        return base
    for cfg_name in configs:
        if stem.startswith(cfg_name + "_"):
            return cfg_name
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else None


def parse_harness_from_stem(stem: str) -> str:
    """Extract harness from file stem.

    'model__local-docs_judge' → 'local-docs'
    'model_judge' → 'closedbook'
    """
    if "__" in stem:
        rest = stem.split("__", 1)[1]
        # harness is everything before the first _ (judge name follows)
        return rest.split("_", 1)[0]
    return "closedbook"


def parse_harness_from_answer(answer: dict) -> str:
    """Extract harness from an answer record."""
    return answer.get("harness", "closedbook")


def collect_raw():
    """Collect all raw grade records with usage/cost data joined in."""
    configs = load_config(CONFIG)
    dataset = load_dataset(DATASET)

    grade_files = sorted(RESULTS.glob("*_*_grades.jsonl"))
    if not grade_files:
        print(f"No *_grades.jsonl in {RESULTS}", file=sys.stderr)
        sys.exit(1)

    raw_rows = []
    for path in grade_files:
        stem = path.stem.replace("_grades", "")
        model_name = match_model_name(stem, configs)
        if model_name is None:
            print(f"  SKIP (unmatched): {path.name}", file=sys.stderr)
            continue

        harness = parse_harness_from_stem(stem)

        grades = load_jsonl(path)
        # Answer file name matches the grades file name pattern
        answers_path = RESULTS / f"{stem}_answers.jsonl"
        answers = {a["id"]: a for a in load_jsonl(answers_path)} if answers_path.exists() else {}
        cfg = configs.get(model_name, {})

        for g in grades:
            if g.get("error") is not None:
                continue
            ans = answers.get(g["id"], {})
            # Skip oracle questions that were skipped (no oracle evidence)
            oracle_meta = ans.get("oracle", {})
            if oracle_meta.get("skipped"):
                continue
            usage = ans.get("usage", {})
            tool_metrics = ans.get("tool_metrics", {})
            
            # Extract per-dimension scores from fixed scores
            fixed_scores = g.get("scores_fixed", [])
            dim_scores = {}
            for s in fixed_scores:
                dim = s.get("dimension", "")
                dim_scores[dim] = (s.get("awarded", 0), s.get("max", 0))
            
            raw_rows.append({
                "model": g["model"],
                "judge": g["judge"],
                "harness": harness,
                "question_id": g["id"],
                "run": g.get("run", 0),
                "total_awarded": g["total_awarded"],
                "total_max": g["total_max"],
                "hallucination_flag": g.get("hallucination_flag", False),
                "factual_awarded": g.get("factual_awarded", 0),
                "factual_max": g.get("factual_max", 5),
                "legal_grounding_awarded": dim_scores.get("legal_grounding", (0, 2))[0],
                "legal_grounding_max": dim_scores.get("legal_grounding", (0, 2))[1],
                "completeness_awarded": dim_scores.get("completeness", (0, 1))[0],
                "completeness_max": dim_scores.get("completeness", (0, 1))[1],
                "actionability_awarded": dim_scores.get("actionability", (0, 1))[0],
                "actionability_max": dim_scores.get("actionability", (0, 1))[1],
                "confidence_awarded": dim_scores.get("confidence", (0, 1))[0],
                "confidence_max": dim_scores.get("confidence", (0, 1))[1],
                "criteria_scores": json.dumps(g.get("scores_specific", g.get("scores", [])), ensure_ascii=False),
                "fixed_scores": json.dumps(fixed_scores, ensure_ascii=False),
                "overall_comment": g.get("overall_comment", ""),
                "elapsed_s": g.get("elapsed_s", 0),
                "prompt_tokens": usage.get("prompt_tokens") or 0,
                "completion_tokens": usage.get("completion_tokens") or 0,
                "total_tokens": usage.get("total_tokens") or 0,
                "cost": calc_cost(cfg, usage),
                "tool_calls": tool_metrics.get("calls", 0),
                "tool_errors": tool_metrics.get("errors", 0),
                "tool_latency_s": tool_metrics.get("latency_s", 0),
                "tool_documents_touched": json.dumps(tool_metrics.get("documents_touched", []), ensure_ascii=False),
            })

    return raw_rows, dataset


def aggregate(raw_rows: list[dict], dataset: dict[str, dict]) -> list[dict]:
    """Aggregate raw rows by (model, harness, question_id) with avg/min/max."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in raw_rows:
        groups[(r["model"], r["harness"], r["question_id"])].append(r)

    def dim_stats(runs, key):
        vals = [r[key] for r in runs]
        return sum(vals) / n, min(vals), max(vals)

    agg_rows = []
    for (model, harness, qid), runs in groups.items():
        n = len(runs)
        cat = dataset.get(qid, {}).get("category", "")
        question = dataset.get(qid, {}).get("question", "")

        awarded = [r["total_awarded"] for r in runs]
        max_pts = [r["total_max"] for r in runs]
        scores = [a / m if m > 0 else 0 for a, m in zip(awarded, max_pts)]
        passes = [1 if s >= 0.5 else 0 for s in scores]
        hallucs = [1 if r["hallucination_flag"] else 0 for r in runs]
        pt = [r["prompt_tokens"] for r in runs]
        ct = [r["completion_tokens"] for r in runs]
        tt = [r["total_tokens"] for r in runs]
        costs = [r["cost"] for r in runs]
        elapsed = [r["elapsed_s"] for r in runs]
        tc = [r["tool_calls"] for r in runs]
        te = [r["tool_errors"] for r in runs]
        tl = [r["tool_latency_s"] for r in runs]

        # Per-dimension stats
        f_avg, f_min, f_max = dim_stats(runs, "factual_awarded")
        lg_avg, lg_min, lg_max = dim_stats(runs, "legal_grounding_awarded")
        cp_avg, cp_min, cp_max = dim_stats(runs, "completeness_awarded")
        ac_avg, ac_min, ac_max = dim_stats(runs, "actionability_awarded")
        cf_avg, cf_min, cf_max = dim_stats(runs, "confidence_awarded")

        judge = runs[0]["judge"]
        criteria = runs[0]["criteria_scores"]

        agg_rows.append({
            "model": model,
            "judge": judge,
            "harness": harness,
            "question_id": qid,
            "category": cat,
            "question": question,
            "n_runs": n,
            "total_awarded_avg": sum(awarded) / n,
            "total_awarded_min": min(awarded),
            "total_awarded_max": max(awarded),
            "total_max_avg": sum(max_pts) / n,
            "total_max_min": min(max_pts),
            "total_max_max": max(max_pts),
            "score_avg": sum(scores) / n,
            "score_min": min(scores),
            "score_max": max(scores),
            "pass_flag_avg": sum(passes) / n,
            "hallucination_flag_avg": sum(hallucs) / n,
            "factual_avg": f_avg, "factual_min": f_min, "factual_max": f_max,
            "legal_grounding_avg": lg_avg, "legal_grounding_min": lg_min, "legal_grounding_max": lg_max,
            "completeness_avg": cp_avg, "completeness_min": cp_min, "completeness_max": cp_max,
            "actionability_avg": ac_avg, "actionability_min": ac_min, "actionability_max": ac_max,
            "confidence_avg": cf_avg, "confidence_min": cf_min, "confidence_max": cf_max,
            "criteria_scores": criteria,
            "prompt_tokens_avg": sum(pt) / n,
            "prompt_tokens_min": min(pt),
            "prompt_tokens_max": max(pt),
            "completion_tokens_avg": sum(ct) / n,
            "completion_tokens_min": min(ct),
            "completion_tokens_max": max(ct),
            "total_tokens_avg": sum(tt) / n,
            "total_tokens_min": min(tt),
            "total_tokens_max": max(tt),
            "cost_avg": sum(costs) / n,
            "cost_min": min(costs),
            "cost_max": max(costs),
            "elapsed_s_avg": sum(elapsed) / n,
            "elapsed_s_min": min(elapsed),
            "elapsed_s_max": max(elapsed),
            "tool_calls_avg": sum(tc) / n,
            "tool_errors_avg": sum(te) / n,
            "tool_latency_s_avg": sum(tl) / n,
        })

    return agg_rows


def write_tsv(path: Path, columns: list[str], rows: list[dict]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in columns})


def main():
    print("Collecting raw grades...")
    raw_rows, dataset = collect_raw()
    print(f"  {len(raw_rows)} raw grade records")

    print("Aggregating...")
    agg_rows = aggregate(raw_rows, dataset)
    print(f"  {len(agg_rows)} aggregated rows")

    print(f"Writing {RAW_OUT}...")
    write_tsv(RAW_OUT, RAW_COLUMNS, raw_rows)

    print(f"Writing {AGG_OUT}...")
    write_tsv(AGG_OUT, AGG_COLUMNS, agg_rows)

    print("Done!")


if __name__ == "__main__":
    main()
