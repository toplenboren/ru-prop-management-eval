#!/usr/bin/env python3
"""Analyze graded results from grades.tsv.

Reads:  results/grades.tsv
Writes: stdout report + results/figures/*.png + optional results/report.md
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
GRADES_TSV = RESULTS / "grades.tsv"
REPORT_MD = RESULTS / "report.md"

COLORS = {
    "blue": "#2563eb",
    "red": "#dc2626",
    "green": "#16a34a",
    "purple": "#9333ea",
    "orange": "#ea580c",
    "cyan": "#0891b2",
    "yellow": "#ca8a04",
    "pink": "#be185d",
    "indigo": "#4f46e5",
    "lime": "#65a30d",
    "brown": "#7c2d12",
    "teal": "#0f766e",
}


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0, center - spread), min(1, center + spread)


def load_data(tsv_path):
    """Load grades.tsv into numpy arrays + metadata.

    Returns a dict with data grouped by harness. If only one harness exists,
    the structure is the same as before (backward compatible).
    """
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    if not rows:
        raise SystemExit(f"No data in {tsv_path}")

    # Unique (judge, harness) combinations — judges are never mixed
    combos = sorted(set(
        (r.get("judge", "?"), r.get("harness", "closedbook")) for r in rows
    ))

    # Group rows by (judge, harness)
    combo_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        combo_rows[(r.get("judge", "?"), r.get("harness", "closedbook"))].append(r)

    # Build per-combo data
    harness_data = {}
    for combo, crows in combo_rows.items():
        harness_data[combo] = _build_harness_data(crows)

    return {
        "combos": combos,
        "harness_data": harness_data,
    }


def _build_harness_data(rows: list[dict]) -> dict:
    """Build numpy arrays and metadata for a single harness's rows."""
    # Unique models and questions
    models = sorted(set(r["model"] for r in rows))
    questions = sorted(set(r["question_id"] for r in rows))
    model_idx = {m: i for i, m in enumerate(models)}
    q_idx = {q: i for i, q in enumerate(questions)}
    n_models = len(models)
    n_questions = len(questions)

    # Build lookup: (model, question) -> row
    lookup = {}
    for r in rows:
        lookup[(r["model"], r["question_id"])] = r

    # Per-question tags and question text (from first matching row)
    tags = {name: np.array([""] * n_questions, dtype=object) for name in ("category", "intent", "knowledge")}
    question_text = [""] * n_questions
    for r in rows:
        qi = q_idx[r["question_id"]]
        for name in tags:
            tags[name][qi] = r.get(name, "")
        question_text[qi] = r["question"]

    # 2D arrays [n_models, n_questions], NaN for missing
    def make_array(getter):
        arr = np.full((n_models, n_questions), np.nan)
        for (model, qid), r in lookup.items():
            mi = model_idx[model]
            qi = q_idx[qid]
            val = getter(r)
            if val != "" and val is not None:
                arr[mi, qi] = float(val)
        return arr

    awarded_avg = make_array(lambda r: r["total_awarded_avg"])
    max_avg = make_array(lambda r: r["total_max_avg"])
    score_avg = make_array(lambda r: r["score_avg"])
    score_min = make_array(lambda r: r["score_min"])
    score_max = make_array(lambda r: r["score_max"])
    fc_avg = make_array(lambda r: r.get("factual_correctness_avg", ""))
    legality_avg = make_array(lambda r: r.get("legality_avg", ""))
    factual_incorrect_avg = 1 - fc_avg  # incorrect rate (NaN propagates)
    pass_avg = make_array(lambda r: r["pass_flag_avg"])
    tokens_avg = make_array(lambda r: r["total_tokens_avg"])
    cost_avg = make_array(lambda r: r["cost_avg"])

    n_runs = make_array(lambda r: r["n_runs"])

    # Per-dimension scores (may not exist in older data)
    def make_array_safe(col, default=0):
        arr = np.full((n_models, n_questions), np.nan)
        for (model, qid), r in lookup.items():
            mi = model_idx[model]
            qi = q_idx[qid]
            val = r.get(col, default)
            if val != "" and val is not None:
                try:
                    arr[mi, qi] = float(val)
                except (ValueError, TypeError):
                    arr[mi, qi] = default
        return arr

    criteria_avg = make_array_safe("criteria_avg")
    task_avg = make_array_safe("task_completion_avg")  # flag rate 0–1

    # Per-model totals
    total_awarded = np.nansum(awarded_avg, axis=1)
    total_max = np.nansum(max_avg, axis=1)
    overall_pct = np.where(total_max > 0, total_awarded / total_max * 100, 0)
    n_answered = np.sum(~np.isnan(awarded_avg), axis=1)
    total_tokens = np.nansum(tokens_avg, axis=1)
    total_cost = np.nansum(cost_avg, axis=1)
    total_factual_incorrect = np.nansum(factual_incorrect_avg, axis=1)
    # Per-dimension totals (normalize to 0-100% for comparison)
    dim_criteria = np.nanmean(criteria_avg, axis=1) / 5 * 100 if np.any(~np.isnan(criteria_avg)) else np.zeros(n_models)
    dim_task = np.nanmean(task_avg, axis=1) * 100 if np.any(~np.isnan(task_avg)) else np.zeros(n_models)
    dim_legal = np.nanmean(legality_avg, axis=1) * 100 if np.any(~np.isnan(legality_avg)) else np.zeros(n_models)
    dim_factcorr = np.nanmean(fc_avg, axis=1) * 100 if np.any(~np.isnan(fc_avg)) else np.zeros(n_models)

    # Sort by overall pct ascending
    sort_idx = np.argsort(overall_pct)
    models = [models[i] for i in sort_idx]
    awarded_avg = awarded_avg[sort_idx]
    max_avg = max_avg[sort_idx]
    score_avg = score_avg[sort_idx]
    score_min = score_min[sort_idx]
    score_max = score_max[sort_idx]
    pass_avg = pass_avg[sort_idx]
    tokens_avg = tokens_avg[sort_idx]
    cost_avg = cost_avg[sort_idx]
    n_runs = n_runs[sort_idx]
    overall_pct = overall_pct[sort_idx]
    total_awarded = total_awarded[sort_idx]
    total_max = total_max[sort_idx]
    n_answered = n_answered[sort_idx]
    total_tokens = total_tokens[sort_idx]
    total_cost = total_cost[sort_idx]
    total_factual_incorrect = total_factual_incorrect[sort_idx]
    dim_criteria = dim_criteria[sort_idx]
    dim_task = dim_task[sort_idx]
    dim_legal = dim_legal[sort_idx]
    dim_factcorr = dim_factcorr[sort_idx]

    return {
        "models": models,
        "questions": questions,
        "question_text": question_text,
        "tags": tags,
        "awarded_avg": awarded_avg,
        "max_avg": max_avg,
        "score_avg": score_avg,
        "score_min": score_min,
        "score_max": score_max,
        "pass_avg": pass_avg,
        "tokens_avg": tokens_avg,
        "cost_avg": cost_avg,
        "n_runs": n_runs,
        "overall_pct": overall_pct,
        "total_awarded": total_awarded,
        "total_max": total_max,
        "n_answered": n_answered,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "total_factual_incorrect": total_factual_incorrect,
        "dim_criteria": dim_criteria,
        "dim_task": dim_task,
        "dim_legal": dim_legal,
        "dim_factcorr": dim_factcorr,
    }


def _report_harness(d: dict, judge: str, harness: str, report: list, figures: list):
    """Generate report sections for a single (judge, harness) combination."""
    models = d["models"]
    n_models = len(models)
    questions = d["questions"]
    question_text = d["question_text"]
    categories = d["tags"]["category"]
    score_avg = d["score_avg"]
    score_min = d["score_min"]
    score_max = d["score_max"]
    awarded_avg = d["awarded_avg"]
    max_avg = d["max_avg"]
    pass_avg = d["pass_avg"]
    overall_pct = d["overall_pct"]
    total_awarded = d["total_awarded"]
    total_max = d["total_max"]
    n_answered = d["n_answered"]
    total_tokens = d["total_tokens"]
    total_cost = d["total_cost"]
    total_factual_incorrect = d["total_factual_incorrect"]
    dim_criteria = d.get("dim_criteria", np.zeros(n_models))
    dim_task = d.get("dim_task", np.zeros(n_models))
    dim_legal = d.get("dim_legal", np.zeros(n_models))
    dim_factcorr = d.get("dim_factcorr", np.zeros(n_models))

    harness_label = f"{judge} · {(harness or 'closedbook').upper()}"

    # --- Section 1: Leaderboard ---
    report.append(f"\n{'='*100}")
    report.append(f"{harness_label} — Leaderboard ({len(questions)} questions)")
    report.append(f"{'='*100}")
    report.append(f"\n{'Model':<30} {'Score':>10} {'Pct':>7} {'95% CI':>14} {'Tokens':>10} {'Cost':>10} {'FactErr':>7} {'N':>4}")
    report.append("-" * 100)

    for i, model in enumerate(models):
        n = int(n_answered[i])
        passes = int(np.nansum(pass_avg[i]))
        lo, hi = wilson_ci(passes, n)
        h = int(total_factual_incorrect[i])
        report.append(f"{model:<30} {f'{total_awarded[i]:.0f}/{total_max[i]:.0f}':>10} {overall_pct[i]:>6.1f}% {f'{lo*100:.1f}-{hi*100:.1f}%':>14} {int(total_tokens[i]):>10,} {total_cost[i]:>10.2f} {h:>7} {n:>4}")

    # --- Section 2: Score distribution ---
    report.append(f"\n{'='*100}")
    report.append(f"{harness_label} — SCORE DISTRIBUTION (per-question score_avg)")
    report.append(f"{'='*100}")
    report.append(f"\n{'Model':<30} {'Mean':>7} {'Median':>7} {'Std':>7} {'Min':>7} {'Max':>7}")
    report.append("-" * 70)
    for i, model in enumerate(models):
        s = score_avg[i]
        valid = s[~np.isnan(s)]
        if len(valid) == 0:
            continue
        report.append(f"{model:<30} {np.mean(valid)*100:>6.1f}% {np.median(valid)*100:>6.1f}% {np.std(valid)*100:>6.1f}% {np.min(valid)*100:>6.1f}% {np.max(valid)*100:>6.1f}%")

    # --- Section 3: Per-tag analysis (category, intent, knowledge) ---
    unique_cats = []
    for tag_name, tag_plural in (("category", "CATEGORIES"), ("intent", "INTENTS"), ("knowledge", "KNOWLEDGE")):
        tag_values = d["tags"][tag_name]
        unique_vals = sorted(v for v in set(tag_values) if v)
        if not unique_vals:
            continue
        if tag_name == "category":
            unique_cats = unique_vals
        report.append(f"\n{'='*100}")
        report.append(f"{harness_label} — {tag_plural} ANALYSIS")
        report.append(f"{'='*100}")
        header = f"\n{tag_name.capitalize():<20}" + "".join(f"{m:>20}" for m in models)
        report.append(header)
        report.append("-" * (20 + 20 * n_models))
        tag_scores = {}
        for val in unique_vals:
            mask = tag_values == val
            row = f"{val:<20}"
            val_model_scores = []
            for i in range(n_models):
                a = np.nansum(awarded_avg[i][mask])
                m = np.nansum(max_avg[i][mask])
                pct = (a / m * 100) if m > 0 else 0
                row += f"{f'{pct:.0f}%':>20}"
                val_model_scores.append(pct)
            report.append(row)
            tag_scores[val] = np.mean(val_model_scores)

        report.append(f"\n{'='*100}")
        report.append(f"{harness_label} — HARDEST {tag_plural} (lowest avg score across all models)")
        report.append(f"{'='*100}")
        for val, avg in sorted(tag_scores.items(), key=lambda x: x[1]):
            report.append(f"  {val:<20} avg={avg:>5.1f}%")

    # --- Section 4: Hardest/easiest questions ---
    report.append(f"\n{'='*100}")
    report.append(f"{harness_label} — HARDEST QUESTIONS (lowest avg score across all models)")
    report.append(f"{'='*100}")
    with np.errstate(invalid="ignore"):
        q_avg = np.nanmean(score_avg, axis=0)
    hardest_order = np.argsort(q_avg)
    for j in hardest_order[:10]:
        if np.isnan(q_avg[j]):
            continue
        qid = questions[j]
        avg = q_avg[j] * 100
        n = int(np.sum(~np.isnan(score_avg[:, j])))
        cat = categories[j]
        q = question_text[j][:70]
        report.append(f"  {qid:<8} avg={avg:>5.1f}%  (n={n})  [{cat}]  {q}...")

    report.append(f"\n{'='*100}")
    report.append(f"{harness_label} — EASIEST QUESTIONS (highest avg score across all models)")
    report.append(f"{'='*100}")
    easiest_order = np.argsort(q_avg)[::-1]
    for j in easiest_order[:10]:
        if np.isnan(q_avg[j]):
            continue
        qid = questions[j]
        avg = q_avg[j] * 100
        n = int(np.sum(~np.isnan(score_avg[:, j])))
        cat = categories[j]
        q = question_text[j][:70]
        report.append(f"  {qid:<8} avg={avg:>5.1f}%  (n={n})  [{cat}]  {q}...")

    # --- Section 5: Factual incorrectness ---
    report.append(f"\n{'='*100}")
    report.append(f"{harness_label} — FACTUAL INCORRECTNESS")
    report.append(f"{'='*100}")
    for i, model in enumerate(models):
        h = int(total_factual_incorrect[i])
        n = int(n_answered[i])
        report.append(f"  {model:<30} {h:>3} / {n}  ({h/n*100:.1f}%)" if n else f"  {model:<30} {h:>3} / 0")

    # --- Section 5b: Per-dimension breakdown ---
    if np.any(dim_criteria > 0):
        report.append(f"\n{'='*100}")
        report.append(f"{harness_label} — PER-DIMENSION BREAKDOWN (% of max)")
        report.append(f"{'='*100}")
        report.append(f"\n{'Model':<30} {'Criteria':>8} {'Task':>8} {'Legal':>8} {'FactCr':>8}")
        report.append("-" * 70)
        for i, model in enumerate(models):
            report.append(f"{model:<30} {dim_criteria[i]:>7.1f}% {dim_task[i]:>7.1f}% {dim_legal[i]:>7.1f}% {dim_factcorr[i]:>7.1f}%")

    report.append("")

    # --- Generate figures ---
    FIGURES.mkdir(parents=True, exist_ok=True)
    DPI = 200
    palette = list(COLORS.values())
    colors = [palette[i % len(palette)] for i in range(n_models)]
    labels = models
    x = np.arange(n_models)

    # Fig 1: Overall scores with Wilson CI
    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)
    pcts = overall_pct
    ci_los, ci_his = [], []
    for i in range(n_models):
        n = int(n_answered[i])
        passes = int(np.nansum(pass_avg[i]))
        lo, hi = wilson_ci(passes, n)
        ci_los.append(max(0.0, pcts[i] - lo * 100))
        ci_his.append(max(0.0, hi * 100 - pcts[i]))
    bars = ax.bar(x, pcts, color=colors, width=0.6, edgecolor="white", linewidth=0.5)
    ax.errorbar(x, pcts, yerr=[ci_los, ci_his], fmt="none", color="black", capsize=4, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Rubric score (%)", fontsize=11)
    ax.set_title(f"{harness_label} — Overall rubric score with 95% Wilson CI", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(pcts) * 1.35 if len(pcts) else 100)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fname = f"{judge}_{harness}_overall_scores.png"
    plt.savefig(FIGURES / fname, dpi=DPI)
    plt.close()
    figures.append(fname)
    print(f"  saved: {FIGURES / fname}")

    # Fig 2: Category heatmap
    if unique_cats:
        matrix = []
        for i in range(n_models):
            row = []
            for cat in unique_cats:
                mask = categories == cat
                a = np.nansum(awarded_avg[i][mask])
                m = np.nansum(max_avg[i][mask])
                row.append((a / m * 100) if m > 0 else 0)
            matrix.append(row)
        fig, ax = plt.subplots(figsize=(max(10, n_models * 1.2), max(5, len(unique_cats) * 0.6)), dpi=DPI)
        im = ax.imshow(np.array(matrix).T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=60)
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(unique_cats)))
        ax.set_yticklabels(unique_cats, fontsize=10)
        for i in range(len(unique_cats)):
            for j in range(n_models):
                ax.text(j, i, f"{matrix[j][i]:.0f}%", ha="center", va="center", fontsize=8, color="black")
        ax.set_title(f"{harness_label} — Score by category (%)", fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, label="%", shrink=0.8)
        plt.tight_layout()
        fname = f"{judge}_{harness}_category_heatmap.png"
        plt.savefig(FIGURES / fname, dpi=DPI)
        plt.close()
        figures.append(fname)
        print(f"  saved: {FIGURES / fname}")

    # Fig 3: Score distribution (box plot)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)
    data = [score_avg[i][~np.isnan(score_avg[i])] * 100 for i in range(n_models)]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Per-question score (%)", fontsize=11)
    ax.set_title(f"{harness_label} — Per-question score distribution", fontsize=13, fontweight="bold")
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = f"{judge}_{harness}_score_distribution.png"
    plt.savefig(FIGURES / fname, dpi=DPI)
    plt.close()
    figures.append(fname)
    print(f"  saved: {FIGURES / fname}")

    # Fig 4: Factual incorrectness
    fig, ax = plt.subplots(figsize=(12, 5), dpi=DPI)
    fails = [int(total_factual_incorrect[i]) for i in range(n_models)]
    bars = ax.bar(x, fails, color=colors, width=0.6, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Factual-correctness failures", fontsize=11)
    ax.set_title(f"{harness_label} — Factual-correctness failures by model", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(fails) * 1.25 if fails else 10)
    for bar, h in zip(bars, fails):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(h),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fname = f"{judge}_{harness}_fact_fails.png"
    plt.savefig(FIGURES / fname, dpi=DPI)
    plt.close()
    figures.append(fname)
    print(f"  saved: {FIGURES / fname}")

    # Fig 5: Flag rates (task completion, legality, factual correctness)
    fig, ax = plt.subplots(figsize=(14, 5), dpi=DPI)
    flags = [("Task completion", dim_task), ("Legality", dim_legal), ("Factual correctness", dim_factcorr)]
    fw = 0.25
    for fi, (fname_, vals) in enumerate(flags):
        ax.bar(x + (fi - 1) * fw, vals, width=fw, label=fname_, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("% of questions", fontsize=11)
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.4)
    ax.set_title(f"{harness_label} — Flag rates by model", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = f"{judge}_{harness}_flags.png"
    plt.savefig(FIGURES / fname, dpi=DPI)
    plt.close()
    figures.append(fname)
    print(f"  saved: {FIGURES / fname}")

    # Fig 6: Cost & efficiency (score vs cost, cost, tokens)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=DPI)

    ax = axes[0]
    for i in range(n_models):
        ax.scatter(total_cost[i], pcts[i], color=colors[i], s=60, zorder=3)
        ax.annotate(labels[i], (total_cost[i], pcts[i]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Total cost (model currency, RUB/USD mixed)", fontsize=10)
    ax.set_ylabel("Rubric score (%)", fontsize=10)
    ax.set_title("Score vs cost", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar(x, total_cost, color=colors, width=0.6, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Total cost (model currency)", fontsize=10)
    ax.set_title("Total cost per model", fontsize=11, fontweight="bold")

    ax = axes[2]
    ax.bar(x, total_tokens / 1000, color=colors, width=0.6, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Tokens (thousands)", fontsize=10)
    ax.set_title("Total tokens spent", fontsize=11, fontweight="bold")

    fig.suptitle(f"{harness_label} — Cost & tokens", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"{judge}_{harness}_efficiency.png"
    plt.savefig(FIGURES / fname, dpi=DPI)
    plt.close()
    figures.append(fname)
    print(f"  saved: {FIGURES / fname}")


def _report_condition_comparison(harness_data: dict, report: list):
    """Compare closedbook, oracle, and local-docs conditions, per judge."""
    for judge in sorted({j for (j, _h) in harness_data}):
        hd = {h: harness_data[(j, h)] for (j, h) in harness_data if j == judge}
        if len(hd) >= 2:
            _condition_comparison_for_judge(judge, hd, report)


def _condition_comparison_for_judge(judge: str, harness_data: dict, report: list):
    harnesses = list(harness_data.keys())

    report.append(f"\n{'='*100}")
    report.append(f"CONDITION COMPARISON — judge: {judge} (closedbook vs oracle vs local-docs)")
    report.append(f"{'='*100}")

    # Get all models across all harnesses
    all_models = sorted(set(
        m for h in harnesses for m in harness_data[h]["models"]
    ))

    report.append(f"\n{'Model':<30}" + "".join(f" {h:>15}" for h in harnesses) + f" {'Best':>15}")
    report.append("-" * (30 + 16 * len(harnesses) + 16))

    for model in all_models:
        row = f"{model:<30}"
        scores = {}
        for h in harnesses:
            d = harness_data[h]
            if model in d["models"]:
                idx = d["models"].index(model)
                pct = d["overall_pct"][idx]
                scores[h] = pct
                row += f" {pct:>14.1f}%"
            else:
                row += f" {'—':>15}"
        if scores:
            best_h = max(scores, key=scores.get)
            row += f" {best_h:>15}"
        report.append(row)

    # Oracle gap analysis (filtered to oracle-eligible questions only)
    if "closedbook" in harness_data and "oracle" in harness_data:
        report.append(f"\n{'='*100}")
        report.append("ORACLE GAP (oracle - closedbook, on same questions, positive = oracle helps)")
        report.append(f"{'='*100}")
        cb = harness_data["closedbook"]
        orc = harness_data["oracle"]
        # Filter closedbook to only questions that exist in oracle
        oracle_qids = set(orc["questions"])
        for model in all_models:
            if model in cb["models"] and model in orc["models"]:
                cb_idx = cb["models"].index(model)
                orc_idx = orc["models"].index(model)
                # Compute closedbook score on oracle-eligible subset only
                cb_scores = []
                orc_scores = []
                for qi, qid in enumerate(cb["questions"]):
                    if qid in oracle_qids:
                        s = cb["score_avg"][cb_idx][qi]
                        if not np.isnan(s):
                            cb_scores.append(s)
                for qi, qid in enumerate(orc["questions"]):
                    s = orc["score_avg"][orc_idx][qi]
                    if not np.isnan(s):
                        orc_scores.append(s)
                if cb_scores and orc_scores:
                    cb_pct = np.mean(cb_scores) * 100
                    orc_pct = np.mean(orc_scores) * 100
                    gap = orc_pct - cb_pct
                    sign = "+" if gap >= 0 else ""
                    report.append(f"  {model:<30} cb={cb_pct:>5.1f}% orc={orc_pct:>5.1f}% gap={sign}{gap:>5.1f}% (n={len(orc_scores)})")

    # Retrieval gap analysis (filtered to local-docs-eligible questions only)
    if "closedbook" in harness_data and "local-docs" in harness_data:
        report.append(f"\n{'='*100}")
        report.append("RETRIEVAL GAP (local-docs - closedbook, on same questions, positive = retrieval helps)")
        report.append(f"{'='*100}")
        cb = harness_data["closedbook"]
        ld = harness_data["local-docs"]
        ld_qids = set(ld["questions"])
        for model in all_models:
            if model in cb["models"] and model in ld["models"]:
                cb_idx = cb["models"].index(model)
                ld_idx = ld["models"].index(model)
                cb_scores = []
                ld_scores = []
                for qi, qid in enumerate(cb["questions"]):
                    if qid in ld_qids:
                        s = cb["score_avg"][cb_idx][qi]
                        if not np.isnan(s):
                            cb_scores.append(s)
                for qi, qid in enumerate(ld["questions"]):
                    s = ld["score_avg"][ld_idx][qi]
                    if not np.isnan(s):
                        ld_scores.append(s)
                if cb_scores and ld_scores:
                    cb_pct = np.mean(cb_scores) * 100
                    ld_pct = np.mean(ld_scores) * 100
                    gap = ld_pct - cb_pct
                    sign = "+" if gap >= 0 else ""
                    report.append(f"  {model:<30} cb={cb_pct:>5.1f}% ld={ld_pct:>5.1f}% gap={sign}{gap:>5.1f}% (n={len(ld_scores)})")

    report.append("")


def main():
    parser = argparse.ArgumentParser(description="Analyze graded results from grades.tsv")
    parser.add_argument("--markdown", action="store_true", help="Also write report.md")
    args = parser.parse_args()

    data = load_data(GRADES_TSV)
    combos = data["combos"]
    harness_data = data["harness_data"]

    report = []
    figures = []

    report.append(f"RuPropManagementEval — Analysis Report")
    report.append(f"Judge × harness: {', '.join(f'{j} · {h}' for j, h in combos)}")

    # Generate per-(judge, harness) sections
    for judge, harness in combos:
        _report_harness(harness_data[(judge, harness)], judge, harness, report, figures)

    # Condition comparison within each judge
    _report_condition_comparison(harness_data, report)

    # --- Output ---
    output = "\n".join(report)
    print(output)

    if args.markdown:
        md_lines = ["# RuPropManagementEval — Analysis Report", ""]
        md_lines.append(f"**Judge × harness:** {', '.join(f'{j} · {h}' for j, h in combos)}")
        md_lines.append("")
        md_lines.append("```")
        md_lines.append(output)
        md_lines.append("```")
        md_lines.append("")
        for fig in figures:
            md_lines.append(f"![{fig}](figures/{fig})")
            md_lines.append("")
        REPORT_MD.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"\nMarkdown report saved to {REPORT_MD}")


if __name__ == "__main__":
    main()
