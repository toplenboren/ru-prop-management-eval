"""Curated top-level charts from results/grades.tsv."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
})

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
DPI = 100  # ~1050px wide — README-friendly
USD_RUB = 85  # user-provided rate for cost normalization

FAMILY_COLORS = {"Open weights": "#2ca02c", "Russian (closed)": "#d62728", "US (closed)": "#1f77b4"}


def _currencies():
    sys.path.insert(0, "src")
    from config import parse_config
    return {c["name"]: c.get("currency", "RUB") for c in parse_config("llms_config.jsonl")}


def family(model: str) -> str:
    if model.startswith("openrouter-"):
        return "US (closed)"
    if model.startswith(("GigaChat", "yandexgpt")):
        return "Russian (closed)"
    return "Open weights"


def load(judge="jev", harness="closedbook"):
    df = pd.read_csv(RESULTS / "grades.tsv", sep="\t")
    df = df[(df["judge"] == judge) & (df["harness"] == harness)]
    per = df.groupby("model").agg(
        awarded=("total_awarded_avg", "sum"), mx=("total_max_avg", "sum"),
        legality=("legality_avg", "mean"), factual=("factual_correctness_avg", "mean"),
        task=("task_completion_avg", "mean"),
        cost=("cost_avg", "sum"), tokens=("total_tokens_avg", "sum"),
    )
    per["illegal_n"] = df.groupby("model")["legality_avg"].apply(lambda s: (1 - s).sum())
    per["facterr_n"] = df.groupby("model")["factual_correctness_avg"].apply(lambda s: (1 - s).sum())
    per["score"] = per["awarded"] / per["mx"] * 100
    for c in ("legality", "factual", "task"):
        per[c] *= 100
    per["family"] = [family(m) for m in per.index]
    cur = _currencies()
    per["cost_rub"] = [row["cost"] * (USD_RUB if cur.get(m) == "USD" else 1)
                       for m, row in per.iterrows()]
    n_q = df.groupby("model").size()
    per["cost_per_q"] = per["cost_rub"] / n_q
    per["price_1m"] = per["cost_rub"] / per["tokens"] * 1e6
    return per.sort_values("score")


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {FIGURES / name}")


def chart_scores(per):
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)
    colors = [FAMILY_COLORS[f] for f in per["family"]]
    ax.barh(per.index, per["score"], color=colors, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(per["score"]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    ax.legend(handles, FAMILY_COLORS.keys(), fontsize=10, loc="lower right")
    ax.set_xlabel("Rubric score (%)")
    ax.set_title("Answer quality by model", fontweight="bold")
    ax.set_xlim(0, per["score"].max() * 1.15)
    save(fig, "top_scores.png")


def chart_family(per):
    fig, ax = plt.subplots(figsize=(9, 4), dpi=DPI)
    fams = list(FAMILY_COLORS)
    for yi, f in enumerate(fams):
        sub = per[per["family"] == f]["score"]
        ax.scatter(sub, [yi] * len(sub), color=FAMILY_COLORS[f], s=60, alpha=0.55, zorder=2)
        ax.hlines(yi, sub.min(), sub.max(), color=FAMILY_COLORS[f], alpha=0.35, linewidth=8, zorder=1)
        ax.scatter(sub.mean(), yi, color=FAMILY_COLORS[f], s=200, marker="D",
                   edgecolor="black", zorder=3, label=f"{f} (mean {sub.mean():.1f}%)")
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels(fams)
    ax.set_xlabel("Rubric score (%)")
    ax.set_title("Model family comparison", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "top_family.png")


def _short(model):
    return model.replace("openrouter-", "").replace("-noreason", "·nr")


def chart_cost_vs(per, y, ylabel, title, name):
    fig, ax = plt.subplots(figsize=(10.5, 7.6), dpi=DPI)
    for f, c in FAMILY_COLORS.items():
        sub = per[per["family"] == f]
        ax.scatter(sub["price_1m"], sub[y], color=c, s=70, label=f, zorder=3)
    ax.margins(x=0.08, y=0.08)
    texts = [ax.text(row["price_1m"], row[y], f"{_short(m)} ({row['cost_per_q']:.2f}₽/q)",
                     fontsize=9) for m, row in per.iterrows()]
    adjust_text(texts, ax=ax, expand=(1.3, 1.8), force_text=(0.4, 0.8),
                seed=0, arrowprops=dict(arrowstyle="-", color="gray", lw=0.6))
    a, b = np.polyfit(per["price_1m"], per[y], 1)
    xs = np.linspace(0, per["price_1m"].max(), 50)
    ax.plot(xs, a * xs + b, color="gray", linestyle="--", linewidth=1.5,
            label=f"trend ({a:+.3f}·₽/1M + {b:.0f})")
    ax.set_xlabel("Effective price, ₽ per 1M tokens (USD converted at 85)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    fig.legend(loc="lower center", ncol=4, fontsize=10, frameon=False,
               bbox_to_anchor=(0.55, 0.005))
    ax.grid(alpha=0.3)
    save(fig, name)


def chart_legality(per):
    srt = per.sort_values("legality", ascending=False)  # most dangerous on top
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)
    norm = plt.Normalize(50, 100)
    colors = plt.cm.RdYlGn(norm(srt["legality"]))
    ax.barh(srt.index, srt["legality"], color=colors, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(srt["legality"]):
        ax.text(v + 0.7, i, f"{v:.0f}%", va="center", fontsize=9, fontweight="bold")
    ax.axvline(100, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Legality — % of answers judged legal & safe to follow")
    ax.set_title("Legality ranking (most dangerous on top)", fontweight="bold")
    ax.set_xlim(0, 112)
    save(fig, "top_legality.png")


def main():
    per = load()
    chart_scores(per)
    chart_family(per)
    chart_cost_vs(per, "score", "Rubric score (%)",
                  "Answer quality vs cost", "top_score_vs_cost.png")
    chart_cost_vs(per, "illegal_n", "# of answers judged illegal",
                  "Cost vs illegal advices", "top_cost_vs_illegal.png")
    chart_cost_vs(per, "facterr_n", "# of factually incorrect answers",
                  "Cost vs hallucinations", "top_cost_vs_halluc.png")
    chart_legality(per)


if __name__ == "__main__":
    main()
