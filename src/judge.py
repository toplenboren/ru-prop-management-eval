#!/usr/bin/env python3
"""Judge model answers against rubrics.

Reads:  results/<model>_answers.jsonl
Writes: results/<model>_<judge>_grades.jsonl

CLI:  python src/judge.py [judge_name ...] [--runs N]
       No judge names → run all judges from judges.jsonl
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from config import parse_config, load_dataset
from providers import get_llm_provider

DATASET = ROOT / "data" / "dataset.tsv"
RESULTS = ROOT / "results"
PROMPTS = ROOT / "prompts"
CONFIG = ROOT / "judges.jsonl"

DEFAULT_JUDGE_PROMPT = (PROMPTS / "judge.txt").read_text(encoding="utf-8").strip()
JUDGE_USER_TEMPLATE = (PROMPTS / "judge_user.txt").read_text(encoding="utf-8").strip()


def get_judge_prompt(config: dict) -> str:
    override = config.get("judge_prompt_override")
    if override:
        return Path(override).read_text(encoding="utf-8").strip()
    return DEFAULT_JUDGE_PROMPT


CRITERIA_MAX = 5  # per-question criteria are normalized to this

# Uniform noul flags — same for every question. Keep in sync with the flag
# definitions in prompts/judge.txt (LLM judges read them there).
FLAG_INSTRUCTIONS = {
    "task_completion": "Модель ответила на заданный вопрос по существу; ответ решает вопрос и присутствует.",
    "legality": "Действия и советы в ответе модели соответствуют законодательству РФ и безопасны для следования.",
    "factual_correctness": "Ответ модели опирается на реальные факты и существующие источники права (не выдумывает законы, статьи, пункты).",
}


def _finalize_grade(answer: dict, judge_name: str, run_idx: int, grade: dict,
                    criteria_max: float, elapsed_s: float) -> dict:
    """Shared scoring math: normalize question-specific criteria to criteria_max."""
    scores = grade.get("scores", [])

    specific_awarded = sum(s.get("awarded", 0) for s in scores)
    specific_max = sum(s.get("max", 0) for s in scores)
    criteria_awarded = (specific_awarded / specific_max) * criteria_max if specific_max else 0

    return {
        "id": answer["id"],
        "model": answer["model"],
        "judge": judge_name,
        "run": run_idx,
        "scores": scores,
        "criteria_awarded": round(criteria_awarded, 2),
        "criteria_max": criteria_max,
        "total_awarded": round(criteria_awarded, 2),
        "total_max": criteria_max,
        "task_completion": grade.get("task_completion", True),
        "legality": grade.get("legality", True),
        "factual_correctness": grade.get("factual_correctness", True),
        "flag_reasons": grade.get("flag_reasons", {}),
        "overall_comment": grade.get("overall_comment", ""),
        "elapsed_s": elapsed_s,
        "error": None,
    }


def _error_grade(answer: dict, judge_name: str, run_idx: int, criteria_max: float,
                 e: Exception) -> dict:
    return {
        "id": answer["id"],
        "model": answer["model"],
        "judge": judge_name,
        "run": run_idx,
        "scores": [],
        "criteria_awarded": 0,
        "criteria_max": criteria_max,
        "total_awarded": 0,
        "total_max": criteria_max,
        "task_completion": True,
        "legality": True,
        "factual_correctness": True,
        "flag_reasons": {},
        "overall_comment": "",
        "elapsed_s": 0,
        "error": str(e),
    }


# --- Laya (non-generative decision model) ---
# Rubric criteria become typed questions; a 3-level score scale mirrors the
# judge prompt's 0 / 0.5 / 1.0 fraction-of-max convention.
_LAYA_LEVELS = ["не выполнен", "частично выполнен", "полностью выполнен"]


def _typed_questions(question_specific: list) -> dict:
    """Rubric → Laya/Jev typed questions (same schema for both)."""
    questions = {}
    for i, c in enumerate(question_specific):
        questions[f"spec_{i}"] = {
            "type": "score",
            "instructions": f"Оцени ответ модели по критерию: {c['criterion']}",
            "criteria": _LAYA_LEVELS,
        }
    for flag, instr in FLAG_INSTRUCTIONS.items():
        questions[flag] = {
            "type": "noul",
            "instructions": instr,
            "labels": {"false": "нет", "true": "да"},
        }
    return questions


def _typed_grade(ans: dict, question_specific: list) -> dict:
    """Laya/Jev answers → grade dict in the LLM-judge JSON shape."""
    n_levels = len(_LAYA_LEVELS) - 1
    return {
        "scores": [
            {"criterion": c["criterion"],
             "awarded": round(ans[f"spec_{i}"]["score"] / n_levels * c["points"], 2),
             "max": c["points"], "reason": ""}
            for i, c in enumerate(question_specific)
        ],
        **{flag: bool(ans[flag]["noul"] >= 0.5) for flag in FLAG_INSTRUCTIONS},
        "overall_comment": "",
    }


def _typed_state(answer: dict, question: dict) -> dict:
    # Laya right-truncates state at max_len: judged answer first, reference last.
    return {
        "Ответ модели": answer["answer"],
        "Вопрос": answer["question"],
        "Эталонный ответ эксперта": question["expert_answer"],
    }


def _load_rubric(question: dict) -> tuple[list, float]:
    """Dataset column dynamic_rubric = per-question criteria list.
    Uniform flags (task_completion/legality/factual_correctness) live in FLAG_INSTRUCTIONS."""
    specific = json.loads(question.get("dynamic_rubric") or "[]")
    return specific, CRITERIA_MAX


def judge_one_typed(provider, judge_config: dict, answer: dict, question: dict, run_idx: int) -> dict:
    question_specific, criteria_max = _load_rubric(question)
    try:
        answers, elapsed = provider.predict(
            _typed_state(answer, question), _typed_questions(question_specific)
        )
        grade = _typed_grade(answers, question_specific)
        return _finalize_grade(answer, judge_config["name"], run_idx, grade, criteria_max, elapsed)
    except Exception as e:
        return _error_grade(answer, judge_config["name"], run_idx, criteria_max, e)


def judge_one(provider, judge_config: dict, judge_prompt: str, answer: dict, question: dict, run_idx: int) -> dict:
    question_specific, criteria_max = _load_rubric(question)

    specific_text = "\n".join(
        f"- {i+1}. {c['criterion']} ({c['points']} балл(ов))"
        for i, c in enumerate(question_specific)
    )

    user_msg = JUDGE_USER_TEMPLATE.format(
        question=answer["question"],
        expert_answer=question["expert_answer"],
        answer=answer["answer"],
        criteria=specific_text,
    )
    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": user_msg},
    ]
    try:
        resp = provider.call_llm(messages=messages, **judge_config.get("params", {}))
        content = resp.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content
            content = content.rsplit("```", 1)[0]
        grade = json.loads(content)
        return _finalize_grade(answer, judge_config["name"], run_idx, grade, criteria_max, resp.elapsed_s)
    except Exception as e:
        return _error_grade(answer, judge_config["name"], run_idx, criteria_max, e)


def main():
    parser = argparse.ArgumentParser(description="Judge model answers against rubrics")
    parser.add_argument("judges", nargs="*", help="Judge names from judges.jsonl (default: all)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per question (default: 3)")
    args = parser.parse_args()

    judge_configs = parse_config(CONFIG, args.judges)

    dataset = load_dataset(DATASET)

    answer_files = sorted(RESULTS.glob("*_answers.jsonl"))
    if not answer_files:
        print(f"No *_answers.jsonl in {RESULTS}", file=sys.stderr)
        sys.exit(1)

    for judge_config in judge_configs:
        judge_name = judge_config["name"]
        judge_prompt = get_judge_prompt(judge_config)
        api_kind = judge_config["api"]
        provider = get_llm_provider(api_kind, **judge_config["provider"])
        is_typed = api_kind in ("laya", "jev")
        workers = int(judge_config.get("concurrency") or os.environ.get("CONCURRENCY") or 5)
        print(f"\n=== Judge: {judge_name} ===")

        for answers_path in answer_files:
            model_slug = answers_path.stem.replace("_answers", "")
            grades_path = RESULTS / f"{model_slug}_{judge_name}_grades.jsonl"
            print(f"\n[judging {model_slug}] → {grades_path}")

            answers = [json.loads(l) for l in answers_path.read_text(encoding="utf-8").split("\n") if l.strip()]

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
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    if is_typed:
                        futures = {
                            pool.submit(judge_one_typed, provider, judge_config, a, dataset[a["id"]], run_idx): (a, run_idx)
                            for a, run_idx in pending
                        }
                    else:
                        futures = {
                            pool.submit(judge_one, provider, judge_config, judge_prompt, a, dataset[a["id"]], run_idx): (a, run_idx)
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
