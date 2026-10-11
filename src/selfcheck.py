"""Smoke test: config files parse, prompts exist, pipeline modules import."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent.parent
FAIL = []


def check(name, fn):
    try:
        fn()
        print(f"  ok   {name}")
    except Exception as e:
        FAIL.append(name)
        print(f"  FAIL {name}: {e}")


def parse_all():
    from config import parse_config, load_dataset
    models = parse_config(ROOT / "llms_config.jsonl")
    judges = parse_config(ROOT / "judges.jsonl")
    assert models and judges, "empty config"
    data = load_dataset(ROOT / "data" / "dataset.tsv")
    assert len(data) > 0, "empty dataset"
    print(f"    {len(models)} models, {len(judges)} judges, {len(data)} questions")


def prompts_exist():
    for p in ("system.txt", "judge.txt", "judge_user.txt"):
        f = ROOT / "prompts" / p
        assert f.exists() and f.read_text().strip(), f"missing or empty: {f}"


def imports():
    import run, judge, combine_results, analyze_results  # noqa: F401


if __name__ == "__main__":
    check("configs + dataset parse", parse_all)
    check("prompts present", prompts_exist)
    check("pipeline modules import", imports)
    sys.exit(1 if FAIL else print("all checks passed"))
