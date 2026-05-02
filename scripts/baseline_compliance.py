"""Baseline compliance scan over MSD pre-extracted graphs.

Runs every rule from a jurisdiction across a sample of floors and writes a
JSON summary to `output/baseline_compliance.json`. Use this to:

1. Sanity-check rule implementations on real data.
2. Get a real-data baseline of how compliant existing buildings actually are
   under each jurisdiction's thresholds (this is interesting in its own right
   for the paper).

Usage:
    python scripts/baseline_compliance.py [--limit 200] [--split train]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_module import JURISDICTIONS, evaluate_jurisdiction  # noqa: E402
from data_module import iter_msd_pickle_floors  # noqa: E402

logger = logging.getLogger(__name__)


def scan(split: str, limit: int) -> dict:
    summary: dict = {
        "split": split,
        "limit": limit,
        "n_floors_scanned": 0,
        "n_floors_failed_to_load": 0,
        "jurisdictions": {},
    }
    for jcode in JURISDICTIONS:
        summary["jurisdictions"][jcode] = {
            "rule_pass_count": defaultdict(int),
            "rule_total_count": defaultdict(int),
            "n_all_passed": 0,
        }

    n_loaded = 0
    for plan in iter_msd_pickle_floors(split=split, limit=limit):
        n_loaded += 1
        for jcode in JURISDICTIONS:
            try:
                report = evaluate_jurisdiction(plan, jcode)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Plan %s jurisdiction %s failed: %s", plan.plan_id, jcode, exc)
                continue
            for rule_name, result in report["results"].items():
                summary["jurisdictions"][jcode]["rule_total_count"][rule_name] += 1
                if result.passed:
                    summary["jurisdictions"][jcode]["rule_pass_count"][rule_name] += 1
            if report["aggregate"]["all_passed"]:
                summary["jurisdictions"][jcode]["n_all_passed"] += 1

    summary["n_floors_scanned"] = n_loaded

    # Compute pass rates and convert defaultdicts to plain dicts for JSON
    for jcode, jdata in summary["jurisdictions"].items():
        rates = {}
        for rule_name, total in jdata["rule_total_count"].items():
            passed = jdata["rule_pass_count"][rule_name]
            rates[rule_name] = {
                "pass": passed,
                "total": total,
                "rate": passed / total if total else 0.0,
            }
        jdata["rule_pass_rates"] = rates
        jdata["rule_pass_count"] = dict(jdata["rule_pass_count"])
        jdata["rule_total_count"] = dict(jdata["rule_total_count"])
        jdata["all_pass_rate"] = (
            jdata["n_all_passed"] / n_loaded if n_loaded else 0.0
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "output" / "baseline_compliance.json",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = scan(args.split, args.limit)
    with args.out.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"Scanned {summary['n_floors_scanned']} floors from {args.split} split.")
    print()
    for jcode, jdata in summary["jurisdictions"].items():
        spec_name = JURISDICTIONS[jcode].name
        print(f"=== {jcode} — {spec_name} ===")
        for rule_name, rates in jdata["rule_pass_rates"].items():
            pct = 100.0 * rates["rate"]
            print(f"  {rule_name:30s} {rates['pass']:4d}/{rates['total']:4d}  ({pct:5.1f}%)")
        all_pct = 100.0 * jdata["all_pass_rate"]
        print(f"  ALL PASS                      {jdata['n_all_passed']:4d}/{summary['n_floors_scanned']:4d}  ({all_pct:5.1f}%)")
        print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
