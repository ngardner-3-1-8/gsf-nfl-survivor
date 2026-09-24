"""
run_pipeline.py

Historical (and live) pipeline orchestrator.

For each date in run_config.HISTORICAL_DATES (or the real today when that list
is empty), run EVERY stage in dependency order before moving to the next date
-- so each historical week is rebuilt start-to-finish, rather than running one
script across all weeks at a time.

Each stage runs as its own subprocess with PIPELINE_AS_OF_DATE set to the
current date, so the whole system treats that date as "today" (see
run_config.py). The per-contest stages run once per SURVIVOR_CONTEST, exactly
like the GitHub workflows.

Usage:
    python run_pipeline.py                      # dates from run_config
    python run_pipeline.py --dates 09/16/2026 09/23/2026   # override dates
    python run_pipeline.py --only daily_2 daily_3          # subset of stages
    python run_pipeline.py --skip weekly_5 weekly_6        # all but these
    python run_pipeline.py --list                          # show stages, exit

Circa is authoritative: if a non-contest stage or a contest stage's Circa run
fails, that date's remaining stages are skipped (unless --keep-going). The
Splash contests are best-effort -- a missing model/picks file for Big Splash or
the World Championship is logged and skipped without aborting the date.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

import run_config

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CONTESTS = ["circa", "big_splash", "world_championship"]

# (short_name, script_filename, per_contest) in dependency order.
STAGES = [
    ("weekly_1", "weekly_1_collect_last_weeks_pick_data.py", False),
    ("weekly_2", "weekly_2_collect_transactions.py", False),
    ("weekly_3b", "weekly_3b_normalize_splash_pick_history.py", False),
    ("build_contest_historical", "build_contest_historical_data.py", False),
    ("weekly_4", "weekly_4_identify_entry_archetypes.py", True),
    ("weekly_5", "weekly_5_build_entry_pick_training_data.py", True),
    ("weekly_6", "weekly_6_entry_pick_choice_model_training.py", True),
    ("daily_1", "daily_1_get_internal_rankings.py", False),
    ("daily_2", "daily_2_consolidate_predict_weekly_data.py", False),
    ("daily_3", "daily_3_calculate_ev.py", False),
    ("daily_4", "daily_4_generate_entry_archetype_pick_estimates.py", True),
]


def _run(script, contest, date, keep_going):
    """Run one stage (one contest) as a subprocess. Returns True on success."""
    env = dict(os.environ)
    env["PIPELINE_AS_OF_DATE"] = date
    if contest:
        env["SURVIVOR_CONTEST"] = contest
    else:
        env.pop("SURVIVOR_CONTEST", None)
    label = script + (f"  [{contest}]" if contest else "")
    print(f"\n▶️  {date}  ::  {label}")
    result = subprocess.run([sys.executable, script], env=env, cwd=REPO_DIR)
    if result.returncode == 0:
        return True
    print(f"❌ {label} exited with code {result.returncode}")
    # Splash contests are best-effort (may have no data/model yet this week).
    if contest in ("big_splash", "world_championship"):
        print(f"   (Splash contest — logging and continuing)")
        return True
    if keep_going:
        print(f"   (--keep-going — continuing despite failure)")
        return True
    return False


def run_stage(name, script, per_contest, date, keep_going):
    """Run a stage across its contests. Returns True if the date may continue."""
    for contest in (CONTESTS if per_contest else [None]):
        if not _run(script, contest, date, keep_going):
            return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the pipeline per as-of date.")
    ap.add_argument("--dates", nargs="+", metavar="MM/DD/YYYY",
                    help="Override run_config.HISTORICAL_DATES for this run.")
    ap.add_argument("--only", nargs="+", metavar="STAGE",
                    help="Run only these stages (by short name).")
    ap.add_argument("--skip", nargs="+", metavar="STAGE",
                    help="Run all stages except these (by short name).")
    ap.add_argument("--keep-going", action="store_true",
                    help="Don't abort a date when an authoritative stage fails.")
    ap.add_argument("--list", action="store_true",
                    help="List the stage order and exit.")
    args = ap.parse_args(argv)

    if args.list:
        print("Stages (in order):")
        for name, script, pc in STAGES:
            print(f"  {name:26s} {script}" + ("  [per contest]" if pc else ""))
        return 0

    stages = STAGES
    if args.only:
        want = set(args.only)
        stages = [s for s in STAGES if s[0] in want]
    if args.skip:
        drop = set(args.skip)
        stages = [s for s in stages if s[0] not in drop]
    if not stages:
        print("No stages selected.")
        return 1

    if args.dates:
        dates = args.dates
    elif run_config.HISTORICAL_DATES:
        dates = [str(d).strip() for d in run_config.HISTORICAL_DATES]
    else:
        dates = [datetime.now().strftime("%m/%d/%Y")]

    print(f"Pipeline: {len(stages)} stage(s) × {len(dates)} date(s): {dates}")
    for date in dates:
        print("\n" + "=" * 72)
        print(f"  AS-OF DATE: {date}")
        print("=" * 72)
        for name, script, per_contest in stages:
            ok = run_stage(name, script, per_contest, date, args.keep_going)
            if not ok:
                print(f"\n🛑 Aborting remaining stages for {date} "
                      f"(stage '{name}' failed). Use --keep-going to override.")
                break
    print("\n✅ Pipeline run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
