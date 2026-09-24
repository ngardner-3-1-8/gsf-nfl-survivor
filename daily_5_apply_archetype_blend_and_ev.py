"""
daily_5_apply_archetype_blend_and_ev.py

The lean tail of the daily pipeline. Runs after daily_4 and does two fast,
in-memory things -- no scraping, no simulation, no odds calls -- so we get the
daily_4-informed pick % and EV without re-running daily_2 (by far the longest
script):

  1. BLEND -- fold daily_4's per-contest archetype/behavioral estimates into
     daily_2's top-down projection for every remaining week and every contest
     (blend_pick_estimates.blend_all_contests, using the fitted stacked-ridge
     weights). This writes the blended pick % back into the sim file, keeping
     the pure projection in the 'Topdown ...' columns.

  2. EV -- recompute true EV on the now-blended pick %, exactly the way daily_3
     does it (daily_3_calculate_ev.loop_through_ev), for Circa and each Splash
     contest.

Because daily_4 runs before this, the blend uses THIS cycle's estimates. Order:
    daily_1 -> daily_2 -> daily_3 -> daily_4 -> daily_5
(daily_2 no longer blends; it just produces the top-down projection.)

Dates come from the central run_config (PIPELINE_AS_OF_DATE when the
orchestrator drives us, else run_config.HISTORICAL_DATES, else today).
"""

from run_config import get_run_dates
from season_dates import resolve_week_context
from blend_pick_estimates import blend_all_contests
from daily_3_calculate_ev import loop_through_ev


def run(date_str):
    ctx = resolve_week_context(date_str)
    print(f"\n=== daily_5 :: {date_str} (target {ctx.target_year}, "
          f"upcoming week {ctx.upcoming_week}) ===")

    # 1. Blend daily_4's behavioral estimates into the pick % (all contests,
    #    all remaining weeks). No-op for a contest/week with no daily_4 file.
    print("daily_5: blending daily_4 archetype estimates into the pick %...")
    blend_all_contests(target_year=ctx.target_year, upcoming_week=ctx.upcoming_week)

    # 2. Recompute true EV on the blended pick % (same engine as daily_3).
    print("daily_5: recomputing true EV on the blended pick %...")
    loop_through_ev(date_str)

    print("daily_5: done.")


if __name__ == "__main__":
    for date in get_run_dates():
        run(date)
