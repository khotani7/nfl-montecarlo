#!/usr/bin/env python3
"""
Test simulation calibration against Polymarket 50/50 lines.

For each game in a given week, finds the Polymarket line closest to 50/50
(where over and under are equally priced) and compares our simulation's
probability against the market.

Usage:
    python test_market_calibration.py --week 6 --season 2026
    python test_market_calibration.py --week 6 --sims 100000  # More accurate
"""

import argparse
import sys
import os
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from main script
from nfl_montecarlo import (
    load_nflverse, team_ratings, model_expectation, fetch_polymarket_event,
    parse_polymarket_event, fit_normal_from_ladder, calibrate_scale, run_simulation
)

def find_50_50_line(markets, market_type="totals"):
    """
    Find the line closest to 50/50 pricing.

    Args:
        markets: List of market dicts from parse_polymarket_event
        market_type: "totals", "spreads", etc.

    Returns:
        (line, over_price, under_price) or None if not found
    """
    best_line = None
    best_diff = float('inf')

    for market in markets:
        if market.get("type") != market_type:
            continue

        line = market.get("line")
        if line is None:
            continue

        prices = market.get("prices", [])
        if len(prices) != 2:
            continue

        # Normalize prices
        total_price = sum(prices)
        if total_price <= 0:
            continue

        norm_prices = [p / total_price for p in prices]

        # Find which is over/under
        outcomes = market.get("outcomes", [])
        if len(outcomes) != 2:
            continue

        over_idx = 0 if "over" in outcomes[0].lower() else 1
        under_idx = 1 - over_idx

        over_price = norm_prices[over_idx]
        under_price = norm_prices[under_idx]

        # How far from 50/50?
        diff = abs(over_price - 0.5)

        if diff < best_diff:
            best_diff = diff
            best_line = (line, over_price, under_price)

    return best_line


def test_game(schedules, teams, home_abbr, away_abbr, args):
    """Test a single game against its 50/50 line."""
    # Get team nicknames for Polymarket
    nick = dict(zip(teams["team_abbr"].to_list(), teams["team_nick"].to_list()))
    home_nick = nick.get(home_abbr, home_abbr)
    away_nick = nick.get(away_abbr, away_abbr)

    # Fetch Polymarket data
    event = fetch_polymarket_event(home_nick, away_nick)

    if not event:
        return {
            "home": home_abbr,
            "away": away_abbr,
            "status": "No Polymarket event found",
            "line": None,
            "market_prob": None,
            "sim_prob": None,
            "diff": None
        }

    pm = parse_polymarket_event(event, home_nick, away_nick)

    # Find 50/50 line
    line_info = find_50_50_line(pm["markets"], "totals")

    if not line_info:
        return {
            "home": home_abbr,
            "away": away_abbr,
            "status": "No total markets found",
            "line": None,
            "market_prob": None,
            "sim_prob": None,
            "diff": None
        }

    line, over_price, under_price = line_info

    # Calculate model expectation
    ratings = team_ratings(schedules)
    model_home, model_away = model_expectation(ratings, home_abbr, away_abbr)
    model_margin, model_total = model_home - model_away, model_home + model_away

    # Blend with market (25% model, 75% market as in CONFIG)
    margin_sd, total_sd = 13.5, 13.0
    mu_m, sd_m = fit_normal_from_ladder(pm["spreads"], 13.5)

    if mu_m is None and pm["moneyline"] is not None:
        from scipy import stats
        mu_m = 13.5 * stats.norm.ppf(pm["moneyline"])

    market_margin = mu_m if mu_m is not None else model_margin

    mu_t, sd_t = fit_normal_from_ladder(pm["totals"], 13.0)
    market_total = mu_t if mu_t is not None else model_total
    if sd_t is not None:
        total_sd = sd_t

    w = 0.25  # model weight
    mu_margin = (1 - w) * market_margin + w * model_margin
    mu_total = (1 - w) * market_total + w * model_total

    home_exp = max((mu_total + mu_margin) / 2, 3.0)
    away_exp = max((mu_total - mu_margin) / 2, 3.0)

    # Run simulation
    rng = np.random.default_rng(args.seed)
    eff_scale = calibrate_scale(rng, home_exp, away_exp, margin_sd)
    home_pts, away_pts, _, _, _, _, _ = run_simulation(
        rng, home_exp, away_exp, eff_scale, {}, {}, args.sims
    )
    total_pts = home_pts + away_pts

    # Calculate our probability
    sim_over_prob = float(np.mean(total_pts > line))

    # Difference from market
    diff = sim_over_prob - over_price

    return {
        "home": home_abbr,
        "away": away_abbr,
        "status": "OK",
        "line": line,
        "market_over_prob": over_price,
        "market_under_prob": under_price,
        "sim_over_prob": sim_over_prob,
        "sim_under_prob": 1 - sim_over_prob,
        "diff": diff,
        "expected_total": home_exp + away_exp
    }


def main():
    ap = argparse.ArgumentParser(description="Test model calibration vs Polymarket 50/50 lines")
    ap.add_argument("--week", type=int, default=6, help="NFL week to test")
    ap.add_argument("--season", type=int, default=2026, help="NFL season")
    ap.add_argument("--sims", type=int, default=50000, help="Simulations per game")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = ap.parse_args()

    print(f"Testing Week {args.week} games against Polymarket 50/50 lines")
    print(f"Using {args.sims:,} simulations per game\n")

    # Load data
    print("Loading nflverse data...")
    schedules, players, teams = load_nflverse(args.season)

    # Filter to specified week
    week_games = schedules.filter(schedules["week"] == args.week)

    if len(week_games) == 0:
        print(f"No games found for week {args.week}")
        return

    print(f"Found {len(week_games)} games in week {args.week}\n")

    # Test each game
    results = []
    for i, row in enumerate(week_games.iter_rows(named=True), 1):
        home = row["home_team"]
        away = row["away_team"]

        print(f"[{i}/{len(week_games)}] Testing {away} @ {home}...", end=" ", flush=True)

        result = test_game(schedules, teams, home, away, args)
        results.append(result)

        if result["status"] == "OK":
            print(f"✓ Line {result['line']}, Diff: {result['diff']:+.1%}")
        else:
            print(f"✗ {result['status']}")

    # Summary
    print("\n" + "="*90)
    print("CALIBRATION RESULTS")
    print("="*90)
    print(f"{'Game':<20} {'Line':<8} {'Market O%':<12} {'Sim O%':<12} {'Diff':<10} {'Exp Total':<10}")
    print("-"*90)

    successful = [r for r in results if r["status"] == "OK"]

    for r in results:
        game = f"{r['away']} @ {r['home']}"
        if r["status"] == "OK":
            line_str = f"O/U {r['line']}"
            market_str = f"{r['market_over_prob']:.1%}"
            sim_str = f"{r['sim_over_prob']:.1%}"
            diff_str = f"{r['diff']:+.1%}"
            exp_str = f"{r['expected_total']:.1f}"
            print(f"{game:<20} {line_str:<8} {market_str:<12} {sim_str:<12} {diff_str:<10} {exp_str:<10}")
        else:
            print(f"{game:<20} {r['status']}")

    if successful:
        print("-"*90)

        # Calculate statistics
        diffs = [r["diff"] for r in successful]
        avg_diff = np.mean(diffs)
        abs_avg_diff = np.mean([abs(d) for d in diffs])
        std_diff = np.std(diffs)

        print(f"\nCalibration Statistics ({len(successful)} games):")
        print(f"  Average difference:     {avg_diff:+.2%}  ({'model favors overs' if avg_diff > 0 else 'model favors unders'})")
        print(f"  Absolute avg diff:      {abs_avg_diff:.2%}")
        print(f"  Standard deviation:     {std_diff:.2%}")
        print(f"  Max positive diff:      {max(diffs):+.2%}")
        print(f"  Max negative diff:      {min(diffs):+.2%}")

        # Quality assessment
        if abs_avg_diff < 0.03:
            print(f"\n✅ Excellent calibration! Model closely matches market consensus.")
        elif abs_avg_diff < 0.05:
            print(f"\n✓ Good calibration. Model is reasonably aligned with market.")
        elif abs_avg_diff < 0.10:
            print(f"\n⚠️  Moderate calibration. Some divergence from market.")
        else:
            print(f"\n❌ Poor calibration. Significant divergence from market.")

    print()


if __name__ == "__main__":
    main()
