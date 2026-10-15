#!/usr/bin/env python3
"""
Quick test script for testing specific lines with high accuracy (300k simulations).

Usage:
    python test_line.py "SEA -2.5" --odds 1.90
    python test_line.py "Over 40.5" --odds 2.00
    python test_line.py "DEN +3.5" --odds 2.50
"""

import argparse
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'core'))

# Import from main script
from nfl_montecarlo import (
    load_nflverse, team_ratings, model_expectation, fetch_polymarket_event,
    parse_polymarket_event, fit_normal_from_ladder, calibrate_scale, run_simulation,
    player_distribution, polymarket_price_to_multiplier, multiplier_to_american,
    prob_to_american
)
import numpy as np
from scipy import stats
import polars as pl


def main():
    ap = argparse.ArgumentParser(description="Quick line tester with 300k simulations")
    ap.add_argument("line", help="Line to test (e.g., 'SEA -2.5', 'Over 40.5', 'DEN ML')")
    ap.add_argument("--odds", type=float, help="Multiplier (e.g., 1.90, 2.00) or American odds (+100, -110)")
    ap.add_argument("--home", default="CAR", help="Home team abbreviation")
    ap.add_argument("--away", default="PHI", help="Away team abbreviation")
    ap.add_argument("--season", type=int, default=2026, help="NFL season")
    ap.add_argument("--sims", type=int, default=300000, help="Number of simulations (default: 300000)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = ap.parse_args()

    # Parse multiplier/odds
    multiplier = args.odds if args.odds else None
    if multiplier:
        # If it's between 1.01 and 20, treat as multiplier (decimal odds)
        if 1.01 <= multiplier <= 20:
            pass  # Already a multiplier
        # Otherwise treat as American odds
        elif multiplier > 0:  # Positive American odds (e.g., +110)
            multiplier = 1 + (multiplier / 100)
        else:  # Negative American odds (e.g., -110)
            multiplier = 1 + (100 / abs(multiplier))

    print(f"Testing: {args.line}")
    if multiplier:
        print(f"Multiplier: {multiplier:.2f}x")
    print(f"Simulations: {args.sims:,}")
    print(f"Game: {args.away} @ {args.home}\n")

    # Load data
    print("Loading nflverse data...")
    schedules, players, teams = load_nflverse(args.season)
    nick = dict(zip(teams["team_abbr"].to_list(), teams["team_nick"].to_list()))
    home_nick, away_nick = nick.get(args.home, args.home), nick.get(args.away, args.away)

    # Get market data
    print(f"Fetching Polymarket data...")
    event = fetch_polymarket_event(home_nick, away_nick)

    # Calculate expectations
    ratings = team_ratings(schedules)
    model_home, model_away = model_expectation(ratings, args.home, args.away)
    model_margin, model_total = model_home - model_away, model_home + model_away

    margin_sd, total_sd = 13.5, 13.0

    if event:
        pm = parse_polymarket_event(event, home_nick, away_nick)
        mu_m, sd_m = fit_normal_from_ladder(pm["spreads"], 13.5)
        if mu_m is None and pm["moneyline"] is not None:
            mu_m = 13.5 * stats.norm.ppf(pm["moneyline"])
        if mu_m is not None:
            market_margin = mu_m
            margin_sd = sd_m

        mu_t, sd_t = fit_normal_from_ladder(pm["totals"], 13.0)
        if mu_t is not None:
            market_total = mu_t
    else:
        market_margin, market_total = model_margin, model_total

    # Blend model and market
    w = 0.25
    mu_margin = (1 - w) * market_margin + w * model_margin
    mu_total = (1 - w) * market_total + w * model_total
    home_exp = max((mu_total + mu_margin) / 2, 3.0)
    away_exp = max((mu_total - mu_margin) / 2, 3.0)

    print(f"Expected: {args.home} {home_exp:.1f} - {args.away} {away_exp:.1f}")
    print(f"Running {args.sims:,} simulations...\n")

    # Run simulation
    rng = np.random.default_rng(args.seed)
    eff_scale = calibrate_scale(rng, home_exp, away_exp, margin_sd)
    home_pts, away_pts, _, home_pts_1h, away_pts_1h, home_pts_2h, away_pts_2h = run_simulation(
        rng, home_exp, away_exp, eff_scale, {}, {}, args.sims
    )
    total_pts = home_pts + away_pts
    total_1h = home_pts_1h + away_pts_1h
    total_2h = home_pts_2h + away_pts_2h
    margin = home_pts - away_pts

    # Evaluate line
    line_lower = args.line.lower()
    hit_rate = None

    if "1h" in line_lower or "first half" in line_lower or "1st half" in line_lower:
        # First half total
        parts = args.line.split()
        line_num = float(parts[-1])
        if "over" in line_lower:
            hit_rate = float(np.mean(total_1h > line_num))
            bet_type = f"1H Over {line_num}"
        elif "under" in line_lower:
            hit_rate = float(np.mean(total_1h < line_num))
            bet_type = f"1H Under {line_num}"

    elif "2h" in line_lower or "second half" in line_lower or "2nd half" in line_lower:
        # Second half total
        parts = args.line.split()
        line_num = float(parts[-1])
        if "over" in line_lower:
            hit_rate = float(np.mean(total_2h > line_num))
            bet_type = f"2H Over {line_num}"
        elif "under" in line_lower:
            hit_rate = float(np.mean(total_2h < line_num))
            bet_type = f"2H Under {line_num}"

    elif "over" in line_lower:
        # Extract line number
        parts = args.line.split()
        line_num = float(parts[-1])
        hit_rate = float(np.mean(total_pts > line_num))
        bet_type = f"Over {line_num}"

    elif "under" in line_lower:
        parts = args.line.split()
        line_num = float(parts[-1])
        hit_rate = float(np.mean(total_pts < line_num))
        bet_type = f"Under {line_num}"

    elif "ml" in line_lower:
        if args.home in args.line.upper():
            hit_rate = float(np.mean(margin > 0))
            bet_type = f"{args.home} ML"
        else:
            hit_rate = float(np.mean(margin < 0))
            bet_type = f"{args.away} ML"

    else:
        # Spread
        parts = args.line.split()
        team = parts[0]
        spread = float(parts[1])

        if team == args.home:
            hit_rate = float(np.mean(margin > -spread))
        else:
            hit_rate = float(np.mean(-margin > -spread))

        bet_type = f"{team} {spread:+g}"

    # Results
    print("="*60)
    print(f"RESULTS FOR: {bet_type}")
    print("="*60)
    print(f"Hit Rate (Simulation): {hit_rate:.2%}")
    print(f"Fair Odds: {prob_to_american(hit_rate):+}")

    if multiplier:
        ev = hit_rate * multiplier - 1
        implied_prob = 1.0 / multiplier
        american = multiplier_to_american(multiplier)

        print(f"\nPolymarket Multiplier: {multiplier:.2f}x ({american:+})")
        print(f"Implied Probability: {implied_prob:.2%}")
        print(f"\nEXPECTED VALUE: {ev:+.2%}")

        if ev > 0.10:
            print("✅ STRONG BET - High positive EV!")
        elif ev > 0.05:
            print("✅ GOOD BET - Moderate positive EV")
        elif ev > 0:
            print("⚠️  SLIGHT EDGE - Small positive EV")
        elif ev > -0.05:
            print("⚠️  BREAK EVEN - Near fair odds")
        else:
            print("❌ AVOID - Negative EV")

    print()


if __name__ == "__main__":
    main()
