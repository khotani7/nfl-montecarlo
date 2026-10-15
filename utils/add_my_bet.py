#!/usr/bin/env python3
"""
Add a bet to your personal betting tracker.

Usage:
    python utils/add_my_bet.py --game "BUF @ KC" --bet "KC -2.5" --stake 100 --odds -110
    python utils/add_my_bet.py --game "DET @ TB" --bet "Over 50.5" --stake 50 --odds 1.95
"""

import argparse
import csv
import sys
import os
from datetime import datetime

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'core'))

from nfl_montecarlo import (
    load_nflverse, team_ratings, model_expectation, fetch_polymarket_event,
    parse_polymarket_event, fit_normal_from_ladder, calibrate_scale, run_simulation,
    prob_to_american
)
import numpy as np
from scipy import stats


def parse_odds(odds_str):
    """Parse odds string to multiplier."""
    try:
        odds = float(odds_str)
        # If between 1.01 and 20, it's already a multiplier
        if 1.01 <= odds <= 20:
            return odds
        # Positive American odds
        elif odds > 0:
            return 1 + (odds / 100)
        # Negative American odds
        else:
            return 1 + (100 / abs(odds))
    except:
        return None


def calculate_model_prediction(game, bet_description, season=2026, sims=100000):
    """Run the model to get hit rate and EV for this bet."""
    try:
        # Parse game
        parts = game.split('@')
        if len(parts) != 2:
            return None, None
        away = parts[0].strip()
        home = parts[1].strip()

        # Load data
        schedules, players, teams = load_nflverse(season)
        ratings = team_ratings(schedules)
        nick = dict(zip(teams["team_abbr"].to_list(), teams["team_nick"].to_list()))

        # Get market data
        home_nick = nick.get(home, home)
        away_nick = nick.get(away, away)
        event = fetch_polymarket_event(home_nick, away_nick)

        # Calculate expectations
        model_home, model_away = model_expectation(ratings, home, away)
        model_margin = model_home - model_away
        model_total = model_home + model_away

        margin_sd, total_sd = 13.5, 13.0
        market_margin, market_total = model_margin, model_total

        if event:
            pm = parse_polymarket_event(event, home_nick, away_nick)
            mu_m, sd_m = fit_normal_from_ladder(pm["spreads"], 13.5)
            if mu_m is not None:
                market_margin = mu_m
                margin_sd = sd_m if sd_m else margin_sd

            mu_t, sd_t = fit_normal_from_ladder(pm["totals"], 13.0)
            if mu_t is not None:
                market_total = mu_t
                total_sd = sd_t if sd_t else total_sd

        # Blend
        w = 0.25
        mu_margin = (1 - w) * market_margin + w * model_margin
        mu_total = (1 - w) * market_total + w * model_total
        home_exp = max((mu_total + mu_margin) / 2, 3.0)
        away_exp = max((mu_total - mu_margin) / 2, 3.0)

        # Simulate
        rng = np.random.default_rng(42)
        eff_scale = calibrate_scale(rng, home_exp, away_exp, margin_sd)
        home_pts, away_pts, _, _, _, _, _ = run_simulation(
            rng, home_exp, away_exp, eff_scale, {}, {}, sims
        )

        total_pts = home_pts + away_pts
        margin = home_pts - away_pts

        # Determine bet type and calculate hit rate
        bet_lower = bet_description.lower()

        if "over" in bet_lower:
            line = float(bet_description.split()[-1])
            hit_rate = float(np.mean(total_pts > line))
        elif "under" in bet_lower:
            line = float(bet_description.split()[-1])
            hit_rate = float(np.mean(total_pts < line))
        else:
            # Spread
            parts = bet_description.split()
            team = parts[0]
            spread = float(parts[1])

            if team == home:
                hit_rate = float(np.mean(margin > -spread))
            else:
                hit_rate = float(np.mean(-margin > -spread))

        return hit_rate, None

    except Exception as e:
        print(f"Warning: Could not calculate model prediction: {e}")
        return None, None


def main():
    ap = argparse.ArgumentParser(description="Add bet to personal tracker")
    ap.add_argument("--game", required=True, help="Game in format 'AWAY @ HOME'")
    ap.add_argument("--bet", required=True, help="Bet description (e.g., 'KC -2.5', 'Over 50.5')")
    ap.add_argument("--stake", type=float, required=True, help="Amount wagered")
    ap.add_argument("--odds", required=True, help="Odds (American or decimal, e.g., -110 or 1.90)")
    ap.add_argument("--date", default=None, help="Date of bet (default: today)")
    ap.add_argument("--notes", default="", help="Optional notes")
    ap.add_argument("--csv", default="tracking/my_bets.csv", help="CSV file to update")
    ap.add_argument("--season", type=int, default=2026, help="NFL season")
    ap.add_argument("--skip-model", action="store_true", help="Skip model calculation (faster)")
    args = ap.parse_args()

    # Parse odds
    multiplier = parse_odds(args.odds)
    if not multiplier:
        print(f"Error: Invalid odds format '{args.odds}'")
        return

    # Get date
    bet_date = args.date if args.date else datetime.now().strftime("%Y-%m-%d")

    # Calculate model prediction
    model_hit_rate = None
    model_ev = None

    if not args.skip_model:
        print(f"Running model simulation for {args.bet}...")
        model_hit_rate, _ = calculate_model_prediction(args.game, args.bet, args.season)
        if model_hit_rate:
            model_ev = model_hit_rate * multiplier - 1
            print(f"Model hit rate: {model_hit_rate:.1%}")
            print(f"Model EV: {model_ev:+.1%}")

    # Determine bet type
    bet_lower = args.bet.lower()
    if "over" in bet_lower:
        bet_type = "Total"
    elif "under" in bet_lower:
        bet_type = "Total"
    elif "ml" in bet_lower or "moneyline" in bet_lower:
        bet_type = "Moneyline"
    else:
        bet_type = "Spread"

    # Convert odds to American for display
    if multiplier >= 2.0:
        american_odds = f"+{int((multiplier - 1) * 100)}"
    else:
        american_odds = f"{int(-100 / (multiplier - 1))}"

    # Add to CSV
    with open(args.csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            bet_date,
            args.game,
            bet_type,
            args.bet,
            f"${args.stake:.2f}",
            american_odds,
            f"{multiplier:.2f}x",
            f"{model_hit_rate:.1%}" if model_hit_rate else "",
            f"{model_ev:+.1%}" if model_ev else "",
            "",  # Actual Result
            "",  # Profit/Loss
            args.notes
        ])

    print(f"\n✓ Added bet to {args.csv}")
    print(f"  Game: {args.game}")
    print(f"  Bet: {args.bet}")
    print(f"  Stake: ${args.stake:.2f}")
    print(f"  Odds: {american_odds} ({multiplier:.2f}x)")
    if model_hit_rate:
        print(f"  Model: {model_hit_rate:.1%} hit rate, {model_ev:+.1%} EV")


if __name__ == "__main__":
    main()
