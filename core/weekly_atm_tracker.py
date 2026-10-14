#!/usr/bin/env python3
"""
Weekly ATM Line Tracker - Scans all ATM lines for Week 6 and creates a tracking spreadsheet.

Outputs a CSV file you can use to track your hit rate predictions vs actual results.

Usage:
    python weekly_atm_tracker.py --week 6 --season 2026 --sims 100000
    python weekly_atm_tracker.py --week 6 --output week6_lines.csv
"""

import argparse
import sys
import os
import csv
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from main scripts
from nfl_montecarlo import (
    load_nflverse, team_ratings, model_expectation, fetch_polymarket_event,
    parse_polymarket_event, fit_normal_from_ladder, calibrate_scale, run_simulation,
    prob_to_american, payout_multiplier, multiplier_to_american, player_distribution
)
from daily_scanner import scan_game, find_atm_lines
from scipy import stats


def scan_all_atm_lines(schedules, teams, players, week, args):
    """Scan for the MOST ATM line (closest to 50/50) per category per game."""

    # Filter to specified week
    week_games = schedules.filter(schedules["week"] == week)

    if len(week_games) == 0:
        print(f"No games found for week {week}")
        return []

    all_lines = []

    for i, row in enumerate(week_games.iter_rows(named=True), 1):
        home = row["home_team"]
        away = row["away_team"]
        game_key = f"{away} @ {home}"

        print(f"[{i}/{len(week_games)}] Scanning {game_key}...", flush=True)

        # First try with normal spread range
        game_opps = scan_game(schedules, teams, home, away, args)

        # Group by type
        spreads = [o for o in game_opps if o['bet_type'] == 'spread']
        totals = [o for o in game_opps if o['bet_type'] == 'total']

        # If no spreads found within default range, try wider range
        if not spreads:
            # Create a copy of args with wider spread range
            import copy
            args_wide = copy.copy(args)
            args_wide.spread_range = 20.0  # Try much wider range
            game_opps_wide = scan_game(schedules, teams, home, away, args_wide)
            spreads = [o for o in game_opps_wide if o['bet_type'] == 'spread']

        # Find the MOST ATM spread (closest to 50%)
        if spreads:
            most_atm_spread = min(spreads, key=lambda x: abs(x['hit_rate'] - 0.5))
            all_lines.append({
                'game': game_key,
                'type': 'spread',
                'description': most_atm_spread['description'],
                'hit_rate': most_atm_spread['hit_rate'],
                'multiplier': most_atm_spread['multiplier'],
                'american_odds': most_atm_spread['american_odds'],
                'ev': most_atm_spread['ev'],
                'fair_odds': most_atm_spread['fair_odds'],
            })

        # Find the MOST ATM total (closest to 50%)
        if totals:
            most_atm_total = min(totals, key=lambda x: abs(x['hit_rate'] - 0.5))
            all_lines.append({
                'game': game_key,
                'type': 'total',
                'description': most_atm_total['description'],
                'hit_rate': most_atm_total['hit_rate'],
                'multiplier': most_atm_total['multiplier'],
                'american_odds': most_atm_total['american_odds'],
                'ev': most_atm_total['ev'],
                'fair_odds': most_atm_total['fair_odds'],
            })

        print(f"   Added {1 if spreads else 0} spread + {1 if totals else 0} total")

    return all_lines


def write_tracking_csv(lines, output_file):
    """Write lines to a CSV file for tracking."""

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Game',
            'Type',
            'Bet Description',
            'Hit Rate',
            'Multiplier',
            'American Odds',
            'EV',
            'Fair Odds',
            'Algorithm Prediction',  # Auto-filled based on EV (Hit >5%, Pass ~0%, Miss <-5%)
            'Actual Result',         # User fills: Hit or Miss
            'Correct?',              # User fills: Yes or No
            'Notes'
        ])

        # Sort by game, then by EV
        lines.sort(key=lambda x: (x['game'], -x['ev']))

        # Write each line
        for line in lines:
            # Algorithm prediction based on EV thresholds
            if line['ev'] > 0.05:  # >5% EV
                algo_prediction = "Hit"
            elif line['ev'] >= -0.05:  # Near 0 EV (-5% to +5%)
                algo_prediction = "Pass"
            else:  # Negative EV < -5%
                algo_prediction = "Miss"

            writer.writerow([
                line['game'],
                line['type'],
                line['description'],
                f"{line['hit_rate']:.1%}",
                f"{line['multiplier']:.2f}x",
                f"{line['american_odds']:+d}",
                f"{line['ev']:+.1%}",
                f"{line['fair_odds']:+d}",
                algo_prediction,  # Algorithm's prediction
                '',               # Actual Result (blank for user)
                '',               # Correct? (blank for user)
                ''                # Notes (blank for user)
            ])


def main():
    ap = argparse.ArgumentParser(description="Weekly ATM Line Tracker")
    ap.add_argument("--week", type=int, default=6, help="NFL week to scan")
    ap.add_argument("--season", type=int, default=2026, help="NFL season")
    ap.add_argument("--sims", type=int, default=100000, help="Simulations per line")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--spread-range", type=float, default=6.5, help="Only spreads within ±X")
    ap.add_argument("--output", type=str, default=None, help="Output CSV filename")
    args = ap.parse_args()

    # Default output filename
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"week{args.week}_atm_lines_{timestamp}.csv"

    print("="*80)
    print(f"WEEKLY ATM LINE TRACKER - Week {args.week}, {args.season}")
    print("="*80)
    print(f"Simulations: {args.sims:,} per line")
    print(f"Output file: {args.output}")
    print()

    # Load data
    print("Loading nflverse data...")
    schedules, players, teams = load_nflverse(args.season)
    print()

    # Scan all ATM lines
    all_lines = scan_all_atm_lines(schedules, teams, players, args.week, args)

    print(f"\n{'='*80}")
    print(f"SCAN COMPLETE: Found {len(all_lines)} total ATM lines")
    print(f"{'='*80}\n")

    # Write to CSV
    write_tracking_csv(all_lines, args.output)

    print(f"✓ Tracking spreadsheet saved to: {args.output}")
    print()
    print("How to use:")
    print("1. Open the CSV file in Excel/Google Sheets")
    print("2. For each line, fill in your prediction (Hit or Miss)")
    print("3. After games complete, fill in actual results (Hit or Miss)")
    print("4. Mark if you were correct (Yes or No)")
    print("5. Track your accuracy over time!")
    print()

    # Show summary stats
    positive_ev = [l for l in all_lines if l['ev'] > 0]
    print(f"Summary:")
    print(f"  Total ATM lines: {len(all_lines)}")
    print(f"  Positive EV lines: {len(positive_ev)}")
    print(f"  Average EV (all lines): {np.mean([l['ev'] for l in all_lines]):.1%}")
    if positive_ev:
        print(f"  Average EV (positive only): {np.mean([l['ev'] for l in positive_ev]):.1%}")
    print()


if __name__ == "__main__":
    main()
