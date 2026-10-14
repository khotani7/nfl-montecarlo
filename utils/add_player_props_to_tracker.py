#!/usr/bin/env python3
"""
Add player props to ATM tracking spreadsheet.

Since Polymarket player props aren't easily accessible via API, use this script
to add player props to your tracking CSV.

Usage:
    python add_player_props_to_tracker.py --csv week6_atm_tracking.csv \
        --game "SEA @ DEN" \
        --prop "Jaxon Smith-Njigba" "receiving_yards" "over" 89.5 -126 \
        --prop "Emanuel Wilson" "rushing_yards" "under" 70.5 -110 \
        --sims 100000
"""

import argparse
import sys
import os
import csv
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'tests'))
sys.path.insert(0, os.path.join(parent_dir, 'core'))

# Import from test_player_props
from test_player_props import simulate_player_prop
from nfl_montecarlo import load_nflverse


def add_props_to_csv(csv_file, new_props):
    """Add player props to existing tracking CSV."""

    # Read existing lines
    existing_lines = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            existing_lines.append(row)

    # Write back with new props appended
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Write existing lines
        for row in existing_lines:
            writer.writerow(row)

        # Add new player props
        for prop in new_props:
            # Algorithm prediction based on EV thresholds
            if prop['ev'] > 0.05:  # >5% EV
                algo_prediction = "Hit"
            elif prop['ev'] >= -0.05:  # Near 0 EV (-5% to +5%)
                algo_prediction = "Pass"
            else:  # Negative EV < -5%
                algo_prediction = "Miss"

            writer.writerow([
                prop['game'],
                'player',
                prop['description'],
                f"{prop['hit_rate']:.1%}",
                f"{prop['multiplier']:.2f}x",
                f"{prop['odds']:+d}",
                f"{prop['ev']:+.1%}",
                f"{prop['fair_odds']:+d}",
                algo_prediction,  # Algorithm Prediction
                '',               # Actual Result
                '',               # Correct?
                ''                # Notes
            ])


def main():
    ap = argparse.ArgumentParser(description="Add player props to ATM tracking CSV")
    ap.add_argument("--csv", required=True, help="CSV file to update")
    ap.add_argument("--game", required=True, help="Game in format 'AWAY @ HOME'")
    ap.add_argument("--prop", action="append", nargs=5, metavar=("PLAYER", "STAT", "SIDE", "LINE", "ODDS"),
                    help="Player prop to add")
    ap.add_argument("--season", type=int, default=2026, help="NFL season")
    ap.add_argument("--sims", type=int, default=100000, help="Simulations per prop")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--player-last-n-games", type=int, default=4, help="Number of recent games")
    args = ap.parse_args()

    # Parse game
    parts = args.game.split("@")
    if len(parts) != 2:
        print(f"Error: Invalid game format '{args.game}'. Use 'AWAY @ HOME'")
        return

    away = parts[0].strip()
    home = parts[1].strip()

    print(f"Adding player props to: {args.csv}")
    print(f"Game: {args.game}")
    print(f"Props to add: {len(args.prop)}")
    print()

    # Load data
    print("Loading nflverse data...")
    schedules, players, teams = load_nflverse(args.season)
    print()

    # Simulate each prop
    new_props = []
    for i, prop in enumerate(args.prop, 1):
        player_name, stat, side, line_str, odds_str = prop

        try:
            line = float(line_str)
            odds = int(odds_str)
        except ValueError:
            print(f"[{i}/{len(args.prop)}] Error: Invalid line or odds for {player_name}")
            continue

        print(f"[{i}/{len(args.prop)}] Simulating {player_name} {side} {line} {stat}...", end=" ", flush=True)

        prop_details = {
            "player": player_name,
            "stat": stat,
            "side": side,
            "line": line,
            "odds": odds
        }

        result = simulate_player_prop(schedules, teams, players, home, away, prop_details, args)

        if result:
            print(f"✓ Hit Rate: {result['hit_rate']:.1%}, EV: {result['ev']:+.1%}")

            new_props.append({
                'game': args.game,
                'description': f"{player_name} {side} {line} {stat.replace('_', ' ')}",
                'hit_rate': result['hit_rate'],
                'multiplier': result['multiplier'],
                'odds': result['odds'],
                'ev': result['ev'],
                'fair_odds': result['fair_odds']
            })
        else:
            print("✗ Player or stat not found")

    # Add to CSV
    if new_props:
        add_props_to_csv(args.csv, new_props)
        print(f"\n✓ Added {len(new_props)} player props to {args.csv}")
    else:
        print("\nNo props added")


if __name__ == "__main__":
    main()
