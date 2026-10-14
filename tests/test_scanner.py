#!/usr/bin/env python3
"""
Quick test scanner for specific games.

Usage:
    python test_scanner.py --games "SEA @ DEN" "CAR @ PHI"
    python test_scanner.py --games "HOU @ JAX" --sims 50000
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from daily scanner
from daily_scanner import (
    scan_game, suggest_sgp, suggest_mgp, load_nflverse
)


def main():
    ap = argparse.ArgumentParser(description="Test scanner with specific games")
    ap.add_argument("--games", nargs="+", required=True, help="Games to scan (e.g., 'SEA @ DEN' 'CAR @ PHI')")
    ap.add_argument("--season", type=int, default=2026, help="NFL season")
    ap.add_argument("--sims", type=int, default=100000, help="Simulations per opportunity")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--min-ev", type=float, default=0.05, help="Minimum EV to display")
    ap.add_argument("--spread-range", type=float, default=6.5, help="Only spreads within ±X")
    args = ap.parse_args()

    print("="*80)
    print(f"QUICK SCANNER TEST - {len(args.games)} Games")
    print("="*80)
    print(f"Simulations: {args.sims:,} per opportunity")
    print(f"Minimum EV: {args.min_ev:+.1%}")
    print()

    # Load data
    print("Loading nflverse data...")
    schedules, players, teams = load_nflverse(args.season)

    # Parse games
    all_opportunities = []
    opportunities_by_game = {}

    for i, game_str in enumerate(args.games, 1):
        parts = game_str.split("@")
        if len(parts) != 2:
            print(f"Error: Invalid game format '{game_str}'. Use 'AWAY @ HOME'")
            continue

        away = parts[0].strip()
        home = parts[1].strip()
        game_key = f"{away} @ {home}"

        print(f"[{i}/{len(args.games)}] Scanning {game_key}...", end=" ", flush=True)

        game_opps = scan_game(schedules, teams, home, away, args)

        if game_opps:
            print(f"✓ Found {len(game_opps)} ATM opportunities")
            all_opportunities.extend(game_opps)
            opportunities_by_game[game_key] = game_opps
        else:
            print("✗ No ATM markets found")

    # Filter by minimum EV
    positive_opps = [o for o in all_opportunities if o["ev"] >= args.min_ev]

    print(f"\n{'='*80}")
    print(f"SCAN COMPLETE: Found {len(positive_opps)} opportunities with EV >= {args.min_ev:+.1%}")
    print(f"{'='*80}\n")

    # Display individual bets
    if positive_opps:
        positive_opps.sort(key=lambda x: x["ev"], reverse=True)

        print("="*80)
        print("TOP INDIVIDUAL BETS (Ranked by EV)")
        print("="*80)
        print(f"{'#':<4} {'Game':<22} {'Bet':<25} {'Hit%':<8} {'Mult':<12} {'EV':<10} {'Fair'}")
        print("-"*80)

        for i, opp in enumerate(positive_opps[:20], 1):
            print(f"{i:<4} {opp['game']:<22} {opp['description']:<25} "
                  f"{opp['hit_rate']:>5.1%}  "
                  f"{opp['multiplier']:>5.2f}x ({opp['american_odds']:>+5}) "
                  f"{opp['ev']:>+7.1%}  "
                  f"{opp['fair_odds']:>+5}")

    # Suggest SGPs
    print(f"\n{'='*80}")
    print(f"SAME-GAME PARLAY SUGGESTIONS")
    print(f"{'='*80}")

    sgp_suggestions = suggest_sgp(opportunities_by_game, min_ev=args.min_ev, max_legs=3)

    if sgp_suggestions:
        for i, sgp in enumerate(sgp_suggestions[:5], 1):
            print(f"\n#{i}: {sgp['game']} - {sgp['n_legs']}-leg parlay")
            print(f"   Multiplier: {sgp['multiplier']:.2f}x, Joint Prob: {sgp['joint_prob']:.2%}, EV: {sgp['ev']:+.1%}")
            for j, leg in enumerate(sgp['legs'], 1):
                print(f"   {j}. {leg['description']:<30} {leg['hit_rate']:>5.1%} ({leg['multiplier']:.2f}x) [{leg['bet_type']}]")
    else:
        print("No SGP combinations found")

    # Suggest MGPs
    if len(args.games) > 1:
        print(f"\n{'='*80}")
        print(f"MULTI-GAME PARLAY SUGGESTIONS")
        print(f"{'='*80}")

        mgp_suggestions = suggest_mgp(all_opportunities, min_ev=args.min_ev, max_legs=4)

        if mgp_suggestions:
            for i, mgp in enumerate(mgp_suggestions[:5], 1):
                print(f"\n#{i}: {mgp['n_legs']}-game parlay")
                print(f"   Multiplier: {mgp['multiplier']:.2f}x, Joint Prob: {mgp['joint_prob']:.2%}, EV: {mgp['ev']:+.1%}")
                for j, leg in enumerate(mgp['legs'], 1):
                    print(f"   {j}. {leg['game']:<22} {leg['description']:<25} {leg['hit_rate']:>5.1%} ({leg['multiplier']:.2f}x)")
        else:
            print("No MGP combinations found")

    print()


if __name__ == "__main__":
    main()
