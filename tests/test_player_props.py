#!/usr/bin/env python3
"""
Test multiple player props for a specific game.

Usage:
    python test_player_props.py --game "SEA @ DEN" \
        --prop "Emanuel Wilson" "rushing_yards" "over" 70.5 -110 \
        --prop "Courtland Sutton" "receiving_yards" "over" 65.5 -115 \
        --prop "DK Metcalf" "receptions" "over" 5.5 -120

    python test_player_props.py --game "LA @ ARI" \
        --prop "Cooper Kupp" "receiving_yards" "over" 19.5 -110 \
        --sims 100000
"""

import argparse
import sys
import os
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import from main script
from core.nfl_montecarlo import (
    load_nflverse, team_ratings, model_expectation, fetch_polymarket_event,
    parse_polymarket_event, fit_normal_from_ladder, calibrate_scale, run_simulation,
    prob_to_american, payout_multiplier, multiplier_to_american, player_distribution
)
from scipy import stats


def simulate_player_prop(schedules, teams, players, home, away, prop_details, args):
    """Simulate a single player prop and return results."""
    player_name = prop_details["player"]
    stat = prop_details["stat"]
    side = prop_details["side"]
    line = prop_details["line"]
    odds = prop_details["odds"]

    # Get team nicknames for Polymarket
    nick = dict(zip(teams["team_abbr"].to_list(), teams["team_nick"].to_list()))
    home_nick, away_nick = nick.get(home, home), nick.get(away, away)

    # Get market data
    event = fetch_polymarket_event(home_nick, away_nick)

    # Calculate expectations
    ratings = team_ratings(schedules)
    model_home, model_away = model_expectation(ratings, home, away)
    model_margin, model_total = model_home - model_away, model_home + model_away

    margin_sd, total_sd = 13.5, 13.0
    market_margin, market_total = model_margin, model_total

    if event:
        pm = parse_polymarket_event(event, home_nick, away_nick)
        mu_m, sd_m = fit_normal_from_ladder(pm["spreads"], 13.5)
        if mu_m is None and pm["moneyline"] is not None:
            mu_m = 13.5 * stats.norm.ppf(pm["moneyline"])
        if mu_m is not None:
            market_margin = mu_m
            margin_sd = sd_m if sd_m else margin_sd

        mu_t, sd_t = fit_normal_from_ladder(pm["totals"], 13.0)
        if mu_t is not None:
            market_total = mu_t
            total_sd = sd_t if sd_t else total_sd

    # Blend model and market
    w = 0.25
    mu_margin = (1 - w) * market_margin + w * model_margin
    mu_total = (1 - w) * market_total + w * model_total
    home_exp = max((mu_total + mu_margin) / 2, 3.0)
    away_exp = max((mu_total - mu_margin) / 2, 3.0)

    # Use player_distribution to get player stats
    try:
        dist = player_distribution(players, player_name, stat, args.player_last_n_games, mean_override=None)
    except ValueError as e:
        return None

    # Determine which team (home or away)
    side_key = "home" if dist.team == home else "away"

    # Build player_dists and leg_corrs for simulation
    default_corr = 0.35 if "rushing" in stat else 0.5
    player_dists = {0: (dist, side_key)}
    leg_corrs = {0: default_corr}

    # Run simulation
    rng = np.random.default_rng(args.seed)
    eff_scale = calibrate_scale(rng, home_exp, away_exp, margin_sd)
    home_pts, away_pts, player_samples, _, _, _, _ = run_simulation(
        rng, home_exp, away_exp, eff_scale, player_dists, leg_corrs, args.sims
    )

    # Get player stat simulations (key 0 since we only have one prop)
    stat_sim = player_samples[0]

    # Calculate hit rate
    if side.lower() == "over":
        hit_rate = float(np.mean(stat_sim > line))
    else:
        hit_rate = float(np.mean(stat_sim < line))

    # Calculate EV
    multiplier = payout_multiplier(odds)
    ev = hit_rate * multiplier - 1
    fair_odds = prob_to_american(hit_rate)

    # Get percentiles
    p10 = float(np.percentile(stat_sim, 10))
    p25 = float(np.percentile(stat_sim, 25))
    p50 = float(np.percentile(stat_sim, 50))
    p75 = float(np.percentile(stat_sim, 75))
    p90 = float(np.percentile(stat_sim, 90))

    return {
        "player": player_name,
        "team": dist.team,
        "stat": stat,
        "side": side,
        "line": line,
        "odds": odds,
        "hit_rate": hit_rate,
        "multiplier": multiplier,
        "ev": ev,
        "fair_odds": fair_odds,
        "recent_games": dist.games,
        "mean": dist.mean,
        "median_sim": p50,
        "p10": p10,
        "p25": p25,
        "p75": p75,
        "p90": p90
    }


def main():
    ap = argparse.ArgumentParser(description="Test multiple player props for a game")
    ap.add_argument("--game", required=True, help="Game in format 'AWAY @ HOME' (e.g., 'SEA @ DEN')")
    ap.add_argument("--prop", action="append", nargs=5, metavar=("PLAYER", "STAT", "SIDE", "LINE", "ODDS"),
                    help="Player prop: 'Player Name' stat over/under line odds (e.g., 'Emanuel Wilson' rushing_yards over 70.5 -110)")
    ap.add_argument("--season", type=int, default=2026, help="NFL season")
    ap.add_argument("--sims", type=int, default=100000, help="Simulations per prop")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--player-last-n-games", type=int, default=4, help="Number of recent games to use for player stats")
    args = ap.parse_args()

    # Parse game
    parts = args.game.split("@")
    if len(parts) != 2:
        print(f"Error: Invalid game format '{args.game}'. Use 'AWAY @ HOME'")
        return

    away = parts[0].strip()
    home = parts[1].strip()

    print("="*100)
    print(f"PLAYER PROP TESTER - {args.game}")
    print("="*100)
    print(f"Simulations: {args.sims:,} per prop")
    print(f"Player Game Logs: Last {args.player_last_n_games} games")
    print()

    # Load data
    print("Loading nflverse data...")
    schedules, players, teams = load_nflverse(args.season)
    print()

    # Test each prop
    results = []
    for i, prop in enumerate(args.prop, 1):
        player_name, stat, side, line_str, odds_str = prop

        try:
            line = float(line_str)
            odds = int(odds_str)
        except ValueError:
            print(f"[{i}/{len(args.prop)}] Error: Invalid line or odds for {player_name}")
            continue

        print(f"[{i}/{len(args.prop)}] Testing {player_name} {side} {line} {stat}...", end=" ", flush=True)

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
            results.append(result)
        else:
            print("✗ Player or stat not found")

    # Display results
    if results:
        print(f"\n{'='*100}")
        print(f"RESULTS - {len(results)} Props Tested")
        print(f"{'='*100}")
        print(f"{'Player':<20} {'Prop':<30} {'Last 4':<15} {'Med':<6} {'Hit%':<8} {'Odds':<7} {'EV':<10} {'Fair'}")
        print("-"*100)

        for r in results:
            prop_desc = f"{r['side'].title()} {r['line']} {r['stat'].replace('_', ' ')}"
            last_4 = str(r['recent_games'][:4]).replace('[', '').replace(']', '')

            print(f"{r['player']:<20} {prop_desc:<30} {last_4:<15} {r['median_sim']:<6.1f} "
                  f"{r['hit_rate']:>5.1%}  {r['odds']:>+5}  {r['ev']:>+7.1%}  {r['fair_odds']:>+5}")

        print()

        # Show detailed distribution for each prop
        print(f"{'='*100}")
        print(f"DETAILED DISTRIBUTIONS")
        print(f"{'='*100}")

        for r in results:
            print(f"\n{r['player']} - {r['side'].title()} {r['line']} {r['stat'].replace('_', ' ')}")
            print(f"  Last {len(r['recent_games'])} games: {r['recent_games']}")
            print(f"  Historical average: {r['mean']:.1f}")
            print(f"  Simulated distribution: 10th: {r['p10']:.0f} | 25th: {r['p25']:.0f} | "
                  f"50th: {r['median_sim']:.0f} | 75th: {r['p75']:.0f} | 90th: {r['p90']:.0f}")
            print(f"  Hit rate: {r['hit_rate']:.1%} | Fair odds: {r['fair_odds']:+d} | "
                  f"Book odds: {r['odds']:+d} | EV: {r['ev']:+.1%}")

    print()


if __name__ == "__main__":
    main()
