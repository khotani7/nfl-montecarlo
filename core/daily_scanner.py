#!/usr/bin/env python3
"""
Daily NFL betting opportunity scanner.

Scans all games for a given day/week, finds at-the-money (ATM) lines close to 50/50,
and recommends the best individual bets and parlay combinations (both SGP and MGP).

Usage:
    # Scan today's games
    python daily_scanner.py

    # Scan specific week
    python daily_scanner.py --week 6 --season 2026

    # High accuracy scan
    python daily_scanner.py --week 6 --sims 100000

    # Show top parlay suggestions
    python daily_scanner.py --week 6 --top-parlays 10
"""

import argparse
import sys
import os
import numpy as np
from datetime import datetime
from itertools import combinations

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from main script
from nfl_montecarlo import (
    load_nflverse, team_ratings, model_expectation, fetch_polymarket_event,
    parse_polymarket_event, fit_normal_from_ladder, calibrate_scale, run_simulation,
    prob_to_american, multiplier_to_american, polymarket_price_to_multiplier
)
from scipy import stats


def find_atm_lines(markets, market_type, home_nick, away_nick, line_range=None):
    """
    Find at-the-money lines (close to 50/50 pricing).

    Args:
        markets: List of market dicts from parse_polymarket_event
        market_type: "totals", "spreads", etc.
        home_nick: Home team nickname for spread parsing
        away_nick: Away team nickname for spread parsing
        line_range: For spreads, only include lines within this range (e.g., 6.5)

    Returns:
        List of bet detail dicts
    """
    atm_lines = []

    for market in markets:
        if market.get("type") != market_type:
            continue

        line = market.get("line")
        if line is None:
            continue

        # Filter spreads by range
        if market_type == "spreads" and line_range is not None:
            if abs(line) > line_range:
                continue

        prices = market.get("prices", [])
        multipliers = market.get("multipliers", [])
        outcomes = market.get("outcomes", [])

        if len(prices) != 2 or len(multipliers) != 2:
            continue

        # Normalize prices
        total_price = sum(prices)
        if total_price <= 0:
            continue

        norm_prices = [p / total_price for p in prices]

        # Check if close to 50/50 (within 10% of 50%)
        # ATM means both sides are between 40% and 60%
        for i, (outcome, price, mult) in enumerate(zip(outcomes, norm_prices, multipliers)):
            if 0.40 <= price <= 0.60:
                bet_detail = {
                    "line": line,
                    "side": outcome,
                    "multiplier": mult,
                    "price": price
                }

                # For spreads, determine which team and actual line
                if market_type == "spreads":
                    # Polymarket format: Line is already signed, first outcome is the team with spread
                    # E.g., Line: -6.5, Outcomes: ['Ravens', 'Browns'] = Ravens -6.5
                    # Check if this outcome is the first in the outcomes array
                    is_first_outcome = (i == 0)

                    if is_first_outcome:
                        # This outcome has the spread (line is already signed correctly)
                        if home_nick.lower() in outcome.lower():
                            bet_detail["team"] = "home"
                            bet_detail["team_line"] = line  # Use signed line as-is
                        else:
                            bet_detail["team"] = "away"
                            bet_detail["team_line"] = line  # Use signed line as-is
                    else:
                        # Skip second outcome (it's the opposite side of same bet)
                        continue

                atm_lines.append(bet_detail)

    return atm_lines


def simulate_opportunity(schedules, teams, home, away, bet_type, bet_details, args):
    """Simulate a single betting opportunity and return EV."""
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

    # Evaluate based on bet type
    hit_rate = None
    description = ""

    if bet_type == "spread":
        team_line = bet_details.get("team_line")
        team = bet_details.get("team")

        if team == "home":
            # Home team with the spread
            # For home -6.5: margin > 6.5 (home wins by more than 6.5)
            # Flip the sign because Polymarket encodes home favorites with negative lines
            hit_rate = float(np.mean(margin > -team_line))
            description = f"{home} {team_line:+g}"
        else:
            # Away team with the spread
            # For away -6.5: -margin > 6.5 (away wins by more than 6.5)
            hit_rate = float(np.mean(-margin > -team_line))
            description = f"{away} {team_line:+g}"

    elif bet_type == "total":
        line = bet_details["line"]
        if "over" in bet_details["side"].lower():
            hit_rate = float(np.mean(total_pts > line))
            description = f"Over {line}"
        else:
            hit_rate = float(np.mean(total_pts < line))
            description = f"Under {line}"

    elif bet_type == "1h_total":
        line = bet_details["line"]
        if "over" in bet_details["side"].lower():
            hit_rate = float(np.mean(total_1h > line))
            description = f"1H Over {line}"
        else:
            hit_rate = float(np.mean(total_1h < line))
            description = f"1H Under {line}"

    elif bet_type == "2h_total":
        line = bet_details["line"]
        if "over" in bet_details["side"].lower():
            hit_rate = float(np.mean(total_2h > line))
            description = f"2H Over {line}"
        else:
            hit_rate = float(np.mean(total_2h < line))
            description = f"2H Under {line}"

    elif bet_type == "moneyline":
        if bet_details["team"] == home:
            hit_rate = float(np.mean(margin > 0))
            description = f"{home} ML"
        else:
            hit_rate = float(np.mean(margin < 0))
            description = f"{away} ML"

    if hit_rate is None:
        return None

    multiplier = bet_details["multiplier"]
    ev = hit_rate * multiplier - 1
    fair_odds = prob_to_american(hit_rate)

    return {
        "game": f"{away} @ {home}",
        "home": home,
        "away": away,
        "description": description,
        "bet_type": bet_type,
        "bet_details": bet_details,
        "hit_rate": hit_rate,
        "multiplier": multiplier,
        "ev": ev,
        "fair_odds": fair_odds,
        "american_odds": multiplier_to_american(multiplier)
    }


def scan_game(schedules, teams, home, away, args):
    """Scan a single game for all ATM opportunities."""
    nick = dict(zip(teams["team_abbr"].to_list(), teams["team_nick"].to_list()))
    home_nick, away_nick = nick.get(home, home), nick.get(away, away)

    event = fetch_polymarket_event(home_nick, away_nick)

    if not event:
        return []

    pm = parse_polymarket_event(event, home_nick, away_nick)
    opportunities = []

    # Find ATM spreads (within ±6.5)
    atm_spreads = find_atm_lines(pm["markets"], "spreads", home_nick, away_nick, line_range=args.spread_range)
    for spread in atm_spreads:
        opp = simulate_opportunity(schedules, teams, home, away, "spread", spread, args)
        if opp:
            opportunities.append(opp)

    # Find ATM totals
    atm_totals = find_atm_lines(pm["markets"], "totals", home_nick, away_nick)
    for total in atm_totals:
        opp = simulate_opportunity(schedules, teams, home, away, "total", total, args)
        if opp:
            opportunities.append(opp)

    # Check moneyline if close to 50/50
    if pm["moneyline"] and 0.40 <= pm["moneyline"] <= 0.60:
        # Get moneyline multipliers from markets
        for market in pm["markets"]:
            if market.get("type") == "moneyline":
                for i, outcome in enumerate(market.get("outcomes", [])):
                    mult = market["multipliers"][i] if i < len(market["multipliers"]) else None
                    if mult:
                        team = home if home_nick.lower() in outcome.lower() else away
                        bet_details = {"team": team, "multiplier": mult}
                        opp = simulate_opportunity(schedules, teams, home, away, "moneyline", bet_details, args)
                        if opp:
                            opportunities.append(opp)

    return opportunities


def suggest_sgp(opportunities_by_game, min_ev=0.0, max_legs=3):
    """
    Suggest best same-game parlays.

    SGP Rules:
    - Maximum ONE spread or moneyline bet per parlay
    - Maximum ONE game total (can't have Over 40.5 AND Over 42.5)
    - Maximum ONE 1H total
    - Maximum ONE 2H total
    - Maximum ONE prop per player per stat
    """
    sgp_suggestions = []

    for game, opps in opportunities_by_game.items():
        # Filter positive EV opportunities
        positive_opps = [o for o in opps if o["ev"] >= min_ev]

        if len(positive_opps) < 2:
            continue

        # Separate by bet type
        spread_ml_bets = [o for o in positive_opps if o["bet_type"] in ["spread", "moneyline"]]
        game_totals = [o for o in positive_opps if o["bet_type"] == "total"]
        h1_totals = [o for o in positive_opps if o["bet_type"] == "1h_total"]
        h2_totals = [o for o in positive_opps if o["bet_type"] == "2h_total"]
        player_props = [o for o in positive_opps if o["bet_type"] == "player"]

        # For each category, pick the BEST opportunity (highest EV)
        best_spread_ml = max(spread_ml_bets, key=lambda x: x["ev"]) if spread_ml_bets else None
        best_game_total = max(game_totals, key=lambda x: x["ev"]) if game_totals else None
        best_1h_total = max(h1_totals, key=lambda x: x["ev"]) if h1_totals else None
        best_2h_total = max(h2_totals, key=lambda x: x["ev"]) if h2_totals else None

        # For player props, group by player and stat, take best from each
        player_stat_groups = {}
        for prop in player_props:
            # Extract player and stat from bet_details or description
            # For now, we'll use description as a proxy
            key = prop["description"]  # Would need better player/stat extraction
            if key not in player_stat_groups or prop["ev"] > player_stat_groups[key]["ev"]:
                player_stat_groups[key] = prop

        best_player_props = list(player_stat_groups.values())

        # Build available components for SGP
        available_components = []
        if best_spread_ml:
            available_components.append(best_spread_ml)
        if best_game_total:
            available_components.append(best_game_total)
        if best_1h_total:
            available_components.append(best_1h_total)
        if best_2h_total:
            available_components.append(best_2h_total)
        available_components.extend(best_player_props)

        # Need at least 2 components for a parlay
        if len(available_components) < 2:
            continue

        # Generate all valid combinations (2 to max_legs)
        for n_legs in range(2, min(max_legs + 1, len(available_components) + 1)):
            for combo in combinations(available_components, n_legs):
                # Enforce rule: max ONE spread/ML per parlay
                spread_ml_count = sum(1 for o in combo if o["bet_type"] in ["spread", "moneyline"])
                if spread_ml_count > 1:
                    continue

                # Calculate parlay value
                joint_prob = 1.0
                parlay_mult = 1.0

                for opp in combo:
                    joint_prob *= opp["hit_rate"]
                    parlay_mult *= opp["multiplier"]

                parlay_ev = joint_prob * parlay_mult - 1

                if parlay_ev >= min_ev:
                    sgp_suggestions.append({
                        "game": game,
                        "legs": combo,
                        "n_legs": n_legs,
                        "joint_prob": joint_prob,
                        "multiplier": parlay_mult,
                        "ev": parlay_ev
                    })

    # Sort by EV
    sgp_suggestions.sort(key=lambda x: x["ev"], reverse=True)
    return sgp_suggestions


def suggest_mgp(all_opportunities, min_ev=0.0, max_legs=4):
    """Suggest best multi-game parlays."""
    mgp_suggestions = []

    # Filter positive EV opportunities
    positive_opps = [o for o in all_opportunities if o["ev"] >= min_ev]

    if len(positive_opps) < 2:
        return []

    # Try 2-leg through max_legs combinations
    for n_legs in range(2, min(max_legs + 1, len(positive_opps) + 1)):
        for combo in combinations(positive_opps, n_legs):
            # Ensure different games
            games = set([o["game"] for o in combo])
            if len(games) != n_legs:
                continue  # Skip if same game appears twice

            # Calculate joint probability (independent games)
            joint_prob = 1.0
            parlay_mult = 1.0

            for opp in combo:
                joint_prob *= opp["hit_rate"]
                parlay_mult *= opp["multiplier"]

            parlay_ev = joint_prob * parlay_mult - 1

            if parlay_ev >= min_ev:
                mgp_suggestions.append({
                    "legs": combo,
                    "n_legs": n_legs,
                    "joint_prob": joint_prob,
                    "multiplier": parlay_mult,
                    "ev": parlay_ev
                })

    # Sort by EV
    mgp_suggestions.sort(key=lambda x: x["ev"], reverse=True)
    return mgp_suggestions


def main():
    ap = argparse.ArgumentParser(description="Daily NFL betting opportunity scanner")
    ap.add_argument("--week", type=int, help="NFL week to scan (default: current week)")
    ap.add_argument("--season", type=int, default=2026, help="NFL season")
    ap.add_argument("--sims", type=int, default=50000, help="Simulations per opportunity")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--min-ev", type=float, default=0.0, help="Minimum EV to display (default: 0.0)")
    ap.add_argument("--spread-range", type=float, default=6.5, help="Only spreads within ±X (default: 6.5)")
    ap.add_argument("--top-bets", type=int, default=20, help="Number of top individual bets to show")
    ap.add_argument("--top-parlays", type=int, default=10, help="Number of top parlays to show")
    args = ap.parse_args()

    # Use current week if not specified
    if args.week is None:
        args.week = 6  # Default to week 6 for now

    print("="*80)
    print(f"NFL DAILY BETTING SCANNER - Week {args.week}, {args.season}")
    print("="*80)
    print(f"Simulations: {args.sims:,} per opportunity")
    print(f"Minimum EV: {args.min_ev:+.1%}")
    print(f"Spread Range: ±{args.spread_range}")
    print()

    # Load data
    print("Loading nflverse data...")
    schedules, players, teams = load_nflverse(args.season)

    # Filter to specified week
    week_games = schedules.filter(schedules["week"] == args.week)

    if len(week_games) == 0:
        print(f"No games found for week {args.week}")
        return

    print(f"Found {len(week_games)} games in week {args.week}\n")

    # Scan each game
    all_opportunities = []
    opportunities_by_game = {}

    for i, row in enumerate(week_games.iter_rows(named=True), 1):
        home = row["home_team"]
        away = row["away_team"]
        game_key = f"{away} @ {home}"

        print(f"[{i}/{len(week_games)}] Scanning {game_key}...", end=" ", flush=True)

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

    # Display top individual bets
    if positive_opps:
        positive_opps.sort(key=lambda x: x["ev"], reverse=True)

        print("="*80)
        print(f"TOP {args.top_bets} INDIVIDUAL BETS (Ranked by EV)")
        print("="*80)
        print(f"{'#':<4} {'Game':<22} {'Bet':<25} {'Hit%':<8} {'Mult':<12} {'EV':<10} {'Fair'}")
        print("-"*80)

        for i, opp in enumerate(positive_opps[:args.top_bets], 1):
            print(f"{i:<4} {opp['game']:<22} {opp['description']:<25} "
                  f"{opp['hit_rate']:>5.1%}  "
                  f"{opp['multiplier']:>5.2f}x ({opp['american_odds']:>+5}) "
                  f"{opp['ev']:>+7.1%}  "
                  f"{opp['fair_odds']:>+5}")

    # Suggest SGPs
    print(f"\n{'='*80}")
    print(f"TOP {args.top_parlays} SAME-GAME PARLAY SUGGESTIONS")
    print(f"{'='*80}")

    sgp_suggestions = suggest_sgp(opportunities_by_game, min_ev=args.min_ev, max_legs=3)

    if sgp_suggestions:
        for i, sgp in enumerate(sgp_suggestions[:args.top_parlays], 1):
            print(f"\n#{i}: {sgp['game']} - {sgp['n_legs']}-leg parlay")
            print(f"   Multiplier: {sgp['multiplier']:.2f}x, Joint Prob: {sgp['joint_prob']:.2%}, EV: {sgp['ev']:+.1%}")
            for j, leg in enumerate(sgp['legs'], 1):
                print(f"   {j}. {leg['description']:<30} {leg['hit_rate']:>5.1%} ({leg['multiplier']:.2f}x)")
    else:
        print("No SGP combinations found with positive EV")

    # Suggest MGPs
    print(f"\n{'='*80}")
    print(f"TOP {args.top_parlays} MULTI-GAME PARLAY SUGGESTIONS")
    print(f"{'='*80}")

    mgp_suggestions = suggest_mgp(all_opportunities, min_ev=args.min_ev, max_legs=4)

    if mgp_suggestions:
        for i, mgp in enumerate(mgp_suggestions[:args.top_parlays], 1):
            print(f"\n#{i}: {mgp['n_legs']}-game parlay")
            print(f"   Multiplier: {mgp['multiplier']:.2f}x, Joint Prob: {mgp['joint_prob']:.2%}, EV: {mgp['ev']:+.1%}")
            for j, leg in enumerate(mgp['legs'], 1):
                print(f"   {j}. {leg['game']:<22} {leg['description']:<25} {leg['hit_rate']:>5.1%} ({leg['multiplier']:.2f}x)")
    else:
        print("No MGP combinations found with positive EV")

    print()


if __name__ == "__main__":
    main()
