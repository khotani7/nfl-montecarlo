#!/usr/bin/env python3
"""
Backtest the NFL Monte Carlo model against historical games.

This script performs walk-forward validation:
- For each historical game, uses only prior games in the season for team ratings
- Simulates the game using the model
- Compares predictions against actual results
- Tracks performance metrics (hit rate, ROI, calibration, etc.)

Usage:
    # Backtest a single season
    python backtest.py --seasons 2023 --sims 50000

    # Backtest multiple seasons
    python backtest.py --seasons 2022 2023 2024 --sims 100000

    # Test specific bet types and EV thresholds
    python backtest.py --seasons 2023 --bet-types spread total \
        --ev-thresholds 0.00 0.02 0.05 --sims 50000

    # Save detailed results to CSV
    python backtest.py --seasons 2023 --output backtest_2023.csv
"""

import argparse
import sys
import os
import csv
import numpy as np
import polars as pl
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'core'))

from nfl_montecarlo import (
    load_nflverse, team_ratings, model_expectation, calibrate_scale,
    run_simulation, american_to_prob, prob_to_american, payout_multiplier
)


@dataclass
class BacktestResult:
    """Result of a single game backtest."""
    season: int
    week: int
    home: str
    away: str
    home_score: int
    away_score: int
    bet_type: str
    side: str  # 'home', 'away', 'over', 'under'
    line: float
    odds: int
    prediction_prob: float  # Model's predicted probability of winning
    actual_won: bool
    actual_margin: float  # home_score - away_score
    actual_total: float  # home_score + away_score
    ev: float
    model_weight: float
    sims: int


@dataclass
class BacktestMetrics:
    """Aggregated performance metrics."""
    total_bets: int
    wins: int
    losses: int
    hit_rate: float
    avg_odds: float
    roi: float  # Return on investment (%)
    profit: float  # Profit in units (assuming $1 per bet)
    brier_score: float  # Lower is better (0 = perfect)
    log_loss: float  # Lower is better
    calibration_buckets: Dict[str, Dict]  # Predicted prob range -> actual hit rate


def backtest_game(
    schedules: pl.DataFrame,
    players: pl.DataFrame,
    home: str,
    away: str,
    week: int,
    season: int,
    bet_types: List[str],
    model_weight: float,
    n_sims: int,
    rng: np.random.Generator
) -> List[BacktestResult]:
    """
    Backtest a single game.

    Returns predictions for all specified bet types.
    Uses only games before this week for team ratings (walk-forward validation).
    """
    results = []

    # Get actual game result
    game = schedules.filter(
        (pl.col("home_team") == home) &
        (pl.col("away_team") == away) &
        (pl.col("week") == week)
    )

    if game.height == 0:
        return []

    home_score = game["home_score"][0]
    away_score = game["away_score"][0]

    # Skip if game hasn't been played yet
    if home_score is None or away_score is None:
        return []

    home_score = float(home_score)
    away_score = float(away_score)
    actual_margin = home_score - away_score
    actual_total = home_score + away_score

    # Get sportsbook lines
    spread_line = game["spread_line"][0]
    total_line = game["total_line"][0]

    if spread_line is None and total_line is None:
        return []  # No lines available for this game

    # Build team ratings using only prior games in this season
    prior_games = schedules.filter(
        (pl.col("week") < week) &
        (pl.col("season") == season) &
        pl.col("home_score").is_not_null()
    )

    ratings = team_ratings(prior_games)

    # Get model expectation (returns home_pts, away_pts)
    home_pts, away_pts = model_expectation(ratings, home, away)
    model_margin = home_pts - away_pts
    model_total = home_pts + away_pts

    # Blend with market (sportsbook lines)
    if spread_line is not None:
        exp_margin = model_weight * model_margin + (1 - model_weight) * float(spread_line)
    else:
        exp_margin = model_margin

    if total_line is not None:
        exp_total = model_weight * model_total + (1 - model_weight) * float(total_line)
    else:
        exp_total = model_total

    # Convert margin and total to expected points for each team
    home_exp = max((exp_total + exp_margin) / 2, 3.0)
    away_exp = max((exp_total - exp_margin) / 2, 3.0)

    # Calibrate scale to match typical variance
    # Using default margin SD of 13.5
    margin_sd = 13.5
    eff_scale = calibrate_scale(rng, home_exp, away_exp, margin_sd)

    # Run simulation
    home_scores, away_scores, _, _, _, _, _ = run_simulation(
        rng=rng,
        home_exp=home_exp,
        away_exp=away_exp,
        eff_scale=eff_scale,
        player_dists={},  # No player props in backtest
        leg_corrs={},
        n_sims=n_sims
    )

    # Calculate predictions for each bet type

    # Moneyline
    if "moneyline" in bet_types:
        home_win_rate = float(np.mean(home_scores > away_scores))
        away_win_rate = float(np.mean(away_scores > home_scores))

        # Home moneyline
        home_ml_odds = prob_to_american(home_win_rate)
        home_ml_ev = home_win_rate * payout_multiplier(home_ml_odds) - 1
        results.append(BacktestResult(
            season=season,
            week=week,
            home=home,
            away=away,
            home_score=int(home_score),
            away_score=int(away_score),
            bet_type="moneyline",
            side="home",
            line=0.0,
            odds=home_ml_odds,
            prediction_prob=home_win_rate,
            actual_won=(home_score > away_score),
            actual_margin=actual_margin,
            actual_total=actual_total,
            ev=home_ml_ev,
            model_weight=model_weight,
            sims=n_sims
        ))

        # Away moneyline
        away_ml_odds = prob_to_american(away_win_rate)
        away_ml_ev = away_win_rate * payout_multiplier(away_ml_odds) - 1
        results.append(BacktestResult(
            season=season,
            week=week,
            home=home,
            away=away,
            home_score=int(home_score),
            away_score=int(away_score),
            bet_type="moneyline",
            side="away",
            line=0.0,
            odds=away_ml_odds,
            prediction_prob=away_win_rate,
            actual_won=(away_score > home_score),
            actual_margin=actual_margin,
            actual_total=actual_total,
            ev=away_ml_ev,
            model_weight=model_weight,
            sims=n_sims
        ))

    # Spread
    if "spread" in bet_types and spread_line is not None:
        spread = float(spread_line)

        # Home covers if (home_score - away_score) > spread
        home_cover_rate = float(np.mean((home_scores - away_scores) > spread))
        home_spread_odds = -110  # Typical spread odds
        home_spread_ev = home_cover_rate * payout_multiplier(home_spread_odds) - 1

        results.append(BacktestResult(
            season=season,
            week=week,
            home=home,
            away=away,
            home_score=int(home_score),
            away_score=int(away_score),
            bet_type="spread",
            side="home",
            line=spread,
            odds=home_spread_odds,
            prediction_prob=home_cover_rate,
            actual_won=(actual_margin > spread),
            actual_margin=actual_margin,
            actual_total=actual_total,
            ev=home_spread_ev,
            model_weight=model_weight,
            sims=n_sims
        ))

        # Away covers if (away_score - home_score) > (-spread)
        away_cover_rate = float(np.mean((away_scores - home_scores) > (-spread)))
        away_spread_odds = -110
        away_spread_ev = away_cover_rate * payout_multiplier(away_spread_odds) - 1

        results.append(BacktestResult(
            season=season,
            week=week,
            home=home,
            away=away,
            home_score=int(home_score),
            away_score=int(away_score),
            bet_type="spread",
            side="away",
            line=-spread,
            odds=away_spread_odds,
            prediction_prob=away_cover_rate,
            actual_won=(actual_margin < spread),
            actual_margin=actual_margin,
            actual_total=actual_total,
            ev=away_spread_ev,
            model_weight=model_weight,
            sims=n_sims
        ))

    # Total
    if "total" in bet_types and total_line is not None:
        total = float(total_line)

        over_rate = float(np.mean((home_scores + away_scores) > total))
        over_odds = -110
        over_ev = over_rate * payout_multiplier(over_odds) - 1

        results.append(BacktestResult(
            season=season,
            week=week,
            home=home,
            away=away,
            home_score=int(home_score),
            away_score=int(away_score),
            bet_type="total",
            side="over",
            line=total,
            odds=over_odds,
            prediction_prob=over_rate,
            actual_won=(actual_total > total),
            actual_margin=actual_margin,
            actual_total=actual_total,
            ev=over_ev,
            model_weight=model_weight,
            sims=n_sims
        ))

        under_rate = float(np.mean((home_scores + away_scores) < total))
        under_odds = -110
        under_ev = under_rate * payout_multiplier(under_odds) - 1

        results.append(BacktestResult(
            season=season,
            week=week,
            home=home,
            away=away,
            home_score=int(home_score),
            away_score=int(away_score),
            bet_type="total",
            side="under",
            line=total,
            odds=under_odds,
            prediction_prob=under_rate,
            actual_won=(actual_total < total),
            actual_margin=actual_margin,
            actual_total=actual_total,
            ev=under_ev,
            model_weight=model_weight,
            sims=n_sims
        ))

    return results


def calculate_metrics(
    results: List[BacktestResult],
    ev_threshold: float = 0.0
) -> BacktestMetrics:
    """
    Calculate aggregate performance metrics.

    Args:
        results: List of backtest results
        ev_threshold: Minimum EV to include a bet (e.g., 0.02 = only bet when EV > 2%)
    """
    # Filter by EV threshold
    filtered = [r for r in results if r.ev >= ev_threshold]

    if len(filtered) == 0:
        return BacktestMetrics(
            total_bets=0,
            wins=0,
            losses=0,
            hit_rate=0.0,
            avg_odds=0,
            roi=0.0,
            profit=0.0,
            brier_score=0.0,
            log_loss=0.0,
            calibration_buckets={}
        )

    wins = sum(1 for r in filtered if r.actual_won)
    losses = len(filtered) - wins
    hit_rate = wins / len(filtered)

    # Calculate profit (assuming $1 bet on each)
    profit = sum(
        payout_multiplier(r.odds) - 1 if r.actual_won else -1
        for r in filtered
    )

    roi = (profit / len(filtered)) * 100  # Return on investment (%)

    avg_odds = sum(r.odds for r in filtered) / len(filtered)

    # Brier score: mean squared error of probabilities
    brier_score = np.mean([
        (r.prediction_prob - (1.0 if r.actual_won else 0.0)) ** 2
        for r in filtered
    ])

    # Log loss (cross-entropy loss)
    log_loss = -np.mean([
        np.log(max(r.prediction_prob, 1e-15)) if r.actual_won
        else np.log(max(1 - r.prediction_prob, 1e-15))
        for r in filtered
    ])

    # Calibration buckets
    buckets = {
        "0-20%": {"predicted": [], "actual": []},
        "20-40%": {"predicted": [], "actual": []},
        "40-50%": {"predicted": [], "actual": []},
        "50-60%": {"predicted": [], "actual": []},
        "60-80%": {"predicted": [], "actual": []},
        "80-100%": {"predicted": [], "actual": []},
    }

    for r in filtered:
        prob = r.prediction_prob
        actual = 1.0 if r.actual_won else 0.0

        if prob < 0.2:
            bucket = "0-20%"
        elif prob < 0.4:
            bucket = "20-40%"
        elif prob < 0.5:
            bucket = "40-50%"
        elif prob < 0.6:
            bucket = "50-60%"
        elif prob < 0.8:
            bucket = "60-80%"
        else:
            bucket = "80-100%"

        buckets[bucket]["predicted"].append(prob)
        buckets[bucket]["actual"].append(actual)

    # Calculate average predicted vs actual for each bucket
    calibration_buckets = {}
    for bucket, data in buckets.items():
        if len(data["predicted"]) > 0:
            calibration_buckets[bucket] = {
                "count": len(data["predicted"]),
                "avg_predicted": np.mean(data["predicted"]),
                "avg_actual": np.mean(data["actual"]),
            }

    return BacktestMetrics(
        total_bets=len(filtered),
        wins=wins,
        losses=losses,
        hit_rate=hit_rate,
        avg_odds=avg_odds,
        roi=roi,
        profit=profit,
        brier_score=brier_score,
        log_loss=log_loss,
        calibration_buckets=calibration_buckets
    )


def print_metrics_report(
    metrics_by_type: Dict[str, BacktestMetrics],
    ev_threshold: float
):
    """Print a formatted metrics report."""
    print(f"\n{'='*80}")
    print(f"BACKTEST RESULTS (EV Threshold: {ev_threshold:+.1%})")
    print(f"{'='*80}\n")

    for bet_type, metrics in metrics_by_type.items():
        if metrics.total_bets == 0:
            continue

        print(f"{'─'*80}")
        print(f"{bet_type.upper()}")
        print(f"{'─'*80}")
        print(f"Total Bets:      {metrics.total_bets}")
        print(f"Wins / Losses:   {metrics.wins} / {metrics.losses}")
        print(f"Hit Rate:        {metrics.hit_rate:.2%}")
        print(f"Avg Odds:        {metrics.avg_odds:+.0f}")
        print(f"ROI:             {metrics.roi:+.2f}%")
        print(f"Profit (units):  {metrics.profit:+.2f}")
        print(f"Brier Score:     {metrics.brier_score:.4f}")
        print(f"Log Loss:        {metrics.log_loss:.4f}")

        if metrics.calibration_buckets:
            print(f"\nCalibration:")
            print(f"  {'Bucket':<15} {'Count':<8} {'Predicted':<12} {'Actual':<12} {'Diff':<10}")
            print(f"  {'-'*60}")
            for bucket, data in sorted(metrics.calibration_buckets.items()):
                diff = data['avg_actual'] - data['avg_predicted']
                print(f"  {bucket:<15} {data['count']:<8} "
                      f"{data['avg_predicted']:<12.2%} {data['avg_actual']:<12.2%} "
                      f"{diff:+.2%}")

        print()


def main():
    parser = argparse.ArgumentParser(description="Backtest NFL Monte Carlo model")
    parser.add_argument("--seasons", nargs="+", type=int, required=True,
                        help="Seasons to backtest (e.g., 2022 2023 2024)")
    parser.add_argument("--bet-types", nargs="+", default=["spread", "total", "moneyline"],
                        choices=["spread", "total", "moneyline"],
                        help="Bet types to test")
    parser.add_argument("--ev-thresholds", nargs="+", type=float, default=[0.0, 0.02, 0.05],
                        help="EV thresholds to test (e.g., 0.00 0.02 0.05)")
    parser.add_argument("--model-weight", type=float, default=0.25,
                        help="Model weight (0=market only, 1=model only)")
    parser.add_argument("--sims", type=int, default=50000,
                        help="Simulations per game")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--min-week", type=int, default=4,
                        help="Minimum week to start backtesting (need prior games for ratings)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file for detailed results")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"NFL Monte Carlo Backtest")
    print(f"Seasons: {args.seasons}")
    print(f"Bet Types: {args.bet_types}")
    print(f"EV Thresholds: {[f'{t:+.1%}' for t in args.ev_thresholds]}")
    print(f"Model Weight: {args.model_weight:.0%}")
    print(f"Simulations: {args.sims:,}")
    print(f"Min Week: {args.min_week}")
    print()

    all_results = []

    # Process each season
    for season in args.seasons:
        print(f"\nLoading {season} season data...")
        schedules, players, teams = load_nflverse(season)

        # Filter to regular season games that have been played
        reg_season = schedules.filter(
            (pl.col("game_type") == "REG") &
            pl.col("home_score").is_not_null() &
            pl.col("away_score").is_not_null() &
            (pl.col("week") >= args.min_week)
        )

        print(f"Found {reg_season.height} completed games (weeks {args.min_week}+)")

        # Backtest each game
        for i, row in enumerate(reg_season.iter_rows(named=True), 1):
            home = row["home_team"]
            away = row["away_team"]
            week = row["week"]

            print(f"[{i}/{reg_season.height}] Week {week}: {away} @ {home}...",
                  end=" ", flush=True)

            game_results = backtest_game(
                schedules=schedules,
                players=players,
                home=home,
                away=away,
                week=week,
                season=season,
                bet_types=args.bet_types,
                model_weight=args.model_weight,
                n_sims=args.sims,
                rng=rng
            )

            all_results.extend(game_results)
            print(f"✓ {len(game_results)} bets")

    print(f"\n{'='*80}")
    print(f"Total games processed: {len(all_results) // (len(args.bet_types) * 2)}")
    print(f"Total bets simulated: {len(all_results)}")
    print(f"{'='*80}")

    # Calculate and display metrics for each EV threshold
    for ev_threshold in args.ev_thresholds:
        # Calculate metrics by bet type
        metrics_by_type = {}
        for bet_type in args.bet_types:
            type_results = [r for r in all_results if r.bet_type == bet_type]
            metrics_by_type[bet_type] = calculate_metrics(type_results, ev_threshold)

        # Overall metrics
        overall_metrics = calculate_metrics(all_results, ev_threshold)
        metrics_by_type["OVERALL"] = overall_metrics

        print_metrics_report(metrics_by_type, ev_threshold)

    # Save detailed results to CSV if requested
    if args.output:
        print(f"\nSaving detailed results to {args.output}...")
        with open(args.output, 'w', newline='') as f:
            if all_results:
                writer = csv.DictWriter(f, fieldnames=asdict(all_results[0]).keys())
                writer.writeheader()
                for result in all_results:
                    writer.writerow(asdict(result))
        print(f"✓ Saved {len(all_results)} results")


if __name__ == "__main__":
    main()
