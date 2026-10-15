# Backtest Results

This directory contains historical backtest results used to validate the NFL Monte Carlo model.

## Key Files

### `backtest_5season_75_spreads_v2.csv` (Recommended Reference)
**The validated optimal strategy results.**

- **Seasons:** 2021-2025 (5 full seasons)
- **Model Weight:** 75% (optimal)
- **Bet Type:** Spreads only
- **Simulations:** 100,000 per game
- **Total Bets:** 1,930 (both sides of each game)
- **Key Finding:** 55-60% probability bucket shows +3.25% ROI (294 bets)

**This is the primary evidence for the optimal strategy documented in `docs/OPTIMAL_STRATEGY.md`.**

### `backtest_multi.csv`
**Comprehensive multi-season test with all bet types.**

- **Seasons:** Multiple (likely 2021-2025)
- **Model Weight:** Likely default (25%)
- **Bet Types:** Spreads, totals, moneylines
- **Use Case:** Demonstrates that default settings lose money

### `my_backtest.csv`
**User-generated backtest results.**

- Created by user during testing
- May contain experimental configurations
- Check file headers to determine exact parameters

## How to Analyze These Files

### Quick Analysis
```bash
# Analyze the optimal strategy results
python core/backtest_analysis.py backtest_results/backtest_5season_75_spreads_v2.csv

# Focus on specific EV thresholds
python core/backtest_analysis.py backtest_results/backtest_5season_75_spreads_v2.csv \
    --ev-thresholds 0.0 0.01 0.02 0.03 0.05
```

### Custom Analysis
Load the CSV in Python/pandas:
```python
import pandas as pd

# Load results
df = pd.read_csv('backtest_results/backtest_5season_75_spreads_v2.csv')

# Filter for 55-60% probability bucket
sweet_spot = df[(df['prediction_prob'] >= 0.55) & (df['prediction_prob'] < 0.60)]

# Calculate ROI
wins = sweet_spot[sweet_spot['actual_won'] == True]
losses = sweet_spot[sweet_spot['actual_won'] == False]

profit = 0
for _, row in sweet_spot.iterrows():
    if row['actual_won']:
        if row['odds'] < 0:
            profit += 100 / abs(row['odds'])
        else:
            profit += row['odds'] / 100
    else:
        profit -= 1

roi = (profit / len(sweet_spot)) * 100
print(f"55-60% Bucket ROI: {roi:.2f}%")
```

## CSV Column Definitions

Each backtest CSV contains the following columns:

- `season`: NFL season (e.g., 2023)
- `week`: Week number (6-18, validated from week 6+)
- `home`: Home team abbreviation
- `away`: Away team abbreviation
- `home_score`: Actual final home score
- `away_score`: Actual final away score
- `bet_type`: Type of bet (spread, total, moneyline)
- `side`: Which side of bet (home, away, over, under)
- `line`: Betting line (e.g., -2.5 for spreads, 45.5 for totals)
- `odds`: American odds (e.g., -110)
- `prediction_prob`: Model's predicted win probability (0.0-1.0)
- `actual_won`: Boolean, whether bet actually won
- `actual_margin`: Actual point differential (home - away)
- `actual_total`: Actual total points scored
- `ev`: Expected Value as decimal (e.g., 0.0325 = 3.25% EV)
- `model_weight`: Model weight used (e.g., 0.75 = 75%)
- `sims`: Number of simulations run per game

## Reproducing These Results

### Reproduce Optimal Strategy Test
```bash
python core/backtest.py \
    --seasons 2021 2022 2023 2024 2025 \
    --bet-types spread \
    --model-weight 0.75 \
    --sims 100000 \
    --min-week 6 \
    --seed 42 \
    --output backtest_results/backtest_5season_75_spreads_v3.csv
```

### Reproduce Multi-Season Test
```bash
python core/backtest.py \
    --seasons 2021 2022 2023 2024 2025 \
    --bet-types spread total moneyline \
    --sims 50000 \
    --min-week 6 \
    --seed 42 \
    --output backtest_results/backtest_multi_v2.csv
```

## File Retention

**Keep These:**
- ✅ `backtest_5season_75_spreads_v2.csv` - Primary validation evidence
- ✅ `backtest_multi.csv` - Reference for default settings failure

**Optional:**
- `my_backtest.csv` - User testing, can be deleted if no longer needed

## Notes

- All backtests use **walk-forward validation** (no look-ahead bias)
- Minimum week 6 start ensures sufficient prior data for team ratings
- Random seed 42 used for reproducibility
- Results are deterministic given same seed and configuration
