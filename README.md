# NFL Monte Carlo Betting Simulator

A drive-by-drive Monte Carlo simulator for NFL games that integrates market data from Polymarket to find betting edges on spreads, totals, and player props.

## 🎯 What This Does

Simulates NFL games 100,000+ times using:
- **Polymarket market prices** (default: 75% weight) - sharp crowd wisdom
- **Team performance data** (default: 25% weight) - season stats from nflverse
- **Drive-by-drive simulation** - realistic score distributions with key numbers (3, 7)
- **Player prop modeling** - correlated with team offensive performance

**⚡ Optimal Configuration:** Use `--model-weight 0.75` for best results (see validated strategy above)

**Output:** Hit rates, expected value (EV), and fair odds for any bet you want to test.

## ⭐ VALIDATED STRATEGY (NEW!)

**After backtesting 5 NFL seasons (2021-2025), we found a proven edge:**

- **Strategy:** Bet spreads with **75% model weight** when predicted probability is **55-60%**
- **Performance:** +3.25% ROI, 54.08% hit rate (294 bets over 5 seasons)
- **Expected Returns:** ~$190 profit/season at $100/bet (59 bets/year)

**Quick Start:**
```bash
# Use optimal model weight
python core/nfl_montecarlo.py --model-weight 0.75 --sims 100000

# Then filter output for 55-60% predicted win probability
```

📖 **Full Strategy Guide:** [docs/OPTIMAL_STRATEGY.md](docs/OPTIMAL_STRATEGY.md)

## 📊 Current Model Strengths

### ✅ What Works Well
- **Team spreads & totals**: Drive-based simulation produces realistic score distributions
- **At-the-money (ATM) line scanning**: Automatically finds 50/50 bets with +EV
- **Polymarket integration**: Uses live market data from Gamma API
- **Parlay analysis**: Both same-game and multi-game parlays
- **Key numbers**: 3 and 7-point margins emerge naturally from TD/FG mechanics

### ⚠️ What Needs Work
- **Player props**: Currently too simplistic (only uses last 4 games)
- **No anytime TD support**: Touchdown props not yet implemented
- **Limited matchup context**: Doesn't account for defensive strength, weather, injuries

## 🚀 Quick Start

### Setup

```bash
# Clone and setup
cd nfl_montecarlo
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run simulation with current CONFIG
python core/nfl_montecarlo.py

# High-accuracy simulation
python core/nfl_montecarlo.py --sims 200000

# Use Polymarket odds automatically
python core/nfl_montecarlo.py --use-polymarket-odds --sims 100000

# Find best available bets
python core/nfl_montecarlo.py --find-best-bets --min-ev 0.05
```

### Daily Scanner (Recommended!)

Scan all games for a week to find at-the-money opportunities:

```bash
# Scan current week
python core/daily_scanner.py --week 6 --season 2026 --sims 100000

# Show top 20 individual bets and top 10 parlays
python core/daily_scanner.py --week 6 --top-bets 20 --top-parlays 10
```

### Backtesting (Validate Your Model!) ✨ NEW

Test your model against historical NFL games to see how it would have performed:

```bash
# Backtest 2023 season
python core/backtest.py --seasons 2023 --sims 50000

# Test multiple seasons for robust statistics
python core/backtest.py --seasons 2022 2023 2024 --sims 100000 --output backtest.csv

# Analyze results in depth
python core/backtest_analysis.py backtest.csv
```

**Features:**
- Walk-forward validation (no look-ahead bias)
- Test different EV thresholds to find optimal strategy
- Comprehensive metrics: hit rate, ROI, Brier score, calibration
- Performance tracking by week, bet type, and probability bucket

See [docs/BACKTESTING.md](docs/BACKTESTING.md) for complete guide.

## 📝 Configuration

Edit `CONFIG` in `core/nfl_montecarlo.py`:

```python
CONFIG = {
    "season": 2026,
    "home": "DEN",
    "away": "SEA",
    "model_weight": 0.25,  # 25% model, 75% market
    "player_last_n_games": 4,

    "legs": [
        {"type": "spread", "team": "SEA", "line": -2.5, "odds": -110},
        {"type": "total", "side": "over", "line": 40.5, "odds": -110},
        {"type": "player", "player": "Cooper Kupp", "stat": "receiving_yards",
         "side": "over", "line": 20.5, "odds": -115},
    ],
}
```

**Supported bet types:**
- `moneyline`: `{"type": "moneyline", "team": "SEA"}`
- `spread`: `{"type": "spread", "team": "SEA", "line": -2.5}`
- `total`: `{"type": "total", "side": "over", "line": 40.5}`
- `team_total`: `{"type": "team_total", "team": "DEN", "side": "under", "line": 19.5}`
- `player`: See player props section below

## 🎲 How It Works

### 1. Market Expectation
- Fetches Polymarket spread/total ladders
- Fits normal distribution to get expected margin, total, and variance
- Uses CLOB API for real-time bid/ask spreads

### 2. Model Blending
- Calculates team strength from season performance (points scored/allowed)
- Blends market data (75%) with model data (25%)
- Shrinks team averages toward league mean to avoid overfitting

### 3. Game Simulation
- Each team gets ~11 possessions (varies by game pace)
- Each drive results in: TD (7 pts), FG (3 pts), or nothing
- Offense efficiency varies game-to-game (calibrated to match market spread)
- **Result**: Realistic score distributions with key numbers

### 4. Player Props (⚠️ Limited)
- Uses last 4 games to calculate mean and variance
- Gamma distribution (right-skewed, non-negative)
- Correlated with team offensive performance (ρ=0.5 receiving, ρ=0.35 rushing)

**Limitations:**
- Small sample size (4 games)
- No injury/role change awareness
- No defensive matchup consideration
- No game script adjustment

## 🔍 Player Props - Current Limitations

**Why player props are challenging:**

1. **Small sample (4 games)**: One outlier drastically affects predictions
   - Example: Sam Darnold had a 13-yard game that lowered his average to 188.8 yds

2. **No context awareness**:
   - Doesn't know if player was benched, injured, or limited
   - No snap share or target share data
   - No defensive matchup ratings

3. **Fixed correlation**: Uses same 0.5 correlation for all receivers
   - Reality: WR1 more correlated with team success than WR3

4. **No usage modeling**: Just averages yards, doesn't model opportunity × efficiency

**Recommendation**: Focus on team bets (spreads/totals) where the model is strongest. Only use player props with caution and manual adjustments via `mean_override`.

## 📈 Development Roadmap

### Phase 1: CURRENT STATE ✅
- [x] Drive-by-drive team simulation
- [x] Polymarket API integration
- [x] ATM line scanner
- [x] Parlay analysis (SGP & MGP)
- [x] Basic player props
- [x] **Backtesting framework with walk-forward validation** ✨ NEW!

### Phase 2: SHORT-TERM (Next 2-4 Weeks)

#### Enhanced Team Modeling
- [ ] Offensive/Defensive EPA splits (better matchup predictions)
- [ ] Recent form weighting (exponential decay, not flat average)
- [ ] Team-specific home/away adjustments
- [ ] Weather integration (temperature, wind, precipitation)

#### Market Analysis Tools
- [x] **Historical tracking & calibration (hit rate by bet type)** ✅
- [x] **EV threshold optimization (backtest to find profitable thresholds)** ✅
- [ ] Line movement monitoring (detect sharp money)

### Phase 3: MEDIUM-TERM (1-3 Months)

#### Advanced Simulation
- [ ] Game script modeling (favorites play differently when ahead)
- [ ] Pace/tempo adjustments (team-specific drive counts)
- [ ] Situational stats (red zone efficiency, 3rd down %)

#### Better Player Props (If Pursued)
- [ ] Add target share, snap count data
- [ ] Defensive matchup ratings
- [ ] Usage-based modeling (opportunity × efficiency)
- [ ] Player correlation matrices

### Phase 4: LONG-TERM (3-6 Months)

#### Machine Learning
- [ ] Gradient boosting for team performance prediction
- [ ] Expected Points Added (EPA) drive modeling
- [ ] Feature engineering (rest, travel, division games)

#### Live Betting
- [ ] In-game win probability updates
- [ ] Automated +EV alerts
- [ ] Live line movement tracking

#### Portfolio Management
- [ ] Kelly Criterion bet sizing
- [ ] Correlation-aware parlay construction
- [ ] Bankroll tracking and ROI analysis

## 🗂️ Project Structure

```
nfl_montecarlo/
├── core/
│   ├── nfl_montecarlo.py       # Main simulation engine
│   ├── daily_scanner.py         # Scan all games for ATM opportunities
│   └── weekly_atm_tracker.py    # Track results over time
├── tests/
│   ├── test_player_props.py     # Test player prop simulation
│   ├── test_parlay.py           # Test parlay calculations
│   └── test_*.py                # Other test files
├── utils/
│   └── add_player_props_to_tracker.py  # Add props to tracking CSV
├── tracking/
│   └── week6_atm_tracking.csv   # Results tracking
├── requirements.txt
├── .env.example                 # Template for API credentials
└── README.md
```

## 🔑 Polymarket API

### Public APIs (No Auth Required)
- **Gamma API**: Event search, market discovery
- **CLOB API**: Real-time orderbook (best bid/ask)
- **Data API**: Historical price trends

### Authenticated Features (Optional)
Create API credentials at https://polymarket.com/settings/api

Add to `.env` file:
```
POLYMARKET_API_KEY=your-key
POLYMARKET_SECRET=your-secret
POLYMARKET_PASSPHRASE=your-passphrase
```

**Authenticated features:**
- `--show-positions`: View your open positions
- `--show-orders`: View your active orders

## 📊 Data Sources

1. **Polymarket** (market-implied probabilities)
   - Spread/total ladders → expected margin/total
   - Moneyline → win probability
   - Real-time orderbook for accurate pricing

2. **nflverse** via nflreadpy
   - Season game results (team ratings)
   - Player game logs (last N games)
   - Sportsbook lines (backup if Polymarket fails)

## 🎯 Recommended Strategy

### FOCUS ON (High Edge Potential):
1. **At-the-money spreads** (±3 to ±6.5)
2. **Game totals** near market consensus
3. **Moneylines** close to 50/50
4. **Same-game parlays** combining correlated bets

### AVOID FOR NOW (Model Limitations):
1. Player receiving/rushing/passing props
2. Anytime touchdowns (not implemented)
3. Large spreads (±14+) - less accurate at extremes

## 🧪 Testing Player Props

If you want to test player props despite limitations:

```bash
# Single prop test
python tests/test_player_props.py --game "SEA @ DEN" \
  --prop "Cooper Kupp" "receiving_yards" "over" 20.5 -110 \
  --sims 100000

# Add to tracking CSV
python utils/add_player_props_to_tracker.py \
  --csv tracking/week6_atm_tracking.csv \
  --game "SEA @ DEN" \
  --prop "Cooper Kupp" "receiving_yards" "over" 20.5 -110
```

## ⚙️ Advanced Options

```bash
# Offline mode (use nflverse sportsbook lines)
python core/nfl_montecarlo.py --offline

# Reproducible results
python core/nfl_montecarlo.py --seed 42

# Show market history
python core/nfl_montecarlo.py --show-history

# Show recent trades
python core/nfl_montecarlo.py --show-trades

# List all available markets
python core/nfl_montecarlo.py --show-polymarket-markets
```

## 📚 Key Concepts

**Expected Value (EV)**: Average profit per $1 bet over many iterations
- **+EV**: Profitable long-term (bet has edge)
- **-EV**: Unprofitable long-term (house/market has edge)
- Example: 60% hit rate at 1.90x odds = (0.60 × 1.90) - 1 = +14% EV

**Hit Rate**: Simulated probability the bet wins

**Fair Odds**: Odds where EV = 0% (break-even line)

**Multiplier**: Decimal odds (1.90x = get $1.90 per $1 bet)

**At-the-Money (ATM)**: Bets priced near 50/50 (usually best EV opportunities)

## ⚠️ Limitations & Disclaimers

1. **Small sample sizes**: 4-5 games of player data is limited
2. **No injury tracking**: Must manually adjust via `mean_override`
3. **No weather/rest**: Only factors in through market prices
4. **Market efficiency**: Polymarket is generally sharp; big +EV should be investigated
5. **Simulation ≠ Reality**: Model is a tool, not a guarantee

**Use responsibly. Only bet what you can afford to lose. This is for educational purposes.**

## 🤝 Contributing

This is an ongoing research project. Areas for contribution:
- Better player prop data sources
- Weather/rest day integration
- Historical backtesting framework
- Defensive matchup ratings
- Live betting features

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **nflverse** for comprehensive NFL data
- **Polymarket** for transparent prediction markets
- **Claude** for development assistance
