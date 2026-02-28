# GuapBot

Production-grade algorithmic cryptocurrency trading system.

**Strategies**
- `money_printer` — USD-base, optimise for dollar returns
- `sat_stacker` — BTC-base, optimise for Bitcoin accumulation

**Exchange:** Kraken only | **Assets:** XBTUSD, ETHUSD | **Signal-only:** ETHBTC

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in your keys
```

## Run Tests

```bash
.venv/bin/pytest tests/unit -v
# or: ./run_tests.sh
```

## Architecture

7 layers, each talking only to the layer directly above/below:

1. **Data Acquisition** — Kraken REST/WS, Reddit, Telegram, RSS, Stocktwits
2. **Feature Construction** — ~100 technical indicators, FinBERT, LLM context
3. **Regime Detection** — HMM per timeframe (1h/4h/daily), vector output
4. **Model Layer** — RL, LSTM, TrendFollowing, MeanReversion, GradientBoost
5. **Ensemble Meta-Learner** — LightGBM, online daily updates
6. **Portfolio Layer** — dual-strategy management, kill switches
7. **Execution** — position sizing, order management, paper/live modes

## Build Sessions

See `guapbot_handoff.docx` for the complete 14-session build spec.
Current: **Session 2 complete** (foundation + data layer).
