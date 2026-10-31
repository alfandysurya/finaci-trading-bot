# Pluang semi-auto trader

A paper-first US stock assistant. Python scores stocks and proposes trades, and you decide what
to do. Nothing in this repo can place a Pluang order.

## Project status

| Part | Status |
|---|---|
| Fixed-rule proposals, `validate` gate, fees and data safety checks | **Built**: 17 tests pass |
| Current strategy | **Not approved for real money** (`PAPER ONLY`) |
| Stock scanner and technical, fundamental, sentiment and risk scores | Designed in part, not built |
| Telegram bot, paper records and performance reports | Not built |
| Claude scheduled task | Planned for the prototype, not set up |

## Product plan

The planned bot posts the top 3 underrated US stocks to a Telegram group, with scores and the reason
for each. It suggests a position size and stop-loss, follows open picks, sends sell alerts, and
records wins and losses.

Agreed defaults, all adjustable within safety limits:

| Setting | Default |
|---|---|
| Score weights | 40% fundamental, 40% technical, 20% sentiment (rule-based) |
| Risk per trade | 1% of the account |
| Holding period | 2-8 weeks |
| Position limits | 20% per stock, about 5% total open risk, $30 minimum order |
| Drawdown pause | No new buys at -20% from peak |

The 5-10% net monthly goal is shown as a target only. It never increases risk. Tested results so far
are about 0.9-1.8% a month.

Changing score weights or holding period restarts the paper proof.

### Release stages

1. **Build check:** the scanner, alerts, fees and records work correctly.
2. **Paper validation:** at least 6 months and 30 closed trades, net ahead of SPY. Free data can't fairly
   backtest smaller stocks, so the live paper record decides this stage.
3. **Small live test:** about 3 months of manual trades with fills close to paper results.
4. **Normal use:** the full chosen amount.

The bot record, which includes every eligible pick automatically, decides the pass. Your own tapped
trades are tracked separately.

### Hosting

The prototype will use Claude scheduled tasks, which run the Python commands from a copy of this repo.
The official records need storage outside the temporary run, and the task's network access must allow
the data and Telegram sites. If the prototype is good enough, the bot moves to a VPS.

### Telegram setup

Copy `.env.example` to `.env`, then fill in the bot token, group chat ID and your user ID. Never commit
`.env`. Only the owner user ID may use trade buttons or settings.

See [the project tracker](docs/PROJECT_TRACKER.md) for current tasks.

## Setup

```bash
uv sync                                  # creates .venv and installs locked dependencies
uv run pytest -q                         # 17 tests, runs offline
```

Edit `config.yaml` first. The settings to check:

- `universe.tickers`: keep only tickers you've checked are tradable on Pluang.
- `portfolio.equity_usd`: the amount you'll allocate.
- `costs.fee_rate`: use `0.0020` if you're a Pluang Plus member.

## Workflow

The strategy is one fixed rule in `config.yaml`: SMA(50) above SMA(200), checked at the last
close of each month. Portfolio rules are the same in validation and in live proposals.

**1. Validate the portfolio** (after changing the universe, costs or portfolio settings):

```bash
uv run python -m trader.cli validate            # writes reports/validation_report.md + validation.json
```

Real money requires a Sharpe above SPY, after costs, on both the trading universe and the
2009 large-cap bias check. If it fails, proposals are marked `PAPER ONLY` and the execution
prompt places no orders. Don't change the rule just to pass: that's overfitting.

**2. Monthly proposals** (once, after the last US close of the month):

```bash
uv run python -m trader.cli daily --holdings holdings.yaml   # holdings file optional
```

This writes `proposals/proposals_YYYY-MM-DD.json` (for Claude) and `.md` (for you).

**3. Execute during US market hours.** The Pluang agent can't execute US stock orders while the
exchange is closed, so run this after 21:30 WIB (20:30 during US daylight time). Open a chat with
the Pluang connector, attach the JSON file and paste `prompts/pluang_execute.md`. Or place the
orders yourself in the Pluang app from the `.md` file.

`walkforward` still works for research, but trading no longer depends on it.

Add `--synthetic` to any command to run on generated data without internet.

## Design choices

| Choice | Why |
|---|---|
| Daily bars, signal at close, fill at next open | Matches a confirm-each-trade workflow. There's no look-ahead (tested). |
| Long-only, no leverage | Pluang leverage costs 7.5% p.a. and dividend tax doubles to 30%. |
| Costs ≈ 0.44% per side (0.30% fee + 0.05% JFX/KBI + 11% VAT + 5 bps slippage) | Pluang's fee schedule. The JFX/KBI cap only binds on orders above $200. |
| One fixed rule, checked monthly | Walk-forward selection never beat fixed SMA(50,200) on Sharpe, and monthly checks cut orders by about half. |
| Validate the real portfolio against SPY | The per-ticker walk-forward report overstated drawdown protection compared with the top-8 portfolio. |
| Inverse-vol sizing, 15% cap, 5% cash buffer, max 8 names | Keeps a single bad name from sinking the account. |

## Known limitations

- **Survivorship bias:** the default universe is today's winners, and even the bias-check list contains only survivors, so backtest returns are still optimistic.
- **Data quality:** yfinance is free but unofficial. Swap to Massive (Polygon) or Alpaca data via `data.source: csv` if it breaks.

## Roadmap

1. **Fundamentals filter:** quality and value scores from SEC EDGAR, keyed by filing date (point-in-time), used to narrow the universe.
2. **News sentiment:** daily per-ticker headline sentiment (FinBERT or Claude API) as a veto or tilt. News text gets treated as untrusted data.
3. **Model stage:** a gradient-boosting model on technical + fundamental + sentiment features predicting forward returns, validated walk-forward, with PSI drift monitoring.
4. **Scheduling:** run `daily` automatically after US close and push the proposal summary to your phone.

_Not financial advice. Paper-check every proposal for a few weeks before committing real money._
