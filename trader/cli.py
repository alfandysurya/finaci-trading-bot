"""Command line entry point.

  python -m trader.cli download            # fetch/refresh daily prices (yfinance)
  python -m trader.cli validate            # test the real portfolio against SPY, write reports/
  python -m trader.cli propose [--holdings holdings.yaml]
  python -m trader.cli daily  [--holdings holdings.yaml]   # download + propose (run after month-end close)
  python -m trader.cli walkforward         # research only: compares indicators, not used for trading

Add --synthetic to any command to run on generated data (offline smoke test).
"""
from __future__ import annotations

import argparse
import json
import sys

from . import proposals as prop
from . import simulate as sim
from . import walkforward as wfm
from .config import ROOT, cost_per_side, load_config
from .data import get_prices, synthetic_prices


def _prices(cfg, args, refresh=False, extra=None):
    if args.synthetic:
        tickers = cfg["universe"]["tickers"] + [cfg["universe"]["benchmark"]] + (extra or [])
        return synthetic_prices(list(dict.fromkeys(tickers)))
    return get_prices(cfg, refresh=refresh, extra=extra)


def cmd_download(cfg, args):
    prices = _prices(cfg, args, refresh=True)
    last = max(df.index[-1] for df in prices.values())
    print(f"[download] {len(prices)} tickers, last bar {last.date()}")


def cmd_walkforward(cfg, args):
    prices = _prices(cfg, args)
    cost = cost_per_side(cfg)
    print(f"[walkforward] cost per side {cost:.3%}, testing indicator configs...")
    res = wfm.run_walkforward(prices, cfg, cost)
    out = ROOT / "reports"
    wfm.save(res, out)
    print(res.oos_metrics[["cagr", "sharpe", "max_drawdown"]].round(3).to_string())
    for n in res.notes:
        print("  -", n)
    print(f"[walkforward] live pick: {res.final_selection['config']} -> reports/selection.json")


def cmd_validate(cfg, args):
    if args.synthetic:
        sys.exit("validate needs real prices: it decides whether real money is allowed.")
    prices = _prices(cfg, args, extra=cfg["universe"]["bias_check_tickers"])
    res = sim.validate(prices, cfg, cost_per_side(cfg), ROOT / "reports")
    print((ROOT / "reports" / "validation_report.md").read_text(encoding="utf-8"))
    print(f"\n[validate] go_live = {res['go_live']}")


def _approved() -> bool:
    path = ROOT / "reports" / "validation.json"
    return path.exists() and json.loads(path.read_text(encoding="utf-8"))["go_live"] is True


def cmd_propose(cfg, args, refresh=False):
    prices = _prices(cfg, args, refresh=refresh)
    holdings = prop.load_holdings(args.holdings)
    p = prop.build(prices, cfg["strategy"], cfg, holdings, go_live=_approved() and not args.synthetic)
    j, m = prop.save(p, ROOT / "proposals")
    print(prop.to_markdown(p))
    print(f"\n[propose] wrote {j.relative_to(ROOT)} and {m.relative_to(ROOT)}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trader")
    ap.add_argument("command", choices=["download", "validate", "walkforward", "propose", "daily"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--holdings", default=None, help="YAML with cash_usd and positions (shares)")
    ap.add_argument("--synthetic", action="store_true", help="use generated prices (offline test)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "download":
        cmd_download(cfg, args)
    elif args.command == "validate":
        cmd_validate(cfg, args)
    elif args.command == "walkforward":
        cmd_walkforward(cfg, args)
    elif args.command == "propose":
        cmd_propose(cfg, args)
    elif args.command == "daily":
        cmd_propose(cfg, args, refresh=True)


if __name__ == "__main__":
    main()
