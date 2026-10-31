# Project tracker

Temporary board, because Notion's free block limit blocked the database. Each stage must pass before the next starts.

## Agreed decisions
- Goal: 5-10% net a month is a dashboard target. It never increases risk.
- Risk: 1% per trade. Pause new buys at -20% from peak.
- Limits: 20% per stock, about 5% total open risk, $30 minimum order.
- Holding period: 2-8 weeks, adjustable.
- Data: free sources only.
- Orders: paper first. No real orders until validation passes.
- Records: the bot record decides the pass. Your tapped trades are tracked separately.
- Scoring: rule-based, with adjustable weights of 40/40/20.
- Hosting: Claude scheduled tasks for the prototype, then a VPS if it is good enough.

## Stage 0: Design
### Next Up
- [ ] Approve scanner and scoring design
- [ ] Approve risk, stop-loss and sell-alert design
- [ ] Approve Telegram flow and paper record design
- [ ] Review the written spec
- [ ] Review the implementation plan
- [ ] Ask Pluang Care whether a Telegram-approved Claude agent is allowed

## Stage 1: Build check
### Backlog
- [ ] Load the Pluang stock list and apply liquidity filters
- [ ] Build technical, fundamental, sentiment and risk scores
- [ ] Choose and test a free news source
- [ ] Build position sizing and portfolio limits
- [ ] Build Telegram picks, paper buy, sell alerts and `/settings`
- [ ] Build bot and user trade records
- [ ] Build the performance report against SPY and the 5-10% target
- [ ] Add failure alerts for missing data or missed runs
- [ ] Exit rule: alerts, fees and records work correctly

## Stage 2: Paper validation
### Backlog
- [ ] Run live with no real orders
- [ ] Review results monthly
- [ ] Exit rule: at least 6 months and 30 closed trades, net ahead of SPY

## Stage 3: Small live test
### Backlog
- [ ] Update the Pluang prompt to follow the connector rules
- [ ] Place real trades manually with a small amount
- [ ] Compare real fills with paper results
- [ ] Exit rule: about 3 months with fills close to paper results

## Stage 4: Normal use
### Backlog
- [ ] Move to the full chosen amount
- [ ] Decide whether to build the Telegram-to-Claude link, if Pluang allows it
