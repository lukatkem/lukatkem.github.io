# Prohibited Practices

These practices fail the account on first detection. Detection is automated
plus manual review of every payout request.

## Hard-prohibited (account failure)

1. **Martingale and grid systems** — position size increasing after losses on
   the same instrument/day.
2. **Latency arbitrage** — exploiting feed delay between the firm feed and
   external prices.
3. **Toxic order flow** — TOB/fill-price games, hedging between the evaluation
   account and any external account.
4. **Copy trading from external live accounts** — mirror-trading your own or
   others' live broker accounts into the evaluation.
5. **Group trading / account sharing** — two humans trading one account, or
   one trader running accounts for others. Account sharing includes shared
   VPS credentials.
6. **IP/proxy manipulation** — residential-proxy rotation to fake location,
   trading from sanctioned countries.
7. **Exploiting pricing errors** — doubling size on stale quotes, gold-silver
   ratio gaps from bad ticks.

## Restricted (warning then failure)

- **Hedged pairs on the same account** — allowed up to 10% of equity notional
  exposure, for scalping management only. Persistent hedge parking is flagged.
- **Lot-size consistency** — if your winning trades are systematically 1/20th
  the size of losing trades, the risk team reviews the account (reverse
  martingale pattern).

## What is allowed

- Manual trading in any style (scalp, swing, news on funded).
- Your own EAs within the EA policy doc.
- Copy trading BETWEEN your own HydrogenProp accounts via the firm's internal
  copier (max 5 accounts).
- Journaling/analytics tools that read statements (read-only API access).

## Detection notes

Trades are analyzed for tick-level consistency with the aggregated feed. The
firm does not publish detection thresholds — specifically so they cannot be
engineered around.
