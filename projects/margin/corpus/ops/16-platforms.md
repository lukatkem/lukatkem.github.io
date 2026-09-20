# Platforms and Data

## Supported platforms

| Platform | Evaluation | Funded | Notes |
|----------|------------|--------|-------|
| MT5      | yes        | yes    | hedging mode, EAs supported |
| cTrader  | yes        | yes    | cBots supported |
| TradingView | yes     | no     | charting + execution via bridge, no EAs |
| DXtrade  | futures    | futures | web + desktop |

- One platform choice per account at creation; switching costs one reset.
- Chart data is the same feed as execution data — no broker-vs-chart mismatch.

## Simulated environment note

All accounts run on simulated liquidity that mirrors the aggregated real feed.
This means: no real market impact from your size, guaranteed fill honesty per
the slippage doc, and exchange rules (limit-up/limit-down bands) simulated on
futures.

## Third-party tools

- Allowed (read-only): Tradezella, Edgewonk, LuxAlgo indicators, journaling
  apps with statement upload.
- Not allowed: tools that place orders (any bridge other than the firm's
  TradingView bridge), trade copiers pointing at external brokers.

## VPS

- NY4 region recommended for futures (12ms median).
- Firm VPS: $15/month, free at $100k funded and above (see EA policy doc).
