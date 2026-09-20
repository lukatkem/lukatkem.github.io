# Execution Quality — Spreads, Commissions, Slippage

## Commissions

| Instrument | Commission | Round-turn |
|------------|------------|------------|
| CME futures micros (MNQ, MES, M2K, MYM) | $0.62 per side | $1.24 |
| CME minis (ES, NQ, RTY, YM) | $1.24 per side | $2.48 |
| Forex majors | $3.00 per lot | $6.00 |
| XAU/USD | $3.00 per lot | $6.00 |

## Typical spreads (aggregated feed)

- MNQ: 1 tick (0.25 index points), occasional 2 ticks 21:55–23:05 UTC.
- ES: 1 tick (0.25).
- EURUSD: 0.1–0.3 pips.
- XAUUSD: 12–20 cents.

## Slippage behavior

- Limit orders: filled at limit or better; no positive slippage is removed.
- Market orders: median slippage 0 ticks on futures during RTH, 1 tick in the
  rollover window.
- Stops become market orders once triggered — gap-through fills are possible
  and are the trader's risk (this is why the daily-loss buffer guidance
  exists).

## Feed and latency

- Price feed: aggregated CME data, co-located match engine simulation;
  median round-trip 12ms from the firm VPS region (NY4).
- No requotes. Slippage is applied honestly; positive slippage is kept by the
  trader.
- Feed delay complaints must be filed within 24 hours with the trade ID; the
  firm compares against the master tape and adjusts trades if feed-side error
  is confirmed.
