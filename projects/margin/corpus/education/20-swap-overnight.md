# Overnight Fees and Swap

## Forex and CFDs (MT5/cTrader)

- Swap is charged at 00:00 server time (NY close) on positions held through
  the rollover.
- Typical rates on the firm's feed: EURUSD long −$4.20 per lot per night, short
  +$1.10; XAUUSD long −$9.80, short +$3.20. Rates float with central-bank
  differentials.
- **Triple swap Wednesday**: positions held across Wednesday's rollover are
  charged 3× swap to cover the weekend.

## Futures

- No swap. Instead, positions carried across the **daily maintenance break**
  (17:00 ET) keep margin at overnight rates (~10× the intraday rate on micros:
  1 MNQ overnight margin ≈ $1,500 vs ≈ $150 intraday).
- Contract **roll**: quarterly (Mar/Jun/Sep/Dec for index futures). The firm
  posts the roll calendar 5 business days ahead; expiring contracts are
  force-closed at the Friday close before first notice.

## Practical notes

- Holding EURUSD longs for a week costs ≈ $29 per lot in swap at current
  rates — most swing strategies ignore this in backtests and then miss the
  0.5–1R it eats over a month.
- The 21:55–23:05 UTC rollover window is where spreads widen and where the
  daily-loss buffer guidance matters most.
