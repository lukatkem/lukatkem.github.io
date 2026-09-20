# Glossary — Instruments and Contract Specs

## CME futures on the firm

| Symbol | Name | Tick size | Tick value (micro/mini) |
|--------|------|-----------|--------------------------|
| MNQ    | Micro E-mini Nasdaq-100 | 0.25 | $0.50 |
| NQ     | E-mini Nasdaq-100       | 0.25 | $5.00 |
| MES    | Micro E-mini S&P 500    | 0.25 | $1.25 |
| ES     | E-mini S&P 500          | 0.25 | $12.50 |
| MYM    | Micro E-mini Dow        | 1    | $0.50 |
| M2K    | Micro E-mini Russell    | 0.10 | $0.50 |

Note: MNQ tick value is $0.50 (not $2 — $2 is the point value; a 20-tick stop
on 1 MNQ risks $10, and 25 micros risk $250 per full stop).

## Forex majors supported

EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD.

- Pip = 0.0001 (0.01 for JPY pairs). Standard lot pip value ≈ $10 for USD-quote
  pairs.
- Leverage 1:30 retail. Swap applies (see overnight fees).

## Metals

- XAUUSD (gold), XAGUSD (silver). CFD only.

## Margin vs leverage

- **Leverage** is the multiplier on notional you may control.
- **Margin** is the collateral required to hold a position. Initial margin for
  1 MNQ on the firm ≈ $150 intraday, $1,500 overnight.

## Index point math

ES/NQ move in 0.25-index-point ticks. A 100-point Nasdaq day is 400 ticks; on
1 MNQ that range is worth $200.
