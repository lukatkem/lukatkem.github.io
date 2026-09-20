# Daily Drawdown — Definition and Examples

The daily loss limit is **5% of the starting balance**, calculated on **equity,
not balance**. It includes floating (open) PnL and commission.

## Reset time

The daily loss counter resets at **00:00 UTC** (server midnight). Positions held
overnight keep counting against the new day's limit once the clock rolls over.

## Worked examples

### Example 1 — $50,000 account

- Starting balance: $50,000. Daily limit: $2,500.
- Trader opens a position, it floats to -$1,800, trader closes at -$1,800.
- Equity is now $48,200. Daily loss used: $1,800 of $2,500.
- Trader may still risk $700 of floating loss on the same day.

### Example 2 — floating breach

- Balance $50,000, trader opens 5 MNQ, position floats to -$2,600 intraday.
- Even without closing, equity -$2,600 breaches the $2,500 daily limit.
- The account fails at the moment equity crosses the limit, not at close.

### Example 3 — overnight carry

- Funded $100,000 account. Tuesday close: balance $101,000.
- Wednesday 00:00 UTC the daily limit is again $5,000 below equity at rollover
  snapshot (equity basis). A gap down of $5,100 in a held position fails the
  account at the open even though the trader placed no new trades.

## Buffer guidance

The firm recommends keeping floating risk at or below 2% of the account so that
a normal spread widening at rollover (roughly 21:55–23:05 UTC) cannot trigger a
daily breach.

## Common misconceptions

- Realized profit earlier in the day does NOT extend the daily loss allowance;
  the limit is measured from the highest equity OR balance point reached during
  the day, whichever produces the tighter constraint.
- Weekend gaps on swing accounts count against the daily loss of the settlement
  day the gap lands in.
