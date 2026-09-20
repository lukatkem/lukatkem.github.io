# Maximum Drawdown — Static vs Trailing

Maximum total drawdown is **10% of the starting balance** during evaluation
phases (static) and converts to a **trailing drawdown** on funded accounts
after the first payout.

## Static drawdown (evaluation)

- Fixed at start: a $100,000 evaluation breaches at $90,000 equity, no matter
  how much profit is made.
- Floating equity counts. Equity below the threshold at any tick fails the
  account.

## Trailing drawdown (funded, post-payout)

- The high-water mark is the greater of starting balance and the highest
  **end-of-day balance**.
- After the first payout the drawdown level freezes 6% below the high-water
  mark? No — it stays 10% below the highest end-of-day balance, but never rises
  above the starting balance + 6%.
- Intraday peaks do NOT ratchet the trailing level; only end-of-day balances
  count. This is an end-of-day trailing drawdown, not a tick-based one.

### Worked example — $100,000 funded

1. Day 3 closes at $104,000 → trail level = $93,600 (10% below EOD high).
2. Day 7 closes at $108,000 → trail level = $97,200.
3. Payout requested and paid at $110,000 → trail level caps: it can never be
   higher than $106,000 (starting balance + 6%).
4. Balance later drops to $105,000 → trail stays at $106,000 cap; a further
   $5,000 loss would breach.

## Buffer rule at payout

Before a payout is processed, equity must stay above the trailing level with a
$100 buffer. A payout request while equity is within $100 of the trail is
rejected and retried next cycle.

## Why this matters

The trailing mechanism means profit taken out of the account (payouts) cannot
be re-lost. The worst case for a funded trader who has reached the cap is
losing 4% of the original balance before breach.
