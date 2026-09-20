# Evaluation Phases — Targets and Limits

The evaluation has two phases plus a funded stage. Both phases use the same
instrument list and the same drawdown definitions.

## Phase 1 (Verification)

| Parameter          | Value                                   |
|--------------------|------------------------------------------|
| Profit target      | 8% of starting balance                   |
| Max daily loss     | 5% of starting balance                   |
| Max total drawdown | 10% of starting balance (static)         |
| Minimum trading days | 3 days                                 |
| Time limit         | 30 calendar days (extendable free twice) |
| Consistency rule   | Best day ≤ 40% of total profit           |

## Phase 2 (Confirmation)

| Parameter          | Value                                   |
|--------------------|------------------------------------------|
| Profit target      | 5% of starting balance                   |
| Max daily loss     | 5% of starting balance                   |
| Max total drawdown | 10% of starting balance (static)         |
| Minimum trading days | 3 days                                 |
| Time limit         | 60 calendar days                         |
| Consistency rule   | Best day ≤ 40% of total profit           |

## Funded stage

- No profit target.
- Max daily loss 5%, max total drawdown 10% — both measured on the funded
  account's starting balance, and the total drawdown becomes **trailing** after
  the first payout (see max drawdown policy).
- Minimum 3 trading days per payout cycle.

## Trading days

A trading day counts only if at least 0.5% of the account balance was closed
profit or loss on that day. Holding a position with no activity does not
create a trading day.

## Failure conditions

Hitting either the max daily loss or max total drawdown at any point fails the
evaluation immediately. There is no warning and no grace tick. Breaching the
consistency rule does not fail the account — it only delays the pass until the
best-day proportion is inside 40%.
