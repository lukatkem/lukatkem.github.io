# Breach Matrix — What Happens When You Break a Rule

## Hard breaches (instant account failure)

| Violation | Consequence |
|-----------|-------------|
| Daily loss > 5% (equity basis) | account closed, no payout from that account |
| Max drawdown breach (static or trailing) | account closed |
| Prohibited practice (martingale, arb, sharing, copy-from-live) | account closed + identity flagged; confirmed fraud = permanent ban, no refunds |
| Trading from sanctioned region | permanent ban |
| Undisclosed EA | treated as account sharing |

## Soft breaches (warnings)

| Violation | Consequence |
|-----------|-------------|
| News-window trade in evaluation | trade profit removed + 1 warning; 3 warnings = failure |
| Friday 21:45 UTC weekend hold on non-swing | forced close + $50 admin fee per occurrence |
| Consistency rule exceeded | pass/payout delayed, not failed |

## After a hard breach in evaluation

- The free reset (within 14 days) restarts Phase 1 — unused even if the breach
  happened in Phase 2.
- Evaluation fee refund eligibility is lost.

## After a hard breach on funded

- Account closed at the breach point. Profit already paid out is kept.
- Current window's unpaid profit is void except where the breach was
  feed-error-confirmed by tape replay.
- One "second chance" funded re-entry (full Phase 1+2 re-run, half-price) is
  granted once per identity per year.

## Appeal path

Breach logs are immutable but replayable. Disputes go through support with
trade IDs within 24 hours; the tape replay is final (see support SLA).
