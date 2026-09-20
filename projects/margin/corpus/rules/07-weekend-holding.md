# Weekend and Overnight Holding

Holding rules depend on account type.

## Evaluation accounts

- **No weekend holding.** All positions must be closed by Friday 21:45 UTC.
- Overnight holding within the week is allowed for futures and forex (swap
  applies on forex, see overnight fees doc).

## Funded standard accounts

- Same as evaluation: flat by Friday 21:45 UTC.
- Overnight holding Monday–Friday allowed.

## Funded swing accounts

- Weekend holding allowed on futures only, with reduced leverage.
- Gap risk is fully the trader's: a weekend gap that breaches the trailing
  drawdown still fails the account. There is no gap waiver.
- Swing accounts cost an additional $25/month add-on and are only available at
  $50k size and above.

## Rollover windows

- Daily maintenance 21:55–23:05 UTC: spreads widen, liquidity thins.
- Futures contract roll: the firm announces the roll calendar 5 business days
  before expiry; positions in the expiring contract are force-closed at the
  session close before first notice date if not rolled manually.

## Force-closure

If the firm detects a weekend-held position on a non-swing account, it is
closed at the Friday close price with a $50 administrative fee per occurrence.
