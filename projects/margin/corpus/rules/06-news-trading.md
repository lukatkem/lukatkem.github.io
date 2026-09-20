# News Trading Policy

News trading rules differ between evaluation accounts and funded accounts.

## Evaluation accounts (Phase 1 and 2)

- No opening or closing trades within **±2 minutes** of high-impact (red)
  economic news on the affected instrument.
- Affected instruments: the currency of the release, indices of that currency,
  and correlated instruments at the firm's discretion (e.g., US CPI affects
  ES, NQ, USD pairs, XAU).
- The restricted calendar is the firm-provided feed (ForexFactory red-tier
  mirror). A trade held THROUGH the news window is allowed; only transactions
  inside the window are violations.

## Funded accounts

- News trading is allowed on all instruments, including the release minute.
- However, slippage beyond the protection band is the trader's risk (below).

## Slippage protection band

- Limit orders get up to 2 ticks of negative slippage protection on red news.
- Market orders during the release window have no protection.
- Guaranteed-stop requests are not honored during releases.

## Violations

- Evaluation news-window violation: the violating trade's profit is removed and
  a warning is issued. Three warnings fail the account.
- Positions opened 2 minutes 1 second before a release are compliant; 1 minute
  59 seconds is not. Timestamps come from the platform server, not the trader's
  machine clock.

## Practical checklist

1. Check the firm calendar before the session, not your local timezone chart.
2. Flatten or hedge-neutralize 3 minutes before red news if in evaluation.
3. If you intend to trade the release, do it on a funded account.
