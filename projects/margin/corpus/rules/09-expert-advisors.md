# Expert Advisor (EA) and Automation Policy

Automation is allowed under strict conditions. This is the firm's most
misunderstood policy.

## Allowed

- **EAs you wrote yourself** on your own accounts — including strategy
  automation of a manual system.
- **EAs you commissioned** — allowed if exclusive to you (no public resale) and
  disclosed at compliance@hydrogenprop.example with the source code.
- **Trade copiers between your own firm accounts** (max 5 accounts, internal
  copier only).
- Risk-management bots: auto-daily-stop, auto-flatten before news/weekend,
  trailing equity protectors.

## Prohibited

- Off-the-shelf EAs from marketplaces (MQL5 market, cTrader cBots store).
- Arbitrage EAs of any kind — latency, triangular, feed mismatch.
- HFT — defined as average holding time under 800ms OR more than 50 orders per
  minute sustained over 10 minutes.
- EAs that trade external accounts and mirror into the firm (copy-trading
  from live brokers is a hard fail, see prohibited practices).

## Disclosure requirement

Every EA must be registered in the dashboard before its first live run:

1. Dashboard → Compliance → Register EA.
2. Upload source (zip) or a hash of the compiled binary.
3. Wait for the green "Approved" badge (usually under 48 hours).

Undisclosed EA trading is treated as account sharing — hard fail, no payout.

## VPS policy

- Firm-provided VPS: free for funded accounts ≥ $100k, otherwise $15/month.
- Own VPS allowed; datacenter IPs are fine, residential proxies are not.
