# Data, Reporting and API Policy

## Trader-facing reports

- Dashboard shows: equity curve, daily loss usage, drawdown distance, ratio of
  best day to total profit (consistency meter), commission totals.
- Statements exportable as CSV/PDF per cycle.

## Programmatic access (this product)

- The firm exposes a **read-only data API** for account metrics: balances,
  drawdown usage, open-position risk, payout status. Base URL
  `https://api.hydrogenprop.example/v1`, bearer token from the dashboard.
- Rate limit: 60 requests/minute per token. 429 responses include
  `Retry-After`.
- Write operations (placing orders) are NOT exposed through this API —
  execution stays inside the supported platforms.

## What third-party tools may do

- Read statements and metrics (journaling, risk analytics, tax reports).
- Compute risk dashboards from the metrics API.

## What they may never do

- Place, modify, or cancel orders (any tool doing so violates the automation
  policy and voids the account).
- Resell firm data or redistribute the feed.
