# Risk Management Playbook (Education)

The evaluation math rewards survivability over win rate. This playbook is the
one the firm's top decile of passing traders converges on.

## The 1% framework

- Risk per trade: **0.5–1% of account**. On a $100k account that is $500–1,000.
- Max concurrent open risk: 2% (two independent ideas, or one with stops
  scaled out).
- Daily personal stop: −2% (leaves 3% of the firm's 5% daily limit as
  spread/rollover buffer).

## Position sizing math

Futures (micros): risk $ = stop ticks × $2 per tick (MNQ/MES) × contracts.

- Risk $1,000 with a 20-tick stop on MNQ: 1,000 ÷ (20 × 2) = 25 micros.
- Forex: risk $ = stop pips × pip value × lots. $1,000 risk, 25-pip stop,
  $10/pip → 4 lots.

## R-multiples and expectancy

- Define stop at 1R. Track every trade in R, not dollars.
- Expectancy = (win% × avgWin_R) − (loss% × avgLoss_R). Above 0.2R with 50+
  trades is a real edge.
- The consistency rule (best day ≤ 40%) is easiest to satisfy with fixed-R
  sizing because day results stay proportional.

## Why traders actually fail

1. Oversizing after wins (revenge by confidence, not just revenge after
   losses).
2. Moving stops on the losing trade — converts 1R losses into 3R losses.
3. Trading through red news in evaluation (see news policy).
4. No daily stop — the 5% daily limit exists, but hitting it is always a
   session-management failure first.

## Checklist per session

- [ ] Max risk per trade set in platform (not in your head)
- [ ] Daily stop order in the journaling bot
- [ ] Red-news times for traded instruments noted
- [ ] Rollover window (21:55–23:05 UTC) avoided for entries
