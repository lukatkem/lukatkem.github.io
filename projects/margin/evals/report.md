# Margin — Eval Report

**Verdict: PASS** · 70 questions · median retrieval latency 7.0ms

| Metric | Score | Threshold |
|--------|-------|-----------|
| recall@5 | 92.9% | 80% |
| precision@3 | 91.4% | — |
| MRR | 0.755 | — |
| answer must-contain | 78.6% | 75% |

## By question type

| Type | n | recall@5 | answer |
|------|---|----------|--------|
| factual | 58 | 91.4% | 77.6% |
| math | 3 | 100.0% | 100.0% |
| reasoning | 9 | 100.0% | 77.8% |

## Failures

- **q003**: `What is the maximum daily loss on a HydrogenProp account?` — gold `rules/02-evaluation-phases.md`, got ['rules/01-overview.md', 'rules/04-max-drawdown.md', 'rules/01-overview.md']
- **q006**: `Is there a time limit to complete the evaluation phases?` — gold `rules/02-evaluation-phases.md`, got ['rules/02-evaluation-phases.md', 'rules/02-evaluation-phases.md', 'rules/04-max-drawdown.md']
- **q009**: `On a $50,000 account, how much can I lose in one day before failing?` — gold `rules/03-daily-drawdown.md`, got ['rules/04-max-drawdown.md', 'rules/01-overview.md', 'rules/10-account-scaling.md']
- **q020**: `Is news trading allowed on funded accounts?` — gold `rules/06-news-trading.md`, got ['rules/06-news-trading.md', 'rules/08-prohibited-practices.md', 'rules/06-news-trading.md']
- **q027**: `Can I run the same trades across multiple HydrogenProp accounts I own?` — gold `rules/08-prohibited-practices.md`, got ['_user/1-my-notes.md', 'rules/08-prohibited-practices.md', 'rules/09-expert-advisors.md']
- **q028**: `Can I use an expert advisor or trading bot on my account?` — gold `rules/09-expert-advisors.md`, got ['_user/1-my-notes.md', 'rules/09-expert-advisors.md', 'rules/09-expert-advisors.md']
- **q029**: `Do I need to disclose my EA to the firm before using it?` — gold `rules/09-expert-advisors.md`, got ['_user/1-my-notes.md', 'rules/06-news-trading.md', 'rules/01-overview.md']
- **q031**: `How does the account scaling plan work?` — gold `rules/10-account-scaling.md`, got ['rules/10-account-scaling.md', 'rules/10-account-scaling.md', 'rules/01-overview.md']
- **q032**: `How often can my funded account be scaled up?` — gold `rules/10-account-scaling.md`, got ['_user/1-my-notes.md', 'rules/10-account-scaling.md', 'rules/05-consistency-rule.md']
- **q034**: `How often can I request payouts?` — gold `rules/11-payouts.md`, got ['_user/1-my-notes.md', 'rules/05-consistency-rule.md', 'rules/04-max-drawdown.md']
- **q036**: `What profit split do I start on?` — gold `rules/11-payouts.md`, got ['rules/10-account-scaling.md', '_user/1-my-notes.md', 'rules/01-overview.md']
- **q040**: `What is the fastest payout method and how fast is it?` — gold `rules/12-payout-processor.md`, got ['rules/05-consistency-rule.md', 'rules/12-payout-processor.md', 'rules/12-payout-processor.md']
- **q042**: `What KYC documents do I need before my first payout?` — gold `rules/13-kyc-aml.md`, got ['_user/1-my-notes.md', 'rules/13-kyc-aml.md', 'rules/13-kyc-aml.md']
- **q049**: `What platforms can I trade on?` — gold `ops/16-platforms.md`, got ['_user/1-my-notes.md', 'ops/22-breach-penalties.md', 'ops/22-breach-penalties.md']
- **q055**: `What happens if I break a rule on a funded account?` — gold `ops/22-breach-penalties.md`, got ['ops/22-breach-penalties.md', 'ops/22-breach-penalties.md', 'rules/01-overview.md']
- **q064**: `What happens to my profit if I scale my account up?` — gold `rules/10-account-scaling.md`, got ['_user/1-my-notes.md', 'ops/22-breach-penalties.md', 'rules/10-account-scaling.md']
