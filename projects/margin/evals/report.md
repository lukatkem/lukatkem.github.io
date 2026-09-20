# Margin — Eval Report

**Verdict: PASS** · 70 questions · median retrieval latency 30.9ms

| Metric | Score | Threshold |
|--------|-------|-----------|
| recall@5 | 100.0% | 80% |
| precision@3 | 97.1% | — |
| MRR | 0.918 | — |
| answer must-contain | 78.6% | 75% |

## By question type

| Type | n | recall@5 | answer |
|------|---|----------|--------|
| factual | 58 | 100.0% | 81.0% |
| math | 3 | 100.0% | 100.0% |
| reasoning | 9 | 100.0% | 55.6% |

## Failures

- **q004**: `What is the maximum total drawdown during the evaluation?` — gold `rules/02-evaluation-phases.md`, got ['rules/04-max-drawdown.md', 'rules/04-max-drawdown.md', 'rules/02-evaluation-phases.md']
- **q005**: `How many minimum trading days do I need in each evaluation phase?` — gold `rules/02-evaluation-phases.md`, got ['rules/02-evaluation-phases.md', 'rules/02-evaluation-phases.md', 'rules/05-consistency-rule.md']
- **q006**: `Is there a time limit to complete the evaluation phases?` — gold `rules/02-evaluation-phases.md`, got ['rules/02-evaluation-phases.md', 'rules/02-evaluation-phases.md', 'ops/22-breach-penalties.md']
- **q016**: `My best trading day made 56% of my total profit. Does that fail my account?` — gold `rules/05-consistency-rule.md`, got ['rules/05-consistency-rule.md', 'rules/02-evaluation-phases.md', '_user/1-my-notes.md']
- **q019**: `What happens if I open a trade 1 minute before a high-impact news release on my evaluation account?` — gold `rules/06-news-trading.md`, got ['rules/06-news-trading.md', 'rules/06-news-trading.md', 'ops/22-breach-penalties.md']
- **q022**: `Can I hold positions over the weekend?` — gold `rules/07-weekend-holding.md`, got ['rules/07-weekend-holding.md', 'rules/07-weekend-holding.md', 'rules/02-evaluation-phases.md']
- **q024**: `What happens if the firm detects a weekend-held position on a non-swing account?` — gold `rules/07-weekend-holding.md`, got ['rules/07-weekend-holding.md', 'rules/07-weekend-holding.md', 'ops/22-breach-penalties.md']
- **q031**: `How does the account scaling plan work?` — gold `rules/10-account-scaling.md`, got ['rules/10-account-scaling.md', 'rules/10-account-scaling.md', 'rules/02-evaluation-phases.md']
- **q032**: `How often can my funded account be scaled up?` — gold `rules/10-account-scaling.md`, got ['rules/10-account-scaling.md', 'rules/10-account-scaling.md', 'ops/23-upgrade-paths.md']
- **q055**: `What happens if I break a rule on a funded account?` — gold `ops/22-breach-penalties.md`, got ['ops/22-breach-penalties.md', 'ops/22-breach-penalties.md', 'rules/11-payouts.md']
- **q057**: `Is there an API to read my account metrics?` — gold `ops/24-data-api-policy.md`, got ['ops/24-data-api-policy.md', 'ops/24-data-api-policy.md', 'ops/16-platforms.md']
- **q059**: `What is the no-consistency badge and what does it remove?` — gold `ops/23-upgrade-paths.md`, got ['ops/23-upgrade-paths.md', 'rules/02-evaluation-phases.md', 'ops/23-upgrade-paths.md']
- **q064**: `What happens to my profit if I scale my account up?` — gold `rules/10-account-scaling.md`, got ['rules/10-account-scaling.md', 'rules/10-account-scaling.md', 'ops/22-breach-penalties.md']
- **q065**: `What is the median spread on MNQ?` — gold `ops/15-execution-spreads.md`, got ['ops/15-execution-spreads.md', 'education/18-glossary-instruments.md', 'ops/15-execution-spreads.md']
- **q069**: `Does holding a position with no activity count as a trading day?` — gold `rules/02-evaluation-phases.md`, got ['rules/02-evaluation-phases.md', 'rules/07-weekend-holding.md', 'rules/07-weekend-holding.md']
