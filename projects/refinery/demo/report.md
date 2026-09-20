# Refinery report

**Inputs:** 100 documents from 1 source(s)  
**Kept:** 53 (53%) · cleaned in 0.03s

## Funnel

| stage | surviving |
|---|---|
| language+quality | 81 | (81% of previous)|
| pii+language+quality | 81 | (100% of previous)|
| after-exact-dedup | 60 | (74% of previous)|
| after-near-dedup | 53 | (88% of previous)|

## Why documents were dropped

| stage | dropped |
|---|---|
| exact-dedup | 21 |
| language | 10 |
| quality | 9 |
| near-dedup | 7 |

## Scrubbed personal data / secrets

- **email**: 4 occurrence(s) replaced with `[REDACTED]`
- **phone**: 4 occurrence(s) replaced with `[REDACTED]`
- **secret**: 4 occurrence(s) replaced with `[REDACTED]`

## Example drops (up to 8)

- `language` · doc doc073.txt: “!!! $$$ ### %%% &&& !!! !!! $$$ ### %%% &&& !!! !!! $$$ ### %%% &&& !!! !!! $$$ ### %%% &&…” — not-en (score 0.00)
- `language` · doc doc074.txt: “!!! $$$ ### %%% &&& !!! !!! $$$ ### %%% &&& !!! !!! $$$ ### %%% &&& !!! !!! $$$ ### %%% &&…” — not-en (score 0.00)
- `language` · doc doc075.txt: “!!! $$$ ### %%% &&& !!! !!! $$$ ### %%% &&& !!! !!! $$$ ### %%% &&& !!! !!! $$$ ### %%% &&…” — not-en (score 0.00)
- `language` · doc doc076.txt: “!!! $$$ ### %%% &&& !!! !!! $$$ ### %%% &&& !!! !!! $$$ ### %%% &&& !!! !!! $$$ ### %%% &&…” — not-en (score 0.00)
- `quality` · doc doc077.txt: “Buy our product now, limited offer, click the link below!!! Buy our product now, limited o…” — repeated-lines (one line ×15)
- `quality` · doc doc078.txt: “Buy our product now, limited offer, click the link below!!! Buy our product now, limited o…” — repeated-lines (one line ×15)
- `quality` · doc doc079.txt: “Buy our product now, limited offer, click the link below!!! Buy our product now, limited o…” — repeated-lines (one line ×15)
- `quality` · doc doc080.txt: “Buy our product now, limited offer, click the link below!!! Buy our product now, limited o…” — repeated-lines (one line ×15)
