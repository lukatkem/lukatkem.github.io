# Security policy — how secrets are kept out of this repo

1. **Provider keys never touch version control.** All provider API keys
   (OpenRouter, NVIDIA, anything else) live in a git-ignored `.env` file on
   the development machine only, with `0600` permissions. Every repo pass
   greps the staged tree for real key material before anything is pushed.
2. **Derived tools are designed so keys can't leak by accident:**
   - the LLM gateway reads upstream keys from env-var *names* at call time
     and never logs or echoes them; test suites run against
     `httpx.MockTransport` fakes, never real providers;
   - the copilot's API keys are stored **hashed** (SHA-256 of the raw key) —
     a stolen database yields no usable keys;
   - sessions are `httpOnly` cookies; passwords are scrypt-hashed.
3. **Scrubbing at the data layer.** The Refinery redacts emails, phone
   numbers, card/ID digit runs, API keys and JWTs (`[REDACTED]`) — so
   training corpora built with it don't bake credentials into model weights.
4. **CI contains no credentials.** Workflows install public packages and run
   tests against fakes only.

## Reporting

Found a leak or a vulnerability? Open an issue — or better, a fix.
