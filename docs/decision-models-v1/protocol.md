# OpenAI Decisions and Clef on the medical examination cohorts

Frozen on 7 October 2026 before any manifested development or test request. Subsequent exploratory extension. Jev, comparator and Luna letter-probability results were known when this extension was proposed.

## Question

Do decision models other than Jev return option probabilities that rank their own errors and support escalation on knowledge-intensive licensing questions? The results inform whether the medical manuscript should evaluate three decision models rather than one. That scope decision is made after the results, and the extension is reported as exploratory whichever way it is made.

## Models

- OpenAI Decisions API, `POST /v1/decisions`, model `gpt-6-luna`, the only model offered on 7 October 2026 (public beta from 6 October). The endpoint is a specialised interface to the same base model as the GPT-6 Luna comparator and letter-probability baseline. It has no reasoning, temperature, seed or output-allowance settings.
- Cloudflare Clef, Workers AI model `@cf/cloudflare/clef`, open-weight 27B decision model (public revision `2f3de3dd85f379784083b0814d997ab627200f0c`). No generation settings apply. The hosted deployment's weights are not independently verified.

## Feasibility probe

Eight requests on two Korean and two English development questions per model (`runs/decision-models-probe-v1`) were all valid. Decisions returned probabilities with two decimals, and three of its four selected probabilities were exactly 1. Clef returned four-decimal probabilities, none saturated. Both reported input tokens and no output tokens. No test question was sent.

## Design

- Questions. KorMedMCQA doctor development (164) and test (435), and MedQA development (1,272) and test (1,273), from the frozen study manifests. For each model, all development requests finish before any test request.
- Request. Jev's state `{"exam": {"question": ...}}`, the cohort's Jev instruction, and option letters with option texts in the published order. Clef receives Jev's request with only the model selector changed. Decisions receives the serialized state as its `input` string and one `choice` question named `answer` whose choices carry the letters and texts. No translation, retrieval or prompt change.
- Probability. The selected-option probability is the returned probability of the returned choice after normalization to sum to one, the rule used for Jev. The native `confidence` field is kept separately and is not the analysis variable.
- Validity. A response is invalid when the HTTP request fails terminally, the reported model differs, there is not exactly one choice answer, the probability vector does not cover exactly the offered options, any probability is outside zero to one, the sum differs from one by more than the rounding bound (number of options times 0.005 for Decisions and 0.00005 for Clef), or the choice is below the reported maximum. Invalid answers count as incorrect on the full denominator. Probability analyses use valid responses.
- Retries. At most two retries for transient HTTP failures (408, 409, 429, 5xx) or connection errors. Other HTTP errors are terminal failures.
- Budget ceiling. US$2.50 per model at list input prices of US$0.10 (Decisions) and US$0.24 (Clef) per million tokens. Neither endpoint bills output.

## Analysis

Identical definitions to the Jev and Luna letter-probability analyses, applied to each model separately.

- Accuracy against the official key with Wilson intervals, on the full planned denominator.
- Error-detection AUROC on each model's own answers, one minus the selected-option probability, with 4,000 source-question bootstrap resamples. Differences between models are descriptive because they make different errors.
- Calibration as secondary outcomes (fixed-bin expected calibration error, Brier score, log loss), the share of selected probabilities equal to one and the number of distinct selected values.
- Error capture by referral share. Share of each model's errors among its least confident 10, 20, 30 and 50% of answers, with ties handled by expected value under random order.
- Escalation. The referral threshold is the twentieth percentile of the model's own development selected probabilities. Questions at or below it take the GPT-6.1 Sol answer from the current panel. Accuracy, rescued errors, lost correct answers and the gain over random referral of the same expected size are reported, together with the full escalation curve to Sol using the shared tie rule.
- Costs are list-price input-token equivalents per 1,000 questions and are secondary.

These answers are a new collection and are reported as their own measurements.
