# Answer-letter probability baseline with GPT-6 Luna

Frozen on 6 October 2026 before any development or test request. Subsequent exploratory extension.

## Question

Do Jev's native option probabilities rank errors and support escalation better than answer-letter probabilities read from a low-cost generative model?

## Feasibility probe

Five requests on Korean development questions (`runs/luna-logprob-probe-v1`) confirmed that the Responses API returns token log probabilities with up to 20 alternatives for `gpt-6-luna` without reasoning, including under the strict answer schema. No test question was sent during the probe.

## Design

- Questions. KorMedMCQA doctor development (164) and test (435), and MedQA development (1,272) and test (1,273), from the frozen study manifests. All development requests finish before any test request.
- Request. The exact current-panel GPT-6 Luna none request, meaning the same instruction, input layout, strict single-answer schema, reasoning effort `none` and 128-token allowance, with `include: ["message.output_text.logprobs"]` and `top_logprobs: 20` added. Temperature is left at its default, as for the panel.
- Probability. At the generated answer token, alternatives that are valid option letters are kept and renormalized. Letters absent from the returned alternatives receive zero. The selected-option probability is the renormalized probability of the generated letter. Whether it is also the most probable letter is recorded.
- Validity. A response is invalid when it is incomplete, does not contain one valid answer, has no locatable answer token, or has less than 0.5 total probability on valid letters. Invalid answers count as incorrect on the full denominator. Probability analyses use valid responses.
- Retries. At most two retries for transient HTTP failures or connection errors.
- Budget ceiling. US$5 at US$0.10 input and US$0.50 output per million tokens.

## Analysis

- Accuracy against the official key with Wilson intervals, on the full planned denominator.
- Error-detection AUROC on each model's own answers, using one minus the selected-option probability, with 4,000 source-question bootstrap resamples. Because Luna and Jev make different errors, their AUROC difference is descriptive.
- Calibration as secondary outcomes, using the same Brier, log loss and fixed-bin expected calibration error definitions as for Jev.
- Escalation. The referral threshold is the twentieth percentile of this run's development selected probabilities, the same rule used for Jev. Escalation curves to the current-panel GPT-6.1 Sol answers use the same tie handling as for Jev. The primary comparison is accuracy at matched total token-price expenditure between a Luna-first and a Jev-first cascade.
- These answers are a new collection on 6 October 2026 and are reported as their own measurement. They are not merged with the 30 September Luna none answers.
