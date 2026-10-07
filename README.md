# Referring uncertain answers by decision-model confidence

Code and data for the preprint *Referring uncertain answers by decision-model confidence: Jev, OpenAI Decisions and Clef on Korean and US medical licensing examination questions* (Sangzin Ahn, 2026).

Three decision models answered all 435 KorMedMCQA doctor test questions and all 1,273 MedQA test questions. The probability of the selected option was used to refer each model's least confident 20% of answers to GPT-6.1 Sol, and the result was compared with random referral and an oracle.

## Layout

| Path | Contents |
| --- | --- |
| `paper/` | Manuscript and supplement (Markdown and DOCX), figures, and the scripts that compute, check and build them |
| `paper/evidence.json` | Every number reported in the paper |
| `jevbench/` | Collection and analysis modules |
| `runs/` | One record per model request, including the provider response |
| `docs/` | Per-question result tables, physician-rating joins and run protocols |

## Reproducing the paper

Python 3.12 or later with the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
cd paper
python evidence.py         # recompute evidence.json from runs/ and docs/ (4,000 bootstrap resamples, slow)
python build_figures.py    # figures/
python build_document.py   # medrxiv-manuscript.docx
python validate.py         # check every table cell and reported number against evidence.json
```

`evidence.py` reads only the files in this repository. No API access is needed to reproduce the analysis.

## Collecting new responses

The collection scripts in `jevbench/` read credentials from a `typesafe.env` file at the repository root (`TYPESAFE_API_KEY`, `OPENAI_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`). This file is ignored by git. Clef requests are sent through a Cloudflare AI Gateway with the header `cf-aig-gateway-id`.

| Script | Collects |
| --- | --- |
| `decision_models.py`, `decision_models_resume.py` | OpenAI Decisions and Clef answers |
| `decision_models_order.py` | Option-rotation and repeat requests for OpenAI Decisions and Clef |
| `medical_order.py` | Option-rotation and repeat requests for Jev |
| `luna_logprob.py` | GPT-6 Luna answer-letter probabilities |
| `sol_relabel.py` | Korean content labels from GPT-6.1 Sol |
| `case_review.py` | Characterizations of the selected Korean errors in Supplementary Table 9 |

Jev main-collection answers and the reference-model answers are provided as per-question results in `docs/comparator-refresh-v1/unified/`.

## Data sources

Questions come from KorMedMCQA (sean0042/KorMedMCQA, revision 79efd6f) and MedQA (GBaker/MedQA-USMLE-4-options-hf, revision 17af935), and physician ratings from google-health/med-gemini-medqa-relabelling. Request records reproduce question text and remain subject to the terms of those datasets.
