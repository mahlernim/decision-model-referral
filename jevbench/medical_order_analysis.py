"""Offline, source-question-level analysis of medical answer-option rotations.

No requests are made here. Rotation/repeat contrasts use the same complete set
of source questions, with all rotations reduced to one value per question
before resampling. Public outputs contain identifiers and scores, never stems
or answer-option text.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from functools import partial

import numpy as np

from .common import now, read
from .report import wilson

NBOOT = 4000
SEED = 2026092201
METRICS = (
    "semantic_flip", "referral_flip", "absolute_selected_probability_change",
    "absolute_reported_confidence_change", "total_variation", "raw_referral_flip",
)
COHORT_NAMES = {"kormed": "Korean", "medqa": "English"}


def estimate(values, nboot=NBOOT, seed=SEED, binomial=False):
    """Percentile CI for a mean, resampling independent source questions."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or (len(a) and not np.isfinite(a).all()):
        raise ValueError("Bootstrap values must be a finite one-dimensional array")
    if not len(a):
        return {"n": 0, "mean": None, "ci95": None}
    if binomial and not np.isin(a, [0, 1]).all():
        raise ValueError("Binomial intervals require one binary outcome per source question")
    rng = np.random.default_rng(seed)
    boot = np.concatenate([
        a[rng.integers(0, len(a), size=(min(200, nboot - start), len(a)))].mean(axis=1)
        for start in range(0, nboot, 200)
    ])
    result = {"n": len(a), "mean": float(a.mean()),
              "ci95": [float(x) for x in np.quantile(boot, [.025, .975])]}
    if binomial and (a.sum() == 0 or a.sum() == len(a)):
        result["boundary_wilson_ci95"] = wilson(int(a.sum()), len(a))
    return result


def _mean(values):
    return float(np.mean(values)) if values else None


def _value(record):
    return record["value"] if record and record["status"] == "success" else None


def _comparison(a, b):
    if set(a["semantic_probabilities"]) != set(b["semantic_probabilities"]):
        raise ValueError("Semantic probability labels differ between variants")
    return {
        "semantic_flip": float(a["semantic_prediction"] != b["semantic_prediction"]),
        "referral_flip": float(a["referred"] != b["referred"]),
        "absolute_selected_probability_change": abs(a["selected_probability"] - b["selected_probability"]),
        "absolute_reported_confidence_change": abs(a["reported_confidence"] - b["reported_confidence"]),
        "total_variation": .5 * sum(abs(a["semantic_probabilities"][k] - b["semantic_probabilities"][k])
                                    for k in a["semantic_probabilities"]),
        "raw_referral_flip": float(a["raw_referred"] != b["raw_referred"]),
    }


def _index(manifest, attempts):
    """Reject duplicate terminal outcomes and mismatched immutable evidence."""
    evaluations = {e["eval_id"]: e for e in manifest["evaluations"]}
    if len(evaluations) != len(manifest["evaluations"]):
        raise ValueError("Duplicate evaluation IDs")
    terminal, seen = {}, set()
    for r in attempts:
        uid = r["eval_id"]
        if uid not in evaluations:
            raise ValueError(f"Unknown evaluation in attempts: {uid}")
        e = evaluations[uid]
        key = (uid, r["attempt"])
        if key in seen:
            raise ValueError(f"Duplicate attempt: {key}")
        seen.add(key)
        for field in ("item_id", "cohort", "variant", "request_hash"):
            if r[field] != e[field]:
                raise ValueError(f"Attempt {field} mismatch: {uid}")
        if r["manifest_hash"] != manifest["sha256"]:
            raise ValueError(f"Attempt manifest mismatch: {uid}")
        if r["status"] not in {"success", "invalid", "error"}:
            raise ValueError(f"Unknown attempt status: {uid}")
        if r.get("terminal", r["status"] in {"success", "invalid"}):
            if uid in terminal:
                raise ValueError(f"Multiple terminal outcomes: {uid}")
            terminal[uid] = r
    return evaluations, terminal


def _accuracy(evaluations, terminal):
    """Full planned denominator with missing calls explicitly disclosed."""
    states = Counter()
    successful = []
    for e in evaluations:
        r = terminal.get(e["eval_id"])
        states[r["status"] if r else "missing"] += 1
        if _value(r):
            successful.append(r["value"])
    n = len(evaluations)
    correct = sum(bool(v["correct"]) for v in successful)
    return {"planned": n, "successful": states["success"], "invalid": states["invalid"],
            "terminal_error": states["error"], "missing": states["missing"], "correct": correct,
            "accuracy_full_denominator": correct / n if n else None,
            "accuracy_successful_only": correct / len(successful) if successful else None,
            "referred_successful": sum(bool(v["referred"]) for v in successful),
            "referral_rate_successful": _mean([float(v["referred"]) for v in successful]),
            "raw_referral_rate_successful": _mean([float(v["raw_referred"]) for v in successful]),
            "renormalized_successful": sum(bool(v["renormalized"]) for v in successful)}


def _attempt_resources(attempts):
    times = [float(r["latency_ms"]) / 1000 for r in attempts if r.get("latency_ms") is not None]
    return {"attempts": len(attempts),
            "estimated_cost_usd": sum(float(r.get("estimated_cost_usd") or 0) for r in attempts),
            "unpriced_attempts": sum(r.get("estimated_cost_usd") is None for r in attempts),
            "budget_charge_usd": sum(float(r.get("budget_charge_usd") or 0) for r in attempts),
            "input_tokens": sum(int(r.get("input_tokens") or 0) for r in attempts),
            "summed_request_seconds": sum(times), "mean_request_seconds": _mean(times),
            "median_request_seconds": float(np.median(times)) if times else None,
            "status_counts": dict(Counter(r["status"] for r in attempts))}


def _historical(items, values, seed=SEED):
    """Historical comparisons are descriptive service drift, not the control."""
    result = {}
    for variant in ("rot0", "repeat"):
        semantic, confidence, correctness, referral = [], [], [], []
        for item in items:
            old = item.get("historical")
            new = values.get((item["id"], variant))
            if not old or not new or old.get("status", "success") != "success":
                continue
            old = old.get("value", old)
            prediction = old.get("semantic_prediction", old.get("prediction"))
            if prediction is not None:
                semantic.append(float(prediction != new["semantic_prediction"]))
            probability = old.get("selected_probability", old.get("confidence"))
            if probability is not None:
                confidence.append(abs(float(probability) - new["selected_probability"]))
            if old.get("correct") is not None:
                correctness.append(float(new["correct"]) - float(old["correct"]))
            if old.get("referred") is not None:
                referral.append(float(bool(old["referred"]) != new["referred"]))
        result[variant] = {"semantic_flip": estimate(semantic, seed=seed),
                           "absolute_selected_probability_change": estimate(confidence, seed=seed),
                           "accuracy_change": estimate(correctness, seed=seed), "referral_flip": estimate(referral, seed=seed)}
    return result


def summarize(manifest, attempts):
    """Return JSON-safe summary and text-free per-source-question rows."""
    evaluations, terminal = _index(manifest, attempts)
    seed = manifest.get("seed", SEED)
    est = partial(estimate, seed=seed)
    by_item, by_cohort = defaultdict(list), defaultdict(list)
    values = {}
    for e in evaluations.values():
        by_item[e["item_id"]].append(e)
        by_cohort[e["cohort"]].append(e)
        val = _value(terminal.get(e["eval_id"]))
        if val is not None:
            values[(e["item_id"], e["variant"])] = val
    summary = {
        "experiment": manifest.get("experiment", "medical-option-order-v1"),
        "model": manifest.get("model"),
        "generated_at": now(), "manifest_hash": manifest["sha256"],
        "planned_evaluations": len(evaluations), "terminal_evaluations": len(terminal),
        "successful_evaluations": sum(r["status"] == "success" for r in terminal.values()),
        "provisional": len(terminal) != len(evaluations),
        "bootstrap": {"replicates": NBOOT, "seed": seed, "unit": "source question",
                      "method": "percentile CI of source-question means or paired differences"},
        "resources": _attempt_resources(attempts), "cohorts": {},
        "interpretation": [
            "Cyclic rotations jointly reassign answer labels and positions. They are not all permutations.",
            "Primary contrasts use only source questions with successful original, repeat and all rotations.",
            "Nonzero rotations are averaged within each question before source-question bootstrap.",
            "Intervals are descriptive pointwise 95% intervals without multiplicity adjustment.",
            "Missing and invalid outcomes are retained in the planned-denominator accuracy tables.",
            "Any-rotation instability is descriptive and cannot be compared directly with one repeat.",
            "Referral is a simulated threshold decision. No downstream model or clinical outcome is measured.",
        ],
    }
    rows = []
    for cohort, cohort_evals in by_cohort.items():
        items = [i for i in manifest["items"] if i["cohort"] == cohort]
        variants = sorted({e["variant"] for e in cohort_evals}, key=lambda x: (x == "repeat", int(x[3:]) if x.startswith("rot") else 0))
        cohort_rows = []
        full_acc = defaultdict(list)
        for item in items:
            uid = item["id"]
            planned = by_item[uid]
            rotations = sorted([e["variant"] for e in planned if e["variant"].startswith("rot") and e["rotation"] != 0])
            base, repeat = values.get((uid, "rot0")), values.get((uid, "repeat"))
            rotated = [values.get((uid, v)) for v in rotations]
            complete = bool(base is not None and repeat is not None and rotations and all(v is not None for v in rotated))
            row = {"item_id": uid, "cohort": cohort, "source_item_hash": item["source_item_hash"],
                   "planned_evaluations": len(planned),
                   "successful_evaluations": sum((uid, e["variant"]) in values for e in planned),
                   "terminal_invalid": sum(terminal.get(e["eval_id"], {}).get("status") == "invalid" for e in planned),
                   "missing_evaluations": sum(e["eval_id"] not in terminal for e in planned),
                   "complete_matched": complete}
            for variant, label in (("rot0", "original"), ("repeat", "repeat")):
                val = values.get((uid, variant))
                full_acc[label].append(float(val["correct"]) if val else 0.)
                row[f"{label}_correct"] = bool(val["correct"]) if val else None
                row[f"{label}_semantic_prediction"] = val["semantic_prediction"] if val else None
                row[f"{label}_selected_probability"] = val["selected_probability"] if val else None
                row[f"{label}_referred"] = bool(val["referred"]) if val else None
            rotation_accuracy = _mean([float(v["correct"]) if v else 0. for v in rotated])
            row["rotation_accuracy_full_denominator"] = rotation_accuracy
            if rotation_accuracy is not None:
                full_acc["rotation_average"].append(rotation_accuracy)
            if complete:
                comparisons = [_comparison(base, v) for v in rotated]
                control = _comparison(base, repeat)
                for metric in METRICS:
                    order_mean = _mean([v[metric] for v in comparisons])
                    row[f"rotation_{metric}"] = order_mean
                    row[f"repeat_{metric}"] = control[metric]
                    row[f"excess_{metric}"] = order_mean - control[metric]
                row["any_rotation_semantic_flip"] = any(v["semantic_flip"] for v in comparisons)
                row["any_rotation_referral_flip"] = any(v["referral_flip"] for v in comparisons)
                row["rotation_accuracy_change"] = rotation_accuracy - float(base["correct"])
                row["repeat_accuracy_change"] = float(repeat["correct"]) - float(base["correct"])
            cohort_rows.append(row)
        matched = [r for r in cohort_rows if r["complete_matched"]]
        metrics = {metric: {name: est([r[f"{name}_{metric}"] for r in matched],
                                     binomial=(name == "repeat" and metric.endswith("flip")))
                            for name in ("rotation", "repeat", "excess")} for metric in METRICS}
        cohort_summary = {
            "planned_items": len(items), "complete_matched_items": len(matched),
            "excluded_from_matched_items": len(items) - len(matched),
            "thresholds": sorted({e["threshold"] for e in cohort_evals}),
            "full_cohort_accuracy": {key: est(v) for key, v in full_acc.items()},
            "matched_accuracy": {
                "original": est([float(r["original_correct"]) for r in matched]),
                "repeat": est([float(r["repeat_correct"]) for r in matched]),
                "rotation_average": est([r["rotation_accuracy_full_denominator"] for r in matched]),
                "rotation_minus_original": est([r["rotation_accuracy_change"] for r in matched]),
                "repeat_minus_original": est([r["repeat_accuracy_change"] for r in matched]),
            },
            "metrics": metrics,
            "any_rotation_instability_descriptive": {
                "semantic": est([float(r["any_rotation_semantic_flip"]) for r in matched], binomial=True),
                "referral": est([float(r["any_rotation_referral_flip"]) for r in matched], binomial=True),
            },
            "variants": {}, "historical_drift_descriptive": _historical(items, values, seed=seed),
        }
        for variant in variants:
            ee = [e for e in cohort_evals if e["variant"] == variant]
            rr = [r for r in attempts if r["cohort"] == cohort and r["variant"] == variant]
            pairs = [(values.get((e["item_id"], "rot0")), values.get((e["item_id"], variant))) for e in ee]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            rescued = sum(not a["correct"] and b["correct"] for a, b in pairs)
            lost = sum(a["correct"] and not b["correct"] for a, b in pairs)
            cohort_summary["variants"][variant] = {
                **_accuracy(ee, terminal), "resources": _attempt_resources(rr),
                "successful_original_pairs": len(pairs), "rescued_vs_original": rescued,
                "lost_vs_original": lost, "net_correct_vs_original": rescued - lost,
                "paired_accuracy_change": est([float(b["correct"]) - float(a["correct"]) for a, b in pairs]),
            }
        # Rotation rows alone form a balanced positional design when complete.
        position, choices = defaultdict(list), Counter()
        matched_ids = {r["item_id"] for r in matched}
        for e in cohort_evals:
            if not e["variant"].startswith("rot") or e["item_id"] not in matched_ids:
                continue
            val = values[(e["item_id"], e["variant"])]
            labels = list(e["display_to_original"])
            position[labels.index(e["gold"]) + 1].append(bool(val["correct"]))
            choices[val["prediction"]] += 1
        nchoices = sum(choices.values())
        cohort_summary["exploratory_balanced_rotations"] = {
            "source_items": len(matched), "evaluations": nchoices,
            "accuracy_by_displayed_gold_position": {str(p): {"n_evaluations": len(v), "correct": sum(v), "accuracy": _mean(v)}
                                                     for p, v in sorted(position.items())},
            "display_label_choice_counts": dict(sorted(choices.items())),
            "display_label_choice_rates": {k: n / nchoices for k, n in sorted(choices.items())},
            "note": "Descriptive repeated evaluations, with each source question contributing once at every correct-answer position. No independent-evaluation CI or hypothesis test.",
        }
        summary["cohorts"][cohort] = cohort_summary
        rows.extend(cohort_rows)
    return summary, rows


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def analyze():
    from .medical_order import DIRECTORY, OUTPUT, manifest, records
    summary, rows = summarize(manifest(), records())
    timing_records = [read(path) for path in sorted((DIRECTORY / "timing").glob("*.json"))]
    phases = defaultdict(float)
    for timing in timing_records:
        seconds = float(timing["wall_seconds"])
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid collection timing record")
        phases[timing["phase"]] += seconds
    summary["collection_timing"] = {
        "completed_timing_records": len(timing_records),
        "summed_invocation_wall_seconds": sum(phases.values()),
        "phase_wall_seconds": dict(phases),
        "note": "Sum of completed collection invocation timings. Excludes preparation, human gaps and an invocation still running.",
    }
    _write_json(DIRECTORY / "summary.json", summary)
    _write_json(OUTPUT / "summary.json", summary)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUTPUT / "item-level.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def _fmt(value, percent=False, signed=False):
    if value is None:
        return "NA"
    suffix, multiplier = ("%", 100) if percent else ("", 1)
    return format(value * multiplier, "+.2f" if signed else ".2f") + suffix


def _estimate_text(value, percent=False, signed=False):
    if value["mean"] is None:
        return "NA"
    low, high = value["ci95"]
    text = f"{_fmt(value['mean'], percent, signed)} [{_fmt(low, percent, signed)}, {_fmt(high, percent, signed)}]"
    if value.get("boundary_wilson_ci95"):
        lo, hi = value["boundary_wilson_ci95"]
        text += f" (Wilson {_fmt(lo, percent)}, {_fmt(hi, percent)})"
    return text


def _figure(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    cohorts = list(summary["cohorts"])
    panels = [("semantic_flip", "A  Answer changes", "Source-question mean (%)", 100),
              ("referral_flip", "B  Referral changes", "Source-question mean (%)", 100),
              ("total_variation", "C  Probability redistribution", "Mean total variation", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), constrained_layout=True)
    for ax, (metric, title, ylabel, scale) in zip(axes, panels):
        x = np.arange(len(cohorts))
        for label, offset, color in (("rotation", -.17, "#2166AC"), ("repeat", .17, "#999999")):
            for j, cohort in enumerate(cohorts):
                result = summary["cohorts"][cohort]["metrics"][metric][label]
                if result["mean"] is None:
                    continue
                y = scale * result["mean"]
                low, high = [scale * z for z in result["ci95"]]
                ax.bar(x[j] + offset, y, .30, color=color,
                       label=("Cyclic rotations" if label == "rotation" else "Identical repeat") if j == 0 else None)
                ax.errorbar(x[j] + offset, y, yerr=[[max(0, y - low)], [max(0, high - y)]],
                            fmt="none", ecolor="#222222", capsize=3, linewidth=1)
        ax.set_xticks(x, [f"{COHORT_NAMES.get(c, c)}\nn = {summary['cohorts'][c]['complete_matched_items']}" for c in cohorts])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        if not any(summary["cohorts"][c]["complete_matched_items"] for c in cohorts):
            ax.text(.5, .5, "No complete source questions", transform=ax.transAxes, ha="center")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, frameon=False, fontsize=8)
    state = "Provisional | " if summary["provisional"] else ""
    fig.suptitle(f"{state}Jev medical answer-option robustness | 95% source-question bootstrap intervals", fontsize=10)
    fig.savefig(output / "option-order-robustness.png", dpi=220)
    fig.savefig(output / "option-order-robustness.pdf")
    plt.close(fig)


def _markdown(summary):
    completed = not summary["provisional"]
    lines = ["# Medical answer-option order robustness", "",
             "**Completed collection.**" if completed else "**Provisional report. Collection is incomplete.**", "",
             f"Successful evaluations {summary['successful_evaluations']:,} of {summary['planned_evaluations']:,}. "
             f"Terminal evaluations {summary['terminal_evaluations']:,}.", "",
             f"Model `{summary.get('model') or 'unavailable'}`.", "",
             "This experiment tests whether Jev gives the same semantic answer, probability distribution and simulated referral decision "
             "after cyclically rotating medical answer options. Each source question has a contemporary original-order request, all nonzero "
             "cyclic rotations and one identical-request repeat. Answer labels are mapped back to their original meanings before comparison.", "",
             "Cyclic rotations jointly reassign labels and positions and cover each answer position once. They do not cover all permutations "
             "or isolate the effects of position from label. The identical-request repeat measures repeat variability within this collection. "
             "It does not establish a deterministic floor or fully remove temporal service variation.", "",
             "Referral uses the frozen cohort-specific threshold on the selected answer's normalized probability. "
             "The decision rule is probability less than or equal to the threshold. Raw-probability referral is a sensitivity analysis. "
             "Referral is simulated and does not measure downstream answers or clinical outcomes.", "",
             "## Matched source-question contrasts", "",
             "All primary comparisons use source questions with successful original, repeat and every planned rotation. "
             "For each question, nonzero rotations are averaged before calculating cohort means. "
             "Excess is the paired within-question difference between this rotation mean and its repeat result. "
             "Brackets show percentile 95% intervals from 4,000 source-question bootstrap samples. "
             "Intervals are pointwise and descriptive, without multiplicity adjustment.", "",
             "When all observed binary outcomes are zero or one, percentile bootstrap intervals can collapse to a point. "
             "This does not establish certainty or invariance. For boundary repeat-change and any-rotation-change rates, "
             "supplementary Wilson binomial intervals are shown alongside the primary bootstrap intervals.", "",
             "| Cohort | Planned questions | Complete matched | Excluded | Referral threshold |",
             "|---|---:|---:|---:|---:|"]
    for cohort, c in summary["cohorts"].items():
        lines.append(f"| {COHORT_NAMES.get(cohort, cohort)} | {c['planned_items']} | {c['complete_matched_items']} | {c['excluded_from_matched_items']} | {', '.join(str(v) for v in c['thresholds'])} |")
    labels = {"semantic_flip": "Semantic answer change", "referral_flip": "Referral decision change",
              "absolute_selected_probability_change": "Absolute selected-probability change",
              "absolute_reported_confidence_change": "Absolute native reported-confidence change",
              "total_variation": "Total variation of semantic probabilities",
              "raw_referral_flip": "Raw-probability referral change (sensitivity)"}
    lines += ["", "| Cohort | Outcome | Mean rotation | Identical repeat | Paired excess |",
              "|---|---|---:|---:|---:|"]
    for cohort, c in summary["cohorts"].items():
        for metric in METRICS:
            percent = metric.endswith("flip")
            values = c["metrics"][metric]
            lines.append(f"| {COHORT_NAMES.get(cohort, cohort)} | {labels[metric]} | "
                         f"{_estimate_text(values['rotation'], percent)} | {_estimate_text(values['repeat'], percent)} | "
                         f"{_estimate_text(values['excess'], percent, True)} |")
    lines += ["", "For change rates, excess values are percentage-point differences. Total variation is half the summed absolute "
              "difference between semantic probability vectors. Selected-probability change compares the confidence in each request's "
              "selected answer, which can refer to different semantic answers when the prediction changes. "
              "Native reported confidence is the service's separate confidence field and is not used for referral.", "",
              "![Answer, referral and probability stability](option-order-robustness.png)", "",
              "## Accuracy and completeness", "",
              "Full-cohort accuracy uses all planned source questions. Terminal invalid outputs and terminal errors count as incorrect. "
              "During incomplete collection, missing evaluations also contribute zero, so provisional full-denominator accuracy is a "
              "completion-sensitive lower bound and should not be interpreted as final model performance. "
              "Rotation-average accuracy gives each source question equal weight.", "",
              "| Cohort | Full original | Full repeat | Full rotation average | Matched original | Matched repeat | Matched rotation average |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for cohort, c in summary["cohorts"].items():
        results = [c[key][variant] for key in ("full_cohort_accuracy", "matched_accuracy")
                   for variant in ("original", "repeat", "rotation_average")]
        lines.append(f"| {COHORT_NAMES.get(cohort, cohort)} | " + " | ".join(_estimate_text(v, True) for v in results) + " |")
    lines += ["", "Rescue and loss compare successful pairs with the contemporary original. Pair counts can differ by variant, "
              "so the matched contrasts above remain primary. Request time is summed measured request latency, not elapsed collection time. "
              "Resource totals include unsuccessful attempts and retries when recorded.", "",
              "| Cohort | Variant | Planned | Success | Invalid | Terminal error | Missing | Correct | Rescued | Lost | Pair n | Est. cost USD | Request seconds |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cohort, c in summary["cohorts"].items():
        for variant, v in c["variants"].items():
            lines.append(f"| {COHORT_NAMES.get(cohort, cohort)} | {variant} | {v['planned']} | {v['successful']} | {v['invalid']} | "
                         f"{v['terminal_error']} | {v['missing']} | {v['correct']} | {v['rescued_vs_original']} | {v['lost_vs_original']} | "
                         f"{v['successful_original_pairs']} | {v['resources']['estimated_cost_usd']:.6f} | {v['resources']['summed_request_seconds']:.1f} |")
    lines += ["", "## Exploratory descriptions", "",
              "Any-rotation instability asks whether at least one nonzero rotation changes the result. "
              "It has multiple opportunities to change and must not be directly compared with a single identical repeat.", "",
              "| Cohort | Any rotation changes answer | Any rotation changes referral |",
              "|---|---:|---:|"]
    for cohort, c in summary["cohorts"].items():
        v = c["any_rotation_instability_descriptive"]
        lines.append(f"| {COHORT_NAMES.get(cohort, cohort)} | {_estimate_text(v['semantic'], True)} | {_estimate_text(v['referral'], True)} |")
    lines += ["", "Displayed correct-answer position and choice-label preference use all rotations of complete matched questions, "
              "including original order. Each question contributes one observation at every correct-answer position. "
              "These repeated observations are descriptive and are not treated as independent samples.", "",
              "| Cohort | Correct-answer position | Evaluations | Correct | Accuracy |",
              "|---|---:|---:|---:|---:|"]
    for cohort, c in summary["cohorts"].items():
        for position, v in c["exploratory_balanced_rotations"]["accuracy_by_displayed_gold_position"].items():
            lines.append(f"| {COHORT_NAMES.get(cohort, cohort)} | {position} | {v['n_evaluations']} | {v['correct']} | {_fmt(v['accuracy'], True)} |")
    lines += ["", "| Cohort | Display label selected | Count | Share |", "|---|---|---:|---:|"]
    for cohort, c in summary["cohorts"].items():
        v = c["exploratory_balanced_rotations"]
        for label, n in v["display_label_choice_counts"].items():
            lines.append(f"| {COHORT_NAMES.get(cohort, cohort)} | {label} | {n} | {_fmt(v['display_label_choice_rates'][label], True)} |")
    lines += ["", "Historical results, when present, are compared separately with current original and repeat requests. "
              "Preparation verified identical historical and contemporary original request hashes. "
              "Differences can reflect temporal service variation and repeat variability and do not replace the contemporary control.", "",
              "| Cohort | Current variant | Historical matched n | Answer change | Accuracy change |",
              "|---|---|---:|---:|---:|"]
    for cohort, c in summary["cohorts"].items():
        for variant, v in c["historical_drift_descriptive"].items():
            lines.append(f"| {COHORT_NAMES.get(cohort, cohort)} | {variant} | {v['semantic_flip']['n']} | "
                         f"{_estimate_text(v['semantic_flip'], True)} | {_estimate_text(v['accuracy_change'], True, True)} |")
    resources = summary["resources"]
    timing = summary.get("collection_timing")
    if timing:
        lines += ["", f"Recorded collection wall time {timing['summed_invocation_wall_seconds']:.1f} seconds "
                  f"across {timing['completed_timing_records']} completed invocation timing records. "
                  "This excludes preparation, human gaps and any invocation still running."]
    lines += ["", "## Evidence and reproducibility", "",
              f"Manifest SHA-256 `{summary['manifest_hash']}`. Generated {summary['generated_at']}. "
              f"Recorded attempts {resources['attempts']:,}, estimated cost ${resources['estimated_cost_usd']:.6f}, "
              f"conservative budget charge ${resources['budget_charge_usd']:.6f}, "
              f"and attempts without a recorded cost estimate {resources['unpriced_attempts']:,}. "
              "Estimated cost sums available estimates and is not an invoice.", "",
              "The immutable local manifest and attempt records are the analysis evidence. "
              "Public outputs contain identifiers and aggregate scores without licensed question text.", "",
              "- [Machine-readable summary](summary.json)",
              "- [Source-question results](item-level.csv)",
              "- [Figure PDF](option-order-robustness.pdf)",
              "- [Frozen experiment protocol](protocol.md)",
              "- [Verification record, when available](verification.json)",
              "- [Analysis source](../../jevbench/medical_order_analysis.py)",
              "- [Experiment runner and protocol](../../jevbench/medical_order.py)",
              "- [Original Korean study](../kormed-study-v1/report.md)",
              "- [Medical extension](../medical-extension-v1/report.md)", "",
              "Rebuild offline with `python -m jevbench.medical_order_analysis --report`. "
              "No API requests are made by the analysis command.", ""]
    return "\n".join(lines)


def _html(markdown):
    """Small deterministic renderer for the limited report Markdown subset."""
    import re

    def inline(text):
        text = html.escape(text)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
        return text

    chunks, in_table, in_list = [], False, False
    for line in markdown.splitlines():
        is_table = line.startswith("|")
        is_list = line.startswith("- ")
        if in_table and not is_table:
            chunks.append("</tbody></table></div>")
            in_table = False
        if in_list and not is_list:
            chunks.append("</ul>")
            in_list = False
        if not line:
            continue
        if is_table:
            cells = [v.strip() for v in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", v) for v in cells):
                continue
            if not in_table:
                chunks.append('<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{inline(v)}</th>" for v in cells) + "</tr></thead><tbody>")
                in_table = True
            else:
                chunks.append("<tr>" + "".join(f"<td>{inline(v)}</td>" for v in cells) + "</tr>")
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[([^\]]+)\]\(([^)]+)\)", line)
            if match:
                chunks.append(f'<img src="{html.escape(match[2])}" alt="{html.escape(match[1])}">')
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            chunks.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
        elif is_list:
            if not in_list:
                chunks.append("<ul>")
                in_list = True
            chunks.append(f"<li>{inline(line[2:])}</li>")
        else:
            chunks.append(f"<p>{inline(line)}</p>")
    return """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Medical answer-option order robustness</title><style>
body{font:16px/1.6 system-ui,sans-serif;color:#17222e;background:#f6f8fa;margin:0}
main{max-width:1200px;margin:28px auto;padding:36px;background:white;border:1px solid #dce2e8}
h1{font-size:2rem;line-height:1.2}h2{margin-top:2em;font-size:1.4rem}a{color:#155d96}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}
th,td{border-bottom:1px solid #dce2e8;padding:9px 10px;text-align:right;white-space:nowrap}
th{background:#edf3f8}th:first-child,td:first-child{text-align:left}img{max-width:100%;height:auto}
code{font:13px ui-monospace,monospace;overflow-wrap:anywhere;background:#f0f3f6;padding:2px 4px}
@media(max-width:640px){main{padding:18px;margin:0}h1{font-size:1.6rem}}
</style><main>""" + "\n".join(chunks) + "</main></html>\n"


def report():
    from .medical_order import OUTPUT
    summary = analyze()
    _figure(summary, OUTPUT)
    markdown = _markdown(summary)
    (OUTPUT / "report.md").write_text(markdown, encoding="utf-8")
    (OUTPUT / "report.html").write_text(_html(markdown), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="Write summary, source-question CSV, report and figures offline")
    args = parser.parse_args()
    summary = report() if args.report else analyze()
    print(json.dumps({"planned": summary["planned_evaluations"], "successful": summary["successful_evaluations"],
                      "provisional": summary["provisional"],
                      "complete_matched": {k: v["complete_matched_items"] for k, v in summary["cohorts"].items()}}, indent=2))


if __name__ == "__main__":
    main()
