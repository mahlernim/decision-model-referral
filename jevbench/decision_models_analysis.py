"""Analysis of OpenAI Decisions and Clef as first tiers, alongside Jev and the Luna letter-probability baseline.

Definitions follow docs/decision-models-v1/protocol.md and match the Jev and Luna analyses.
"""
import csv
import json
from collections import defaultdict

import numpy as np

from . import luna_logprob, luna_logprob_analysis
from .comparator_refresh_analysis import cascade
from .common import read
from .decision_models import DIRECTORY, MODELS, OUTPUT, manifest
from .escalation_curve import SOURCE, THRESHOLDS, curve
from .luna_logprob_analysis import calibration, scored
from .report import wilson
from .study_analysis import auc, bootstrap_auc, ci

SEED = 2026100702
NBOOT = 4000
SHARES = [10, 20, 30, 50]
TIERS = ['jev', 'luna_letter', 'decisions', 'clef']


def terminal(model, m):
    latest = {}
    for p in sorted((DIRECTORY / model / 'attempts').glob('*.json')):
        r = read(p)
        if r['manifest_hash'] != m['sha256']:
            raise ValueError('Foreign record')
        if r['terminal'] and (r['job'] not in latest or r['attempt'] > latest[r['job']]['attempt']):
            latest[r['job']] = r
    return latest


def tier_rows(model, latest, items):
    rows, probs = {}, {}
    for i in items:
        r = latest[f"{model}__{i['id']}"]; v = r.get('value')
        rows[i['id']] = dict(valid=str(v is not None), correct=str(bool(v and v['correct'])),
                             confidence='' if v is None else repr(v['selected_probability']),
                             standardized_cost_usd=repr(r['usage']['standardized_usd'] if r.get('usage') else 0.0))
        if v is not None:
            probs[i['id']] = v['probabilities']
    return rows, probs


def capture(rows, share):
    """Expected share of errors (among valid answers) in the least confident share of valid answers, ties averaged."""
    valid = sorted((float(r['confidence']), r['correct'] != 'True') for r in rows.values() if r['valid'] == 'True')
    errors = sum(e for _, e in valid)
    k = round(len(valid) * share / 100)
    if not errors:
        return None
    groups = defaultdict(list)
    for p, e in valid:
        groups[p].append(e)
    taken, caught = 0, 0.
    for p in sorted(groups):
        g = groups[p]
        if taken + len(g) <= k:
            caught += sum(g); taken += len(g)
        else:
            caught += (k - taken) * sum(g) / len(g); break
    return caught / errors


def roc(rows):
    v = [r for r in rows.values() if r['valid'] == 'True']
    return auc([r['correct'] != 'True' for r in v], [1 - float(r['confidence']) for r in v])


def cascade_rows(rows, sol):
    return [dict(id=q, jev_valid=r['valid'] == 'True', jev_correct=r['correct'] == 'True',
                 jev_p=float(r['confidence']) if r['valid'] == 'True' else None,
                 jev_cost_usd=float(r['standardized_cost_usd']),
                 destination_valid=sol[q]['valid'] == 'True', destination_correct=sol[q]['correct'] == 'True',
                 destination_cost_usd=float(sol[q]['standardized_cost_usd']),
                 jev_latency_ms=None, destination_latency_ms=None) for q, r in sorted(rows.items())]


def analyze(models=tuple(MODELS), write=True, cohorts=('kormed', 'medqa')):
    m = manifest()
    unified = defaultdict(dict)
    with open(SOURCE, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            unified[(r['cohort'], r['configuration_id'])][r['question_id']] = r
    latest = {model: terminal(model, m) for model in models}
    lm = luna_logprob.manifest(); luna_latest = luna_logprob_analysis.terminal_records(lm)
    luna_summary = read(luna_logprob.OUTPUT / 'summary.json')
    result = {}
    for cohort in cohorts:
        dev = [i for i in m['items'] if i['cohort'] == cohort and i['split'] == 'dev']
        test = [i for i in m['items'] if i['cohort'] == cohort and i['split'] == 'test']
        gold = {i['id']: i['gold'] for i in test}
        sol, jev = unified[(cohort, 'sol')], unified[(cohort, 'jev')]
        tiers = {'jev': (jev, {q: json.loads(r['option_probabilities_json']) for q, r in jev.items() if r['valid'] == 'True'},
                         THRESHOLDS[cohort], None)}
        luna_test = [i for i in lm['items'] if i['cohort'] == cohort and i['split'] == 'test']
        luna_rows = luna_logprob_analysis.first_tier(cohort, luna_latest, luna_test)
        luna_probs = {i['id']: luna_latest[f"luna__{i['id']}"]['value']['probabilities'] for i in luna_test
                      if luna_latest[f"luna__{i['id']}"].get('value')}
        tiers['luna_letter'] = (luna_rows, luna_probs, luna_summary[cohort]['threshold'], None)
        for model in models:
            rows, probs = tier_rows(model, latest[model], test)
            dev_rows, _ = tier_rows(model, latest[model], dev)
            dev_p = [float(r['confidence']) for r in dev_rows.values() if r['valid'] == 'True']
            tiers[model] = (rows, probs, float(np.quantile(dev_p, .2)), dict(n=len(dev), valid=len(dev_p)))
        entry = dict(n=len(test))
        for name, (rows, probs, thr, dev_info) in tiers.items():
            assert set(rows) == set(sol) == set(gold)
            sc = scored(rows, probs, gold)
            k = sum(r['correct'] == 'True' for r in rows.values())
            err = [not r['correct'] for r in sc]; score = [1 - r['p'] for r in sc]
            ps = [r['p'] for r in sc]
            c = cascade(cascade_rows(rows, sol), thr)
            pts = curve(rows, sol)
            referred = set(c['referred_ids'])
            errors_referred = sum(1 for q, r in rows.items() if q in referred and r['correct'] != 'True')
            entry[name] = dict(
                correct=k, valid=len(sc), accuracy=k / len(test), accuracy_ci95=wilson(k, len(test)),
                errors_among_valid=sum(err),
                standalone_cost_per_1000=sum(float(r['standardized_cost_usd']) for r in rows.values()) / len(test) * 1000,
                error_auroc=auc(err, score), error_auroc_ci95=bootstrap_auc(err, score),
                calibration=calibration(sc), distinct_selected_values=len(set(round(p, 6) for p in ps)),
                selected_probability_quantiles=[float(x) for x in np.quantile(ps, [.1, .25, .5, .75, .9])],
                error_capture={str(s): capture(rows, s) for s in SHARES},
                development=dev_info, threshold=thr,
                threshold_cascade=dict(referred=c['referred'], errors_referred=errors_referred,
                                       error_share_referred=errors_referred / sum(err) if sum(err) else None,
                                       accuracy=c['accuracy'], ci95=c['ci95'],
                                       rescued=c['rescued'], lost=c['lost'],
                                       cost_per_1000=c['standardized_cost']['per_1000_usd'],
                                       gain_over_matched_cost_random=c['matched_cost_random']['gain'],
                                       gain_over_matched_cost_random_ci95=c['matched_cost_random']['gain_ci95'],
                                       gain_over_same_count_random=c['same_count_random']['gain'],
                                       gain_over_same_count_random_ci95=c['same_count_random']['gain_ci95']),
                curve_at_share={str(s): dict(accuracy=pts[min(round(len(test) * s / 100), pts[-1]['k'])]['accuracy'],
                                             cost_per_1000=pts[min(round(len(test) * s / 100), pts[-1]['k'])]['cost_per_1000'])
                                for s in SHARES + [100]})
        # Paired source-question bootstrap of AUROC differences against Jev. Descriptive, different errors.
        ids = sorted(gold); rng = np.random.default_rng(SEED); boots = defaultdict(list)
        for draw in rng.integers(0, len(ids), size=(NBOOT, len(ids))):
            sample = [ids[i] for i in draw]
            a = roc({(q, n): jev[q] for n, q in enumerate(sample)})
            for name in list(tiers)[1:]:
                b = roc({(q, n): tiers[name][0][q] for n, q in enumerate(sample)})
                if a is not None and b is not None:
                    boots[name].append(a - b)
        entry['auroc_jev_minus'] = {name: dict(estimate=entry['jev']['error_auroc'] - entry[name]['error_auroc'], ci95=ci(boots[name]))
                                    for name in list(tiers)[1:]}
        result[cohort] = entry
    result['_settings'] = dict(seed=SEED, nboot=NBOOT, manifest=m['sha256'], protocol='docs/decision-models-v1/protocol.md')
    if write:
        (OUTPUT / 'summary.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    return result


def brief(result):
    for c in [c for c in ['kormed', 'medqa'] if c in result]:
        e = result[c]; print(c)
        for name in [t for t in TIERS if t in e]:
            x = e[name]; t = x['threshold_cascade']
            print(f"  {name:12s} acc {x['accuracy']*100:5.1f} valid {x['valid']}/{e['n']} AUROC {x['error_auroc']:.3f} "
                  f"{[round(v, 3) for v in x['error_auroc_ci95']]} ECE {x['calibration']['ece_fixed_bins']:.3f} "
                  f"p1 {x['calibration']['share_selected_probability_one']*100:4.1f}% distinct {x['distinct_selected_values']} "
                  f"capture {[None if v is None else round(v*100, 1) for v in x['error_capture'].values()]} "
                  f"thr {x['threshold']:.3f} cascade {t['accuracy']*100:.1f} ref {t['referred']} errs {t['error_share_referred']*100:.1f}% +{t['rescued']}/-{t['lost']} "
                  f"${t['cost_per_1000']:.4f} gain {t['gain_over_matched_cost_random']*100:.2f} "
                  f"{[round(v*100, 2) for v in t['gain_over_matched_cost_random_ci95']]} solo ${x['standalone_cost_per_1000']:.4f}")
        for name, d in e['auroc_jev_minus'].items():
            print(f"  AUROC jev-{name}: {d['estimate']:.3f} {[round(v, 3) for v in d['ci95']]}")


if __name__ == '__main__':
    brief(analyze())
