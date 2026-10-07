"""Analysis of the frozen Luna answer-letter probability baseline against Jev, both escalating to GPT-6.1 Sol."""
import csv
import json
import math
from collections import defaultdict

import numpy as np

from .common import ROOT, read
from .escalation_curve import SOURCE, THRESHOLDS, curve
from .luna_logprob import DIRECTORY, OUTPUT, manifest
from .report import wilson
from .study_analysis import auc, bootstrap_auc, paired_ci

EDGES = [0, .5, .7, .8, .9, .95, 1.00000001]
LOG_FLOOR = 1e-15


def terminal_records(m):
    latest = {}
    for p in sorted((DIRECTORY / 'attempts').glob('*.json')):
        r = read(p)
        if r['manifest_hash'] != m['sha256']:
            raise ValueError('Foreign record')
        if r['terminal'] and (r['job'] not in latest or r['attempt'] > latest[r['job']]['attempt']):
            latest[r['job']] = r
    return latest


def calibration(rows):
    p = np.array([r['p'] for r in rows]); j = np.array([r['correct'] for r in rows]); n = len(rows)
    ece = 0.
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum():
            ece += mask.sum() / n * abs(j[mask].mean() - p[mask].mean())
    return dict(ece_fixed_bins=float(ece),
                multiclass_brier=float(np.mean([r['brier'] for r in rows])),
                log_loss=float(np.mean([r['log_loss'] for r in rows])),
                share_selected_probability_one=float(np.mean(p >= 1 - 1e-12)))


def first_tier(cohort, latest, items):
    """Rows shaped like the unified item results, so the shared escalation curve can be reused."""
    out = {}
    for i in items:
        r = latest[f"luna__{i['id']}"]
        v = r.get('value')
        out[i['id']] = dict(valid=str(v is not None), correct=str(bool(v and v['correct'])),
                            confidence='' if v is None else repr(v['selected_probability']),
                            standardized_cost_usd=repr(r['usage']['standardized_usd'] if r.get('usage') else 0.0))
    return out


def scored(rows_by_id, probs_by_id, gold):
    rows = []
    for q, r in rows_by_id.items():
        if r['valid'] != 'True':
            continue
        probs = probs_by_id[q]
        brier = sum((probs[k] - (k == gold[q])) ** 2 for k in probs)
        rows.append(dict(q=q, p=float(r['confidence']), correct=r['correct'] == 'True', brier=brier,
                         log_loss=-math.log(max(probs[gold[q]], LOG_FLOOR))))
    return rows


def interpolate(points, cost):
    """Accuracy of a curve at a given total cost per 1,000 questions, linear between evaluated points."""
    xs = [p['cost_per_1000'] for p in points]; ys = [p['accuracy'] for p in points]
    if cost <= xs[0]: return ys[0] if cost == xs[0] else None
    if cost >= xs[-1]: return ys[-1]
    k = next(n for n in range(1, len(xs)) if xs[n] >= cost)
    return ys[k - 1] + (ys[k] - ys[k - 1]) * (cost - xs[k - 1]) / (xs[k] - xs[k - 1])


def analyze():
    m = manifest(); latest = terminal_records(m)
    unified = defaultdict(dict)
    with open(SOURCE, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            unified[(r['cohort'], r['configuration_id'])][r['question_id']] = r
    result = {}
    for cohort in ['kormed', 'medqa']:
        dev = [i for i in m['items'] if i['cohort'] == cohort and i['split'] == 'dev']
        test = [i for i in m['items'] if i['cohort'] == cohort and i['split'] == 'test']
        gold = {i['id']: i['gold'] for i in test}
        dev_p = [latest[f"luna__{i['id']}"]['value']['selected_probability'] for i in dev if latest[f"luna__{i['id']}"].get('value')]
        threshold = float(np.quantile(dev_p, .2))
        luna = first_tier(cohort, latest, test)
        luna_probs = {i['id']: latest[f"luna__{i['id']}"]['value']['probabilities'] for i in test if latest[f"luna__{i['id']}"].get('value')}
        jev, sol = unified[(cohort, 'jev')], unified[(cohort, 'sol')]
        assert set(jev) == set(sol) == set(luna)
        jev_probs = {q: json.loads(r['option_probabilities_json']) for q, r in jev.items() if r['valid'] == 'True'}
        n = len(test); entry = dict(n=n, development_n=len(dev), development_valid=len(dev_p), threshold=threshold)
        for name, rows_by_id, probs, thr in [('luna', luna, luna_probs, threshold), ('jev', jev, jev_probs, THRESHOLDS[cohort])]:
            rows = scored(rows_by_id, probs, gold)
            k = sum(r['correct'] == 'True' for r in rows_by_id.values())
            err = [not r['correct'] for r in rows]; score = [1 - r['p'] for r in rows]
            pts = curve(rows_by_id, sol)
            referred = sum(r['valid'] == 'True' and float(r['confidence']) <= thr for r in rows_by_id.values())
            # Fixed-threshold cascade at item level, for paired contrasts.
            cascade = {q: (sol[q]['correct'] == 'True') if r['valid'] == 'True' and float(r['confidence']) <= thr else (r['correct'] == 'True')
                       for q, r in rows_by_id.items()}
            cost = sum(float(r['standardized_cost_usd']) + (float(sol[q]['standardized_cost_usd']) if r['valid'] == 'True' and float(r['confidence']) <= thr else 0)
                       for q, r in rows_by_id.items()) / n * 1000
            entry[name] = dict(correct=k, valid=len(rows), accuracy=k / n, accuracy_ci95=wilson(k, n),
                               standalone_cost_per_1000=sum(float(r['standardized_cost_usd']) for r in rows_by_id.values()) / n * 1000,
                               error_auroc=auc(err, score), error_auroc_ci95=bootstrap_auc(err, score),
                               calibration=calibration(rows), threshold=thr, threshold_referred=referred,
                               threshold_cascade_correct=sum(cascade.values()), threshold_cascade_accuracy=sum(cascade.values()) / n,
                               threshold_cascade_cost_per_1000=cost, curve=pts, _cascade=cascade)
        ids = sorted(gold)
        entry['threshold_cascade_luna_minus_jev'] = paired_ci([entry['jev']['_cascade'][q] for q in ids], [entry['luna']['_cascade'][q] for q in ids])
        entry['standalone_luna_minus_jev'] = paired_ci([jev[q]['correct'] == 'True' for q in ids], [luna[q]['correct'] == 'True' for q in ids])
        grid = sorted({round(c, 4) for c in [entry['jev']['threshold_cascade_cost_per_1000'], entry['luna']['threshold_cascade_cost_per_1000'], .1, .2, .3, .5]})
        entry['matched_cost'] = [dict(cost_per_1000=c, jev_first=interpolate(entry['jev']['curve'], c), luna_first=interpolate(entry['luna']['curve'], c)) for c in grid]
        for name in ['luna', 'jev']:
            entry[name].pop('_cascade')
        result[cohort] = entry
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'summary.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    return result


def brief(result):
    for c, e in result.items():
        print(c, 'threshold', round(e['threshold'], 3))
        for name in ['jev', 'luna']:
            x = e[name]
            print(f"  {name}: acc {x['accuracy']:.3f} valid {x['valid']}/{e['n']} AUROC {x['error_auroc']:.3f} {[round(v, 3) for v in x['error_auroc_ci95']]} "
                  f"ECE {x['calibration']['ece_fixed_bins']:.3f} p=1 share {x['calibration']['share_selected_probability_one']:.2f} "
                  f"cascade {x['threshold_cascade_accuracy']:.3f} referred {x['threshold_referred']} at ${x['threshold_cascade_cost_per_1000']:.4f}")
        d = e['threshold_cascade_luna_minus_jev']
        print('  cascade luna-jev', round(d['difference'] * 100, 2), [round(v * 100, 2) for v in d['ci95']])
        for g in e['matched_cost']:
            print('  ', g)


if __name__ == '__main__':
    brief(analyze())
