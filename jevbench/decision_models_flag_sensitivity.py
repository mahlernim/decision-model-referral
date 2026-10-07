"""MedQA sensitivity analysis excluding physician-flagged questions, for Jev, Luna letter probabilities, Decisions and Clef.

A question is flagged when a strict majority of its three or four US physician raters reported missing important
information, no acceptable official answer or several acceptable answers (Saab et al. 2024), as in the manuscript.
Offline, existing answers only. Definitions and thresholds are those of decision_models_analysis.
"""
import csv
import json
from collections import defaultdict

from . import luna_logprob, luna_logprob_analysis
from .common import read
from .comparator_refresh_analysis import cascade
from .decision_models import MODELS, OUTPUT, manifest
from .decision_models_analysis import cascade_rows, capture, roc, terminal, tier_rows
from .error_bands import PHYSICIANS
from .escalation_curve import SOURCE, THRESHOLDS
from .report import wilson
from .study_analysis import bootstrap_auc

TIERS = ['jev', 'luna_letter', 'decisions', 'clef']


def first_tiers(m):
    unified = defaultdict(dict)
    with open(SOURCE, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            unified[(r['cohort'], r['configuration_id'])][r['question_id']] = r
    test = [i for i in m['items'] if i['cohort'] == 'medqa' and i['split'] == 'test']
    summary = read(OUTPUT / 'summary.json')
    lm = luna_logprob.manifest(); ll = luna_logprob_analysis.terminal_records(lm)
    luna_test = [i for i in lm['items'] if i['cohort'] == 'medqa' and i['split'] == 'test']
    tiers = {'jev': (unified[('medqa', 'jev')], THRESHOLDS['medqa']),
             'luna_letter': (luna_logprob_analysis.first_tier('medqa', ll, luna_test), summary['medqa']['luna_letter']['threshold'])}
    for model in MODELS:
        tiers[model] = (tier_rows(model, terminal(model, m), test)[0], summary['medqa'][model]['threshold'])
    return tiers, unified[('medqa', 'sol')]


def metrics(rows, sol, threshold, ids):
    sub = {q: rows[q] for q in ids}; n = len(sub)
    k = sum(r['correct'] == 'True' for r in sub.values())
    valid = [r for r in sub.values() if r['valid'] == 'True']
    err = [r['correct'] != 'True' for r in valid]; score = [1 - float(r['confidence']) for r in valid]
    c = cascade(cascade_rows(sub, {q: sol[q] for q in ids}), threshold)
    return dict(n=n, correct=k, accuracy=k / n, accuracy_ci95=wilson(k, n), errors=n - k,
                error_auroc=roc(sub), error_auroc_ci95=bootstrap_auc(err, score),
                errors_in_least_confident_20=capture(sub, 20),
                cascade=dict(referred=c['referred'], accuracy=c['accuracy'], ci95=c['ci95'], rescued=c['rescued'], lost=c['lost'],
                             gain_over_matched_cost_random=c['matched_cost_random']['gain'],
                             gain_ci95=c['matched_cost_random']['gain_ci95']))


def run():
    m = manifest(); tiers, sol = first_tiers(m)
    phys = {x['id']: x for x in read(PHYSICIANS)}
    all_ids = sorted(tiers['jev'][0])
    unflagged = [q for q in all_ids if not phys[q]['majority_concern']]
    flagged = [q for q in all_ids if phys[q]['majority_concern']]
    result = dict(n_all=len(all_ids), n_unflagged=len(unflagged), n_flagged=len(flagged))
    for name in TIERS:
        rows, thr = tiers[name]
        result[name] = dict(threshold=thr, all=metrics(rows, sol, thr, all_ids), unflagged=metrics(rows, sol, thr, unflagged),
                            flagged_accuracy=sum(rows[q]['correct'] == 'True' for q in flagged) / len(flagged))
    result['sol'] = dict(all=sum(sol[q]['correct'] == 'True' for q in all_ids) / len(all_ids),
                         unflagged=sum(sol[q]['correct'] == 'True' for q in unflagged) / len(unflagged))
    (OUTPUT / 'flag-sensitivity.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    return result


if __name__ == '__main__':
    r = run()
    print('questions', r['n_all'], 'unflagged', r['n_unflagged'], 'flagged', r['n_flagged'], '| Sol', round(r['sol']['all'] * 100, 1), round(r['sol']['unflagged'] * 100, 1))
    for name in TIERS:
        for part in ['all', 'unflagged']:
            x = r[name][part]; c = x['cascade']
            print(f"  {name:12s} {part:9s} acc {x['accuracy']*100:5.1f} AUROC {x['error_auroc']:.3f} {[round(v, 3) for v in x['error_auroc_ci95']]} "
                  f"errors {x['errors']} in20 {x['errors_in_least_confident_20']*100:.1f} | cascade {c['accuracy']*100:.1f} ref {c['referred']} "
                  f"gain {c['gain_over_matched_cost_random']*100:.2f} {[round(v*100, 2) for v in c['gain_ci95']]}")
        print(f"  {name:12s} flagged   acc {r[name]['flagged_accuracy']*100:5.1f}")
