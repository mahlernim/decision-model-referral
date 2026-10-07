"""Cautious versus confident Jev errors, as fixed in docs/error-confidence-bands-v1/analysis-note.md."""
import json
from collections import defaultdict

import numpy as np

from .common import ROOT, read
from .study_analysis import ci

SOURCE = ROOT / 'docs/jev-error-patterns-v1/joined-items.json'
PHYSICIANS = ROOT / 'docs/jev-error-patterns-v1/physician-annotation-join.json'
LABELS = ROOT / 'runs/sol-relabel-v1/attempts'
OUTPUT = ROOT / 'docs/error-confidence-bands-v1'
THRESHOLDS = {'kormed': .626, 'medqa': .770}
STRONG = ['sol', 'opus', 'gemini', 'lunamax']
# Post hoc addition of 7 October 2026, disclosed in the analysis note.
PANEL = ['sol', 'opus', 'gemini', 'lunamax', 'sonnet', 'luna']
SEED = 2026100702
NBOOT = 4000


def law_labels():
    out = {}
    for p in sorted(LABELS.glob('*.json')):
        r = read(p)
        if r.get('value'):
            out[r['job'].split('__')[1]] = r['value']['korean_law_policy']
    return out


def rows(cohort):
    phys = {x['id']: x for x in read(PHYSICIANS)}; law = law_labels(); out = []
    for i in read(SOURCE):
        if i['cohort'] != cohort or not i['valid']:
            continue
        o = i.get('order') or {}
        r = dict(id=i['id'], error=not i['correct'], confident=i['p'] > THRESHOLDS[cohort], p=i['p'],
                 flip=float(o['rotation_semantic_flip']) if o.get('complete_matched') == 'True' else None,
                 sol=i['comparators']['sol']['correct'],
                 shared=sum(i['comparators'][m]['prediction'] == i['prediction'] for m in STRONG) >= 3,
                 panel_accuracy=sum(i['comparators'][m]['correct'] for m in PANEL) / len(PANEL),
                 luna_same_answer=i['comparators']['luna']['valid'] and i['comparators']['luna']['prediction'] == i['prediction'])
        if cohort == 'kormed':
            r['law'] = law[i['id']]
        else:
            x = phys[i['id']]
            r.update(flagged=x['majority_concern'], physicians_chose_jev=x['endorses_jev'] * 2 > x['n_raters'])
        out.append(r)
    return out


def rate(rs, key):
    v = [r[key] for r in rs if r[key] is not None]
    return sum(v) / len(v) if v else None


def contrasts(rs):
    err = [r for r in rs if r['error']]
    hi, lo = [r for r in err if r['confident']], [r for r in err if not r['confident']]
    out = {}
    for key in ['flip', 'sol']:
        a, b = rate(hi, key), rate(lo, key)
        out[key] = None if a is None or b is None else a - b
    return out


def profile(rs, cohort):
    keys = ['flip', 'sol', 'shared', 'panel_accuracy', 'luna_same_answer'] + (['law'] if cohort == 'kormed' else ['flagged', 'physicians_chose_jev'])
    bands = {'cautious': lambda r: not r['confident'], 'confident': lambda r: r['confident']}
    if cohort == 'medqa':
        bands['at_least_0.90'] = lambda r: r['p'] >= .9
    out = {}
    for name, f in bands.items():
        for outcome in ['error', 'correct']:
            g = [r for r in rs if f(r) and r['error'] == (outcome == 'error')]
            out[f'{name}_{outcome}'] = dict(n=len(g), flip_n=sum(r['flip'] is not None for r in g),
                                            **{k: rate(g, k) for k in keys})
    return out


def run():
    rng = np.random.default_rng(SEED); result = {}
    for cohort in ['kormed', 'medqa']:
        rs = rows(cohort); point = contrasts(rs); boots = defaultdict(list)
        for draw in rng.integers(0, len(rs), size=(NBOOT, len(rs))):
            for k, v in contrasts([rs[i] for i in draw]).items():
                boots[k].append(v)
        result[cohort] = dict(
            threshold=THRESHOLDS[cohort], profile=profile(rs, cohort),
            contrasts={k: dict(confident_minus_cautious=point[k], ci95=ci(boots[k]),
                               excludes_zero=not (ci(boots[k])[0] <= 0 <= ci(boots[k])[1])) for k in point})
    result['_settings'] = dict(seed=SEED, nboot=NBOOT, note='docs/error-confidence-bands-v1/analysis-note.md')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'results.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    return result


if __name__ == '__main__':
    r = run()
    for c in ['kormed', 'medqa']:
        print(c)
        for k, v in r[c]['profile'].items():
            print(' ', k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
        for k, v in r[c]['contrasts'].items():
            print('  contrast', k, round(v['confident_minus_cautious'] * 100, 1), [round(x * 100, 1) for x in v['ci95']], v['excludes_zero'])
