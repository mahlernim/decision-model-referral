"""Where errors remain for Jev, OpenAI Decisions and Clef, using the Jev error-band definitions for every model.

Exploratory, after all Jev error analyses were known. Thresholds are each model's own development threshold.
"""
import json
from collections import defaultdict

import numpy as np

from .common import ROOT, read
from .decision_models import OUTPUT, items
from .decision_models_analysis import terminal
from .decision_models import manifest as main_manifest
from .decision_models_order_analysis import load as load_order
from .error_bands import PANEL, PHYSICIANS, SOURCE, STRONG, law_labels
from .study_analysis import ci

SEED = 2026100704
NBOOT = 4000
MODELS = ['jev', 'decisions', 'clef']


def answers():
    """Test answers per model: prediction, correctness and selected probability, or None when invalid."""
    joined = {i['id']: i for i in read(SOURCE)}
    out = {'jev': {q: (dict(prediction=i['prediction'], correct=i['correct'], p=i['p']) if i['valid'] else None)
                   for q, i in joined.items()}}
    m = main_manifest()
    for model in ['decisions', 'clef']:
        latest = terminal(model, m)
        out[model] = {}
        for i in m['items']:
            if i['split'] != 'test':
                continue
            v = latest[f"{model}__{i['id']}"].get('value')
            out[model][i['id']] = dict(prediction=v['answer'], correct=v['correct'], p=v['selected_probability']) if v else None
    return joined, out


def thresholds():
    s = read(OUTPUT / 'summary.json')
    return {m: {c: s[c][m]['threshold'] for c in ('kormed', 'medqa')} for m in MODELS}


def flips(model):
    """Per-question mean semantic answer change across nonzero rotations, complete valid blocks only."""
    v = load_order(model); out = {}
    for (c, q, var) in list(v):
        if var != 'rot0':
            continue
        n = 5 if c == 'kormed' else 4
        base = v[(c, q, 'rot0')]; rep = v.get((c, q, 'repeat')); rot = [v.get((c, q, f'rot{k}')) for k in range(1, n)]
        if base is None or rep is None or any(x is None for x in rot):
            out[q] = None
        else:
            out[q] = sum(x['semantic_prediction'] != base['semantic_prediction'] for x in rot) / len(rot)
    return out


def rows(model, cohort, joined, ans, thr, flip, law, phys):
    out = []
    for q, i in joined.items():
        a = ans[model][q]
        if i['cohort'] != cohort or a is None:
            continue
        comp = i['comparators']
        r = dict(id=q, error=not a['correct'], confident=a['p'] > thr[model][cohort], p=a['p'], flip=flip.get(q),
                 sol=comp['sol']['correct'],
                 shared=sum(comp[m]['valid'] and comp[m]['prediction'] == a['prediction'] for m in STRONG) >= 3,
                 panel_accuracy=sum(comp[m]['correct'] for m in PANEL) / len(PANEL),
                 luna_same_answer=comp['luna']['valid'] and comp['luna']['prediction'] == a['prediction'])
        if cohort == 'kormed':
            r['law'] = law[q]
        else:
            x = phys[q]
            r.update(flagged=x['majority_concern'],
                     physicians_chose_model=sum(a['prediction'] in rt['answers'] for rt in x['raters']) * 2 > x['n_raters'])
        out.append(r)
    return out


def rate(rs, key):
    v = [r[key] for r in rs if r.get(key) is not None]
    return sum(v) / len(v) if v else None


def contrasts(rs):
    err = [r for r in rs if r['error']]
    hi, lo = [r for r in err if r['confident']], [r for r in err if not r['confident']]
    out = {}
    for key in ['flip', 'sol']:
        a, b = rate(hi, key), rate(lo, key)
        out[key] = None if a is None or b is None else a - b
    return out


def concentration(model, cohort, joined, ans, thr, law, phys):
    """Full denominator. Invalid answers count as errors and are never cautious, as in the manuscript."""
    key = 'law' if cohort == 'kormed' else 'flagged'
    rs = []
    for q, i in joined.items():
        if i['cohort'] != cohort:
            continue
        a = ans[model][q]
        rs.append(dict(error=a is None or not a['correct'], cautious=a is not None and a['p'] <= thr[model][cohort],
                       content=law[q] if cohort == 'kormed' else phys[q]['majority_concern']))
    sub, other = [r for r in rs if r['content']], [r for r in rs if not r['content']]
    errors = [r for r in rs if r['error']]
    return dict(content=key, n=len(sub), accuracy_in=1 - rate(sub, 'error'), accuracy_out=1 - rate(other, 'error'),
                share_of_errors=sum(r['content'] for r in errors) / len(errors) if errors else None,
                share_of_content_errors_cautious=(sum(r['cautious'] for r in errors if r['content']) / max(1, sum(r['content'] for r in errors))))


def run():
    joined, ans = answers(); thr = thresholds(); law = law_labels()
    phys = {x['id']: x for x in read(PHYSICIANS)}
    flip = {m: flips(m) for m in MODELS}
    rng = np.random.default_rng(SEED); result = {}
    for cohort in ['kormed', 'medqa']:
        result[cohort] = {}
        for model in MODELS:
            rs = rows(model, cohort, joined, ans, thr, flip[model], law, phys)
            keys = ['flip', 'sol', 'shared', 'panel_accuracy', 'luna_same_answer'] + (['law'] if cohort == 'kormed' else ['flagged', 'physicians_chose_model'])
            profile = {}
            for band, f in [('cautious', lambda r: not r['confident']), ('confident', lambda r: r['confident'])]:
                g = [r for r in rs if f(r) and r['error']]
                profile[band] = dict(n=len(g), flip_n=sum(r['flip'] is not None for r in g), **{k: rate(g, k) for k in keys})
            point = contrasts(rs); boots = defaultdict(list)
            for draw in rng.integers(0, len(rs), size=(NBOOT, len(rs))):
                for k, v in contrasts([rs[i] for i in draw]).items():
                    if v is not None:
                        boots[k].append(v)
            result[cohort][model] = dict(threshold=thr[model][cohort], errors=sum(r['error'] for r in rs), profile=profile,
                                         concentration=concentration(model, cohort, joined, ans, thr, law, phys),
                                         contrasts={k: dict(confident_minus_cautious=point[k], ci95=ci(boots[k])) for k in point})
        # Errors common to all three decision models, among questions all three answered validly.
        common = [q for q, i in joined.items() if i['cohort'] == cohort and all(ans[m][q] is not None for m in MODELS)]
        wrong_all = [q for q in common if all(not ans[m][q]['correct'] for m in MODELS)]
        same_wrong = [q for q in wrong_all if len({ans[m][q]['prediction'] for m in MODELS}) == 1]
        conf_all = [q for q in wrong_all if all(ans[m][q]['p'] > thr[m][cohort] for m in MODELS)]
        result[cohort]['shared'] = dict(questions=len(common), wrong_all_three=len(wrong_all), same_wrong_answer=len(same_wrong),
                                        confident_wrong_all_three=len(conf_all),
                                        sol_correct_on_wrong_all_three=sum(joined[q]['comparators']['sol']['correct'] for q in wrong_all),
                                        content_flag_on_wrong_all_three=sum((law[q] if cohort == 'kormed' else phys[q]['majority_concern']) for q in wrong_all))
    result['_settings'] = dict(seed=SEED, nboot=NBOOT, thresholds=thr)
    (OUTPUT / 'error-summary.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    return result


if __name__ == '__main__':
    r = run()
    for c in ['kormed', 'medqa']:
        print(c)
        for m in MODELS:
            x = r[c][m]; pc, pf = x['profile']['cautious'], x['profile']['confident']; k = x['concentration']
            f = lambda v: '-' if v is None else f'{v*100:.1f}'
            print(f"  {m:10s} errors {x['errors']} cautious {pc['n']} confident {pf['n']} | flip {f(pc['flip'])}/{f(pf['flip'])} "
                  f"diff {f(x['contrasts']['flip']['confident_minus_cautious'])} {[round(v*100, 1) for v in x['contrasts']['flip']['ci95']]} | "
                  f"sol fixes {f(pc['sol'])}/{f(pf['sol'])} diff {f(x['contrasts']['sol']['confident_minus_cautious'])} "
                  f"{[round(v*100, 1) for v in x['contrasts']['sol']['ci95']]} | panel acc {f(pc['panel_accuracy'])}/{f(pf['panel_accuracy'])} "
                  f"luna same {f(pc['luna_same_answer'])}/{f(pf['luna_same_answer'])} | {k['content']} n {k['n']} acc {f(k['accuracy_in'])} vs "
                  f"{f(k['accuracy_out'])} share of errors {f(k['share_of_errors'])} cautious share {f(k['share_of_content_errors_cautious'])}"
                  + (f" | phys chose model {f(pc['physicians_chose_model'])}/{f(pf['physicians_chose_model'])}" if c == 'medqa' else ''))
        print('  shared', r[c]['shared'])
