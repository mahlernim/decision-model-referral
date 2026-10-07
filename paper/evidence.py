"""Compute every number in the three-decision-model manuscript, offline, from the frozen source records.

Writes evidence.json, which validate.py checks against the manuscript and supplement.
First tiers: Jev 1.13.0, OpenAI Decisions, Clef and the GPT-6 Luna letter-probability baseline.
Reference configurations appear only as standalone points. GPT-6.1 Sol is the escalation destination.
The operating point refers the least confident 20% of each first tier's own test answers, which needs no labels.
Internal-only configurations (Claude Haiku 4.5, GPT-6.1 Sol medium) are never read.
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from jevbench import luna_logprob, luna_logprob_analysis  # noqa: E402
from jevbench.decision_models import manifest as dm_manifest  # noqa: E402
from jevbench.decision_models_analysis import terminal, tier_rows  # noqa: E402
from jevbench.decision_models_order_analysis import load as load_order  # noqa: E402
from jevbench.escalation_curve import curve  # noqa: E402
from jevbench.luna_logprob_analysis import calibration, scored  # noqa: E402
from jevbench.medical_order_analysis import estimate  # noqa: E402
from jevbench.study_analysis import auc, bootstrap_auc, ci  # noqa: E402

UNIFIED_CSV = ROOT / 'docs/comparator-refresh-v1/unified/item-results.csv'
UNIFIED_JSON = ROOT / 'docs/comparator-refresh-v1/unified/unified-results.json'
JOINED = ROOT / 'docs/jev-error-patterns-v1/joined-items.json'
PHYSICIANS = ROOT / 'docs/jev-error-patterns-v1/physician-annotation-join.json'
HARD = ROOT / 'docs/hard-items-v1/summary.json'
SOL_LABELS = ROOT / 'runs/sol-relabel-v1/attempts'
REVIEW = HERE / 'label-review/review-results.json'
FOLLOW_UP = HERE / 'label-review/follow-up-review.json'
DM_SUMMARY = ROOT / 'docs/decision-models-v1/summary.json'
COHORTS = ['kormed', 'medqa']
TIERS = ['jev', 'decisions', 'clef', 'luna_letter']
DECISION = ['jev', 'decisions', 'clef']
REFERENCE = ['sol', 'opus', 'gemini', 'lunamax', 'sonnet', 'luna']
STRONG = ['sol', 'opus', 'gemini', 'lunamax']
SHARES = [10, 20, 30, 50]
SHARE = 20
SEED = 2026100705
NBOOT = 4000


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_tiers():
    """Per cohort: first-tier rows shaped like the unified item results, option probabilities, and reference rows."""
    unified = defaultdict(dict)
    with UNIFIED_CSV.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r['configuration_id'] in ['jev'] + REFERENCE:
                unified[(r['cohort'], r['configuration_id'])][r['question_id']] = r
    m = dm_manifest()
    latest = {model: terminal(model, m) for model in ('decisions', 'clef')}
    lm = luna_logprob.manifest(); ll = luna_logprob_analysis.terminal_records(lm)
    out = {}
    for c in COHORTS:
        test = [i for i in m['items'] if i['cohort'] == c and i['split'] == 'test']
        gold = {i['id']: i['gold'] for i in test}
        rows, probs, preds, native = {}, {}, {}, {}
        rows['jev'] = unified[(c, 'jev')]
        native['jev'] = {q: float(r['provider_confidence']) for q, r in rows['jev'].items() if r['valid'] == 'True'}
        probs['jev'] = {q: json.loads(r['option_probabilities_json']) for q, r in rows['jev'].items() if r['valid'] == 'True'}
        preds['jev'] = {q: r['predicted_option'] for q, r in rows['jev'].items() if r['valid'] == 'True'}
        for model in ('decisions', 'clef'):
            rows[model], probs[model] = tier_rows(model, latest[model], test)
            preds[model] = {i['id']: latest[model][f"{model}__{i['id']}"]['value']['answer'] for i in test
                            if latest[model][f"{model}__{i['id']}"].get('value')}
            native[model] = {i['id']: latest[model][f"{model}__{i['id']}"]['value']['native_confidence'] for i in test
                             if latest[model][f"{model}__{i['id']}"].get('value')}
        luna_test = [i for i in lm['items'] if i['cohort'] == c and i['split'] == 'test']
        rows['luna_letter'] = luna_logprob_analysis.first_tier(c, ll, luna_test)
        probs['luna_letter'] = {i['id']: ll[f"luna__{i['id']}"]['value']['probabilities'] for i in luna_test if ll[f"luna__{i['id']}"].get('value')}
        preds['luna_letter'] = {i['id']: ll[f"luna__{i['id']}"]['value']['answer'] for i in luna_test if ll[f"luna__{i['id']}"].get('value')}
        for t in TIERS:
            assert set(rows[t]) == set(gold), t
        out[c] = dict(gold=gold, rows=rows, probs=probs, preds=preds, native=native, ref={k: unified[(c, k)] for k in REFERENCE})
    return out


def arrays(rows, sol, ids=None):
    """Valid first-tier answers as arrays: probability, error, gain from replacing with Sol, Sol cost."""
    ids = sorted(rows) if ids is None else ids
    v = [q for q in ids if rows[q]['valid'] == 'True']
    p = np.array([float(rows[q]['confidence']) for q in v])
    err = np.array([rows[q]['correct'] != 'True' for q in v], dtype=float)
    gain = np.array([(sol[q]['correct'] == 'True') - (rows[q]['correct'] == 'True') for q in v], dtype=float)
    base = sum(rows[q]['correct'] == 'True' for q in ids)
    return p, err, gain, base, len(ids)


def expected_top(p, values, k):
    """Expected sum of values over the k lowest probabilities, ties taken in random order."""
    if k <= 0:
        return 0.
    order = np.argsort(p, kind='stable'); ps, vs = p[order], values[order]
    uniq, start = np.unique(ps, return_index=True)
    sums = np.add.reduceat(vs, start); sizes = np.diff(np.r_[start, len(ps)])
    cum = np.cumsum(sizes); j = int(np.searchsorted(cum, k))
    taken = cum[j - 1] if j else 0
    return float(sums[:j].sum() + (k - taken) * sums[j] / sizes[j]) if j < len(sizes) else float(sums.sum())


def at_share(p, err, gain, base, n, share):
    k = min(round(n * share / 100), len(p))
    errors = err.sum(); caught = expected_top(p, err, k)
    oracle_gain = np.sort(gain)[::-1][:k].clip(min=0).sum()
    random_gain = k / len(p) * gain.sum()
    conf_gain = expected_top(p, gain, k)
    return dict(k=k, errors=float(errors), caught=caught, capture=caught / errors, oracle_capture=min(k, errors) / errors,
                random_capture=k / len(p), accuracy=(base + conf_gain) / n, random_accuracy=(base + random_gain) / n,
                oracle_accuracy=(base + oracle_gain) / n,
                achievable=(conf_gain - random_gain) / (oracle_gain - random_gain) if oracle_gain > random_gain else None)


def first_tier_summary(c, data, sol):
    rows, probs, gold = data['rows'], data['probs'], data['gold']
    out = {}
    for t in TIERS:
        r = rows[t]; n = len(r)
        k = sum(x['correct'] == 'True' for x in r.values())
        sc = scored(r, probs[t], gold)
        err = [not x['correct'] for x in sc]; score = [1 - x['p'] for x in sc]
        p, e, g, base, _ = arrays(r, sol)
        cost = sum(float(x['standardized_cost_usd']) for x in r.values()) / n * 1000
        pts = curve(r, sol)
        out[t] = dict(n=n, correct=k, valid=len(sc), accuracy=k / n, accuracy_ci95=wilson(k, n), cost_per_1000=cost,
                      error_auroc=auc(err, score), error_auroc_ci95=bootstrap_auc(err, score),
                      ece=calibration(sc)['ece_fixed_bins'], brier=calibration(sc)['multiclass_brier'],
                      share_probability_one=calibration(sc)['share_selected_probability_one'],
                      threshold_20=float(np.quantile(p, .2)),
                      shares={str(s): at_share(p, e, g, base, n, s) for s in SHARES},
                      curve=[dict(share=x['share'], accuracy=x['accuracy'], cost_per_1000=x['cost_per_1000']) for x in pts],
                      capture_curve=[dict(share=kk / n, capture=expected_top(p, e, kk) / e.sum()) for kk in range(len(p) + 1)])
        out[t]['share_20_cost_per_1000'] = pts[min(round(n * .2), pts[-1]['k'])]['cost_per_1000']
    return out


def bootstrap_share(c, data, sol, rng):
    """Source-question bootstrap of capture, accuracy and gain over random at the 20% share, and AUROC differences."""
    ids = sorted(data['gold']); n = len(ids); rows = data['rows']
    boots = defaultdict(list)
    pairs = [('decisions', 'luna_letter'), ('jev', 'decisions'), ('jev', 'clef'), ('decisions', 'clef'), ('jev', 'luna_letter')]

    def roc(t, sample):
        v = [rows[t][q] for q in sample if rows[t][q]['valid'] == 'True']
        return auc([x['correct'] != 'True' for x in v], [1 - float(x['confidence']) for x in v])

    for draw in rng.integers(0, n, size=(NBOOT, n)):
        sample = [ids[i] for i in draw]
        for t in TIERS:
            p, e, g, base, _ = arrays(rows[t], sol, sample)
            s = at_share(p, e, g, base, n, SHARE)
            boots[f'{t}_capture'].append(s['capture']); boots[f'{t}_accuracy'].append(s['accuracy'])
            boots[f'{t}_gain'].append(s['accuracy'] - s['random_accuracy'])
        a = {t: roc(t, sample) for t in TIERS}
        for x, y in pairs:
            boots[f'auroc_{x}_minus_{y}'].append(a[x] - a[y])
    out = {k: ci(v) for k, v in boots.items()}
    point = {t: None for t in TIERS}
    for x, y in pairs:
        out[f'auroc_{x}_minus_{y}_estimate'] = roc(x, ids) - roc(y, ids)
    return out


def order_summary(thresholds):
    """Option-order stability with referral at each model's 20% share threshold from the main test collection."""
    out = {}
    for model in DECISION:
        values = load_order(model); out[model] = {}
        for c in COHORTS:
            nopt = 5 if c == 'kormed' else 4; thr = thresholds[c][model]
            ids = sorted({q for cc, q, _ in values if cc == c}); rows = []
            for q in ids:
                base, rep = values.get((c, q, 'rot0')), values.get((c, q, 'repeat'))
                rot = [values.get((c, q, f'rot{k}')) for k in range(1, nopt)]
                if base is None or rep is None or any(v is None for v in rot):
                    continue
                ref = lambda v: v['selected_probability'] <= thr
                flip = lambda a, b: float(a['semantic_prediction'] != b['semantic_prediction'])
                rflip = lambda a, b: float(ref(a) != ref(b))
                rows.append(dict(
                    answer_rotation=np.mean([flip(base, v) for v in rot]), answer_repeat=flip(base, rep),
                    referral_rotation=np.mean([rflip(base, v) for v in rot]), referral_repeat=rflip(base, rep),
                    accuracy_original=float(base['correct']), accuracy_rotation=np.mean([float(v['correct']) for v in rot])))
            o = dict(planned=len(ids), complete=len(rows))
            for key in ('answer', 'referral'):
                o[key] = dict(rotation=estimate([r[f'{key}_rotation'] for r in rows]),
                              repeat=estimate([r[f'{key}_repeat'] for r in rows]),
                              excess=estimate([r[f'{key}_rotation'] - r[f'{key}_repeat'] for r in rows]))
            o['accuracy_original'] = float(np.mean([r['accuracy_original'] for r in rows]))
            o['accuracy_rotation'] = float(np.mean([r['accuracy_rotation'] for r in rows]))
            out[model][c] = o
    return out


def flips(model):
    values = load_order(model); out = {}
    for (c, q, var) in list(values):
        if var != 'rot0':
            continue
        nopt = 5 if c == 'kormed' else 4
        base, rep = values[(c, q, 'rot0')], values.get((c, q, 'repeat'))
        rot = [values.get((c, q, f'rot{k}')) for k in range(1, nopt)]
        out[q] = None if base is None or rep is None or any(v is None for v in rot) else \
            float(np.mean([v['semantic_prediction'] != base['semantic_prediction'] for v in rot]))
    return out


def sol_labels():
    out = {}
    for p in sorted(SOL_LABELS.glob('*.json')):
        r = read(p)
        if r.get('terminal') and r.get('value'):
            out[r['job'].split('__')[1]] = r['value']['korean_law_policy']
    assert len(out) == 435
    return out


def kappa(pairs):
    n = len(pairs); po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe)


def labels(items, sol):
    terra = {q: items[q]['tags']['korean_law_policy'] for q in sol}
    review = read(REVIEW)['items']; follow = read(FOLLOW_UP)['items']
    author = [(r['author'] == 'Y', sol[r['item_id']]) for r in review]
    return dict(sol_positive=sum(sol.values()), n=len(sol), sol_terra_agree=sum(sol[q] == terra[q] for q in sol),
                review_n=len(review), review_terra_positive=sum(r['terra'] for r in review),
                review_random_negative=sum(not r['terra'] for r in review),
                author_sol_agree=sum(a == b for a, b in author), author_sol_kappa=kappa(author),
                author_yes_sol_no=sum(a and not b for a, b in author), author_no_sol_yes=sum(b and not a for a, b in author),
                follow_up_n=len(follow), follow_up_author_yes=sum(f['author'] == 'Y' for f in follow))


def errors(c, data, sol_rows, thresholds, flip, law, phys, items, rng):
    """Cautious (at or below the 20% share threshold) and confident errors, with the content and panel measures."""
    out = {}
    for t in DECISION:
        rows, preds = data['rows'][t], data['preds'][t]; thr = thresholds[c][t]; rs = []
        for q, r in rows.items():
            if r['valid'] != 'True':
                continue
            comp = items[q]['comparators']; pred = preds[q]
            x = dict(error=r['correct'] != 'True', cautious=float(r['confidence']) <= thr, flip=flip[t].get(q),
                     sol=sol_rows[q]['correct'] == 'True',
                     panel_accuracy=sum(comp[m]['correct'] for m in REFERENCE) / len(REFERENCE),
                     luna_same=comp['luna']['valid'] and comp['luna']['prediction'] == pred)
            if c == 'kormed':
                x['law'] = law[q]
            else:
                x['flagged'] = phys[q]['majority_concern']
                x['physicians_chose'] = sum(pred in rt['answers'] for rt in phys[q]['raters']) * 2 > phys[q]['n_raters']
            rs.append(x)
        keys = ['flip', 'sol', 'panel_accuracy', 'luna_same'] + (['law'] if c == 'kormed' else ['flagged', 'physicians_chose'])
        rate = lambda g, k: (lambda v: sum(v) / len(v) if v else None)([x[k] for x in g if x[k] is not None])
        bands = {}
        for name, cond in (('cautious', True), ('confident', False)):
            g = [x for x in rs if x['error'] and x['cautious'] == cond]
            bands[name] = dict(n=len(g), flip_n=sum(x['flip'] is not None for x in g), **{k: rate(g, k) for k in keys})

        def contrast(sample):
            err = [x for x in sample if x['error']]
            hi, lo = [x for x in err if not x['cautious']], [x for x in err if x['cautious']]
            return {k: (None if rate(hi, k) is None or rate(lo, k) is None else rate(lo, k) - rate(hi, k)) for k in ('flip', 'sol')}
        point = contrast(rs); boots = defaultdict(list)
        for draw in rng.integers(0, len(rs), size=(NBOOT, len(rs))):
            for k, v in contrast([rs[i] for i in draw]).items():
                if v is not None:
                    boots[k].append(v)
        out[t] = dict(threshold=thr, valid=len(rs), cautious_share=sum(x['cautious'] for x in rs) / len(rs), bands=bands,
                      cautious_minus_confident={k: dict(estimate=point[k], ci95=ci(boots[k])) for k in point})
    return out


def content(c, data, law, phys):
    """Accuracy inside and outside the content group for every configuration, on the full denominator."""
    ids = sorted(data['gold']); inside = [q for q in ids if (law[q] if c == 'kormed' else phys[q]['majority_concern'])]
    other = [q for q in ids if q not in set(inside)]
    out = dict(n_inside=len(inside), n_other=len(other))
    allrows = {**{t: data['rows'][t] for t in TIERS}, **data['ref']}
    for name, rows in allrows.items():
        ki = sum(rows[q]['correct'] == 'True' for q in inside); ko = sum(rows[q]['correct'] == 'True' for q in other)
        errs = len(ids) - ki - ko
        out[name] = dict(inside_correct=ki, other_correct=ko, inside_accuracy=ki / len(inside), other_accuracy=ko / len(other),
                         other_ci95=wilson(ko, len(other)), share_of_errors=(len(inside) - ki) / errs if errs else None)
    return out


def shared(c, data, law, phys):
    ids = [q for q in sorted(data['gold']) if all(data['rows'][t][q]['valid'] == 'True' for t in DECISION)]
    wrong = [q for q in ids if all(data['rows'][t][q]['correct'] != 'True' for t in DECISION)]
    same = [q for q in wrong if len({data['preds'][t][q] for t in DECISION}) == 1]
    flag = [q for q in wrong if (law[q] if c == 'kormed' else phys[q]['majority_concern'])]
    sol_right = [q for q in wrong if data['ref']['sol'][q]['correct'] == 'True']
    return dict(answered=len(ids), wrong_all=len(wrong), same_option=len(same), content=len(flag), sol_correct=len(sol_right))


def flag_sensitivity(data, phys):
    ids = sorted(data['gold']); clear = [q for q in ids if not phys[q]['majority_concern']]
    sol = data['ref']['sol']; out = dict(n=len(clear))
    for t in TIERS:
        rows = {q: data['rows'][t][q] for q in clear}
        k = sum(r['correct'] == 'True' for r in rows.values())
        v = [r for r in rows.values() if r['valid'] == 'True']
        p, e, g, base, n = arrays(rows, sol)
        s = at_share(p, e, g, base, n, SHARE)
        out[t] = dict(accuracy=k / len(clear), error_auroc=auc([r['correct'] != 'True' for r in v], [1 - float(r['confidence']) for r in v]),
                      capture=s['capture'], accuracy_20=s['accuracy'], random_accuracy_20=s['random_accuracy'],
                      oracle_accuracy_20=s['oracle_accuracy'])
    out['sol'] = sum(sol[q]['correct'] == 'True' for q in clear) / len(clear)
    return out


def native_confidence(data):
    """The separately reported confidence field against the selected-option probability, on the same valid answers."""
    out = {}
    for t in DECISION:
        rows = data['rows'][t]; ids = [q for q, r in rows.items() if r['valid'] == 'True']
        assert set(ids) == set(data['native'][t])
        p = np.array([float(rows[q]['confidence']) for q in ids]); nc = np.array([data['native'][t][q] for q in ids])
        correct = np.array([rows[q]['correct'] == 'True' for q in ids]); err = (~correct).astype(float)
        k = round(len(data['gold']) * SHARE / 100)
        res = dict(mean_difference=float(np.mean(nc - p)), identical=float(np.mean(np.abs(nc - p) < 1e-9)))
        for name, s in (('probability', p), ('confidence', nc)):
            sc = [dict(p=float(x), correct=bool(y), brier=0., log_loss=0.) for x, y in zip(s, correct)]
            res[name] = dict(error_auroc=auc(err, 1 - s), ece=calibration(sc)['ece_fixed_bins'],
                             capture=expected_top(s, err, k) / err.sum())
        out[t] = res
    return out


def development_drift():
    s = read(DM_SUMMARY); out = {}
    for c in COHORTS:
        out[c] = {t: dict(threshold=s[c][t]['threshold'], referred=s[c][t]['threshold_cascade']['referred'], n=s[c]['n'])
                  for t in DECISION}
    return out


def main():
    data = load_tiers(); unified = read(UNIFIED_JSON)
    items = {r['id']: r for r in read(JOINED)}
    phys = {r['id']: r for r in read(PHYSICIANS)}
    law = sol_labels(); rng = np.random.default_rng(SEED)
    ev = dict(cohorts={}, labels=labels(items, law))
    thresholds = {}
    for c in COHORTS:
        sol = data[c]['ref']['sol']
        e = dict(n=len(data[c]['gold']), first_tiers=first_tier_summary(c, data[c], sol))
        thresholds[c] = {t: e['first_tiers'][t]['threshold_20'] for t in TIERS}
        e['bootstrap'] = bootstrap_share(c, data[c], sol, rng)
        e['reference'] = {k: dict(name=unified['configurations'][k]['name'], correct=unified['cohorts'][c]['models'][k]['correct'],
                                  valid=unified['cohorts'][c]['models'][k]['valid_n'], accuracy=unified['cohorts'][c]['models'][k]['accuracy'],
                                  accuracy_ci95=unified['cohorts'][c]['models'][k]['ci95'],
                                  cost_per_1000=unified['cohorts'][c]['models'][k]['cost_per_1000']) for k in REFERENCE}
        e['content'] = content(c, data[c], law, phys)
        e['shared'] = shared(c, data[c], law, phys)
        e['native_confidence'] = native_confidence(data[c])
        ev['cohorts'][c] = e
    flip = {t: flips(t) for t in DECISION}
    for c in COHORTS:
        ev['cohorts'][c]['errors'] = errors(c, data[c], data[c]['ref']['sol'], thresholds, flip, law, phys, items, rng)
    ev['order'] = order_summary(thresholds)
    ev['flag_sensitivity'] = flag_sensitivity(data['medqa'], phys)
    ev['development_drift'] = development_drift()
    hard = read(HARD); m = hard['medqa']
    ev['consensus_missed'] = dict(n=m['n'], missed=m['consensus_missed'], same_wrong_option=m['same_wrong_option'],
                                  majority_concern=m['majority_concern'], no_endorsed_key=m['no_endorsed_key:majority'],
                                  physicians_endorse=m['physicians_endorse_consensus_option'])
    ev['sources'] = [p.relative_to(ROOT).as_posix() for p in [UNIFIED_CSV, UNIFIED_JSON, JOINED, PHYSICIANS, HARD, REVIEW, FOLLOW_UP, DM_SUMMARY]] + \
                    ['runs/decision-models-v1', 'runs/decision-models-order-v1', 'runs/medical-option-order-v1', 'runs/luna-logprob-v1',
                     SOL_LABELS.relative_to(ROOT).as_posix()]
    (HERE / 'evidence.json').write_text(json.dumps(ev, indent=1), encoding='utf-8')
    return ev


def brief(ev):
    f = lambda x: f'{100*x:.1f}'
    for c in COHORTS:
        e = ev['cohorts'][c]; b = e['bootstrap']; print(c)
        for t in TIERS:
            x = e['first_tiers'][t]; s = x['shares']['20']
            print(f"  {t:11s} acc {f(x['accuracy'])} AUROC {x['error_auroc']:.3f} ECE {x['ece']:.3f} p1 {f(x['share_probability_one'])} "
                  f"thr20 {x['threshold_20']:.3f} | 20%: capture {f(s['capture'])} {[round(100*v, 1) for v in b[t+'_capture']]} oracle {f(s['oracle_capture'])} | "
                  f"acc {f(s['accuracy'])} {[round(100*v, 1) for v in b[t+'_accuracy']]} random {f(s['random_accuracy'])} oracle {f(s['oracle_accuracy'])} "
                  f"achievable {None if s['achievable'] is None else round(100*s['achievable'])}% gain CI {[round(100*v, 2) for v in b[t+'_gain']]} cost {x['share_20_cost_per_1000']:.3f}")
        for k in [k for k in b if k.endswith('_estimate')]:
            print('  ', k, round(b[k], 3), [round(v, 3) for v in b[k.replace('_estimate', '')]])
        for t, x in e['errors'].items():
            bc, bf = x['bands']['cautious'], x['bands']['confident']
            print(f"  errors {t}: cautious share {f(x['cautious_share'])} cautious {bc['n']} confident {bf['n']} flip {bc['flip'] and f(bc['flip'])}/{bf['flip'] and f(bf['flip'])} "
                  f"sol {f(bc['sol'])}/{bf['sol'] and f(bf['sol'])} contrast {x['cautious_minus_confident']}")
        print('  shared', e['shared'])
    for t, o in ev['order'].items():
        for c, x in o.items():
            print('order', t, c, x['complete'], {k: (round(100*x[k]['excess']['mean'], 2), [round(100*v, 2) for v in x[k]['excess']['ci95']]) for k in ('answer', 'referral')},
                  round(100*x['referral']['rotation']['mean'], 2), round(100*x['referral']['repeat']['mean'], 2))
    print('flag', {k: v for k, v in ev['flag_sensitivity'].items()})


if __name__ == '__main__':
    brief(main())
