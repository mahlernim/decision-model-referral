"""Offline analysis of the frozen initial KorMedMCQA study."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter

import numpy as np

from .common import ROOT, digest, filehash, frozen, now, read, write_new
from .study import DIRECTORY, OUTPUT, SEED, manifest, all_records
from .report import wilson


def auc(labels, scores):
    """Mann-Whitney AUROC, ties receive half credit."""
    labels = np.asarray(labels, dtype=int); scores = np.asarray(scores, dtype=float)
    n1 = labels.sum(); n0 = len(labels) - n1
    if not n1 or not n0: return None
    order = np.argsort(scores, kind='stable'); s = scores[order]; y = labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]; ends = np.r_[starts[1:], len(s)]
    ranks = np.repeat((starts + 1 + ends) / 2, ends - starts)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ci(values):
    x = [v for v in values if v is not None and math.isfinite(v)]
    return [float(v) for v in np.quantile(x, [.025, .975])] if x else None


def bootstrap_auc(labels, scores, nboot=4000):
    rng = np.random.default_rng(SEED); n = len(labels)
    y = np.asarray(labels); s = np.asarray(scores)
    return ci([auc(y[ix], s[ix]) for ix in rng.integers(0, n, size=(nboot, n))])


def paired_ci(a, b, nboot=4000):
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(SEED)
    samples = d[rng.integers(0, len(d), size=(nboot, len(d)))].mean(axis=1)
    return {'difference': float(d.mean()), 'ci95': ci(samples)}


def mcnemar(a, b):
    lost = int(sum(x and not y for x, y in zip(a, b))); won = int(sum(y and not x for x, y in zip(a, b))); n = lost + won
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(lost, won) + 1)) / 2 ** n) if n else 1.
    return {'a_correct_b_wrong': lost, 'a_wrong_b_correct': won, 'exact_two_sided_p': p}


def good_records():
    rr = all_records()
    return {(r['role'], r['item_id'], r['repeat'], r.get('selected_for_review')): r for r in rr if r['status'] == 'success'}


def probability(r):
    v = r['value']; return v['probabilities'][v['prediction']]


def freeze_threshold():
    m = manifest(); good = good_records()
    if (DIRECTORY / 'threshold.json').exists():
        saved = read(DIRECTORY / 'threshold.json')
        if saved['manifest_hash'] != m['sha256']: raise ValueError('Threshold manifest mismatch')
        for uid, h in saved['development_response_hashes'].items():
            if digest(good[('jev', uid, 0, None)]) != h: raise ValueError('Threshold development evidence changed')
        return saved
    dev = [i for i in m['items'] if i['split'] == 'dev']
    values = []
    for i in dev:
        r = good.get(('jev', i['id'], 0, None))
        if r is None: raise RuntimeError('Development Jev responses incomplete')
        values.append(probability(r))
    if any(r['split'] == 'test' and r['role'] in ['jev', 'comparator'] for r in all_records()):
        if not (DIRECTORY / 'threshold.json').exists(): raise RuntimeError('Cannot prospectively freeze after test inference')
        return read(DIRECTORY / 'threshold.json')
    v = {'created_at': now(), 'manifest_hash': m['sha256'], 'n_dev': len(dev),
         'quantile': .2, 'method': 'numpy linear', 'threshold': float(np.quantile(values, .2)),
         'rule': 'route selected probability <= threshold',
         'development_referral_count': int(sum(x <= np.quantile(values, .2) for x in values)),
         'development_response_hashes': {i['id']: digest(good[('jev', i['id'], 0, None)]) for i in dev}}
    frozen(DIRECTORY / 'threshold.json', v)
    frozen(OUTPUT / 'threshold.json', v)
    return v


def routing(j, c, route):
    j = np.asarray(j, dtype=bool); c = np.asarray(c, dtype=bool); route = np.asarray(route, dtype=bool)
    combined = np.where(route, c, j); n = len(j); k = int(route.sum())
    rescued = int((route & ~j & c).sum()); lost = int((route & j & ~c).sum())
    return {'n': n, 'referred': k, 'referral_rate': k / n, 'correct': int(combined.sum()),
        'accuracy': float(combined.mean()), 'ci95': wilson(int(combined.sum()), n),
        'rescued': rescued, 'lost': lost, 'net_correct': rescued - lost,
        'random_referral_expected_accuracy': float(j.mean() + k / n * (c.mean() - j.mean())),
        'retained_n': n - k, 'retained_errors': int((~route & ~j).sum()),
        'retained_error_rate': float((~route & ~j).sum() / (n - k)) if k < n else None,
        'captured_jev_errors': int((route & ~j).sum()),
        'error_capture_fraction': float((route & ~j).sum() / (~j).sum()) if (~j).sum() else None,
        'vs_jev': paired_ci(j, combined)}


def analyze():
    m = manifest(); rr = all_records(); good = good_records()
    threshold = read(DIRECTORY / 'threshold.json')
    test = [i for i in m['items'] if i['split'] == 'test']
    rows = []
    for i in test:
        j = good.get(('jev', i['id'], 0, None)); c = good.get(('comparator', i['id'], 0, None))
        if not j or not c: continue
        v = j['value']; probabilities = v['probabilities']; ps = sorted(probabilities.values(), reverse=True)
        tags = good.get(('tags', i['id'], 0, None), {}).get('value', {})
        rows.append({'id': i['id'], 'year': i['year'], 'gold': i['gold'], 'pilot_exposed': i['pilot_exposed'],
            'jev_answer': v['prediction'], 'comparator_answer': c['value']['prediction'],
            'jev_correct': v['correct'], 'comparator_correct': c['value']['correct'],
            'selected_probability': probability(j), 'reported_confidence': v['rank_confidence'],
            'margin': ps[0] - ps[1], 'entropy': -sum(p * math.log(p) for p in ps if p > 0) / math.log(5),
            'brier': v['brier'], 'log_loss': v['log_loss'],
            'raw_brier': v['raw_vector_brier'], 'raw_log_loss': v['raw_gold_log_loss'],
            'probability_sum': v['reported_probability_sum'], 'renormalized': v['probabilities_renormalized'],
            'jev_cost': v['estimated_cost_usd'], 'comparator_cost': c['estimated_cost_usd'],
            'jev_latency_ms': j['latency_ms'], 'comparator_latency_ms': c['latency_ms'],
            'tags': tags})
    if not rows: raise RuntimeError('No paired test responses')
    j = np.array([r['jev_correct'] for r in rows]); c = np.array([r['comparator_correct'] for r in rows])
    p = np.array([r['selected_probability'] for r in rows]); n = len(rows)
    result = {'generated_at': now(), 'manifest_hash': m['sha256'], 'planned_test': len(test), 'paired_test': n,
              'threshold': threshold['threshold'], 'models': {}, 'counts': dict(Counter(r['role'] + '/' + r['split'] + '/' + r['status'] for r in rr)),
              'budget_upper_usd': sum(r['budget_charge_usd'] for r in rr),
              'estimated_cost_usd': sum(r.get('estimated_cost_usd', r.get('value', {}).get('estimated_cost_usd', r['budget_charge_usd'])) for r in rr)}
    for role, answers in [('jev', j), ('comparator', c)]:
        records = [r for r in rr if r['role'] == role and r['split'] == 'test' and r['repeat'] == 0 and r['status'] == 'success']
        correct = sum(r['value']['correct'] for r in records)
        result['models'][role] = {'model': sorted({r['raw_response']['model'] for r in records}),
            'successful': len(records), 'correct': correct, 'operational_accuracy': correct / len(test),
            'operational_ci95': wilson(correct, len(test)), 'successful_accuracy': correct / len(records),
            'median_latency_ms': float(np.median([r['latency_ms'] for r in records])),
            'p95_latency_ms': float(np.quantile([r['latency_ms'] for r in records], .95))}
    result['comparison'] = {'paired_difference': paired_ci(j, c), **mcnemar(j, c),
        'both_correct': int((j & c).sum()), 'both_wrong': int((~j & ~c).sum()),
        'same_wrong_option': sum(not r['jev_correct'] and not r['comparator_correct'] and r['jev_answer'] == r['comparator_answer'] for r in rows),
        'recoverable_jev_errors': int((~j & c).sum())}
    result['uncertainty'] = {}
    for name, sign in [('selected_probability', -1), ('reported_confidence', -1), ('margin', -1), ('entropy', 1)]:
        x = np.array([r[name] for r in rows]); scores = sign * x
        result['uncertainty'][name] = {'error_detection_auc': auc(~j, scores), 'ci95': bootstrap_auc(~j, scores),
            'correct_median': float(np.median(x[j])) if j.any() else None,
            'wrong_median': float(np.median(x[~j])) if (~j).any() else None}
    result['recoverable_error_detection'] = {'auc': auc(~j & c, 1 - p), 'ci95': bootstrap_auc(~j & c, 1 - p)}
    result['calibration'] = {'multiclass_brier': float(np.mean([r['brier'] for r in rows])),
        'log_loss': float(np.mean([r['log_loss'] for r in rows])),
        'raw_brier': float(np.mean([r['raw_brier'] for r in rows])),
        'raw_log_loss': float(np.mean([r['raw_log_loss'] for r in rows])),
        'renormalized_n': sum(r['renormalized'] for r in rows),
        'mean_selected_probability': float(p.mean()), 'accuracy': float(j.mean()), 'bins': []}
    edges = [0, .5, .7, .8, .9, .95, 1.00000001]
    ece = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi); k = int(mask.sum())
        if not k: continue
        a = float(j[mask].mean()); mp = float(p[mask].mean()); ece += k / n * abs(a - mp)
        result['calibration']['bins'].append({'lo': lo, 'hi': min(hi, 1), 'n': k, 'correct': int(j[mask].sum()),
            'mean_probability': mp, 'accuracy': a, 'ci95': wilson(int(j[mask].sum()), k)})
    result['calibration']['ece_fixed_bins'] = ece
    high = p >= .9
    result['calibration']['high_probability'] = {'cutoff': .9, 'n': int(high.sum()), 'errors': int((high & ~j).sum()),
        'error_rate': float((high & ~j).sum() / high.sum()) if high.any() else None}
    route = p <= threshold['threshold']
    result['routing_frozen'] = routing(j, c, route)
    result['routing_frozen']['threshold'] = threshold['threshold']
    hybrid = np.where(route, c, j); rng = np.random.default_rng(SEED)
    boot_lift = []
    for ix in rng.integers(0, n, size=(4000, n)):
        rate = route[ix].mean()
        boot_lift.append(float(hybrid[ix].mean() - (j[ix].mean() + rate * (c[ix].mean() - j[ix].mean()))))
    result['routing_frozen']['vs_random_expected'] = {'difference': float(hybrid.mean()) - result['routing_frozen']['random_referral_expected_accuracy'], 'ci95': ci(boot_lift)}
    costj = np.array([r['jev_cost'] for r in rows]); costc = np.array([r['comparator_cost'] for r in rows])
    latj = np.array([r['jev_latency_ms'] for r in rows]); latc = np.array([r['comparator_latency_ms'] for r in rows])
    result['routing_frozen']['simulated_cost_usd'] = float(costj.sum() + costc[route].sum())
    result['routing_frozen']['always_comparator_cost_usd'] = float(costc.sum())
    result['routing_frozen']['simulated_median_latency_ms'] = float(np.median(latj + route * latc))
    result['routing_frozen']['simulated_p95_latency_ms'] = float(np.quantile(latj + route * latc, .95))
    order = np.argsort(p, kind='stable') # rows are in frozen ID order, deterministic ties
    result['routing_budgets'] = []
    for fraction in [.1, .2, .3]:
        rmask = np.zeros(n, bool); rmask[order[:math.ceil(n * fraction)]] = True
        result['routing_budgets'].append({'target_fraction': fraction, **routing(j, c, rmask)})
    result['sensitivity'] = {}
    subsets = {'pilot_exposed': [r['pilot_exposed'] for r in rows], 'not_pilot_exposed': [not r['pilot_exposed'] for r in rows],
        'automated_complete_input': [r['tags'].get('input_status') == 'complete' for r in rows]}
    for year in sorted({r['year'] for r in rows}): subsets[f'year_{year}'] = [r['year'] == year for r in rows]
    for name, mask in subsets.items():
        mask = np.array(mask, dtype=bool); k = int(mask.sum())
        if k:
            result['sensitivity'][name] = {'n': k, 'jev_correct': int(j[mask].sum()), 'comparator_correct': int(c[mask].sum()),
                'jev_accuracy': float(j[mask].mean()), 'comparator_accuracy': float(c[mask].mean()),
                'jev_ci95': wilson(int(j[mask].sum()), k), 'auc': auc(~j[mask], 1 - p[mask]),
                'routing': routing(j[mask], c[mask], route[mask])}
    groups = {}
    for feature in ['task', 'laboratory', 'numeric', 'negative_wording', 'korean_law_policy', 'input_status']:
        values = sorted({str(r['tags'][feature]) for r in rows if feature in r['tags']})
        for value in values:
            subset = [r for r in rows if str(r['tags'].get(feature, 'missing')) == value]
            k = len(subset); errors = sum(not r['jev_correct'] for r in subset)
            groups[feature + '=' + value] = {'n': k, 'jev_errors': errors, 'jev_error_rate': errors / k,
                'error_ci95': wilson(errors, k), 'comparator_errors': sum(not r['comparator_correct'] for r in subset)}
    result['automated_tags'] = {'completed': sum(bool(r['tags']) for r in rows),
        'evidence_exact': sum(r['tags'].get('evidence_exact_match', False) for r in rows), 'groups': groups}
    repeats = []
    for uid in m['repeat_ids']:
        base = good.get(('jev', uid, 0, None)); variants = [good.get(('jev', uid, r, None)) for r in [1, 2]]
        if not base or not all(variants): continue
        tv = [sum(abs(base['value']['probabilities'][k] - v['value']['probabilities'][k]) for k in 'ABCDE') / 2 for v in variants]
        flips = [v['value']['prediction'] != base['value']['prediction'] for v in variants]
        repeats.append({'id': uid, 'flips': sum(flips), 'max_total_variation': max(tv),
                        'referral_flips': sum((probability(v) <= threshold['threshold']) != (probability(base) <= threshold['threshold']) for v in variants),
                        'raw_answers_identical': all(v['raw_response']['answers'] == base['raw_response']['answers'] for v in variants)})
    result['repeatability'] = {'items_complete': len(repeats), 'extra_calls': len(repeats) * 2,
        'items_with_flip': sum(r['flips'] > 0 for r in repeats), 'answer_flips': sum(r['flips'] for r in repeats),
        'items_with_exact_outputs': sum(r['raw_answers_identical'] for r in repeats),
        'referral_decision_flips': sum(r['referral_flips'] for r in repeats),
        'items_with_referral_flip': sum(r['referral_flips'] > 0 for r in repeats),
        'maximum_total_variation': max((r['max_total_variation'] for r in repeats), default=None)}
    reviews = [r for r in rr if r['role'] == 'review' and r['status'] == 'success']
    result['automated_reviews'] = {'completed_unique_wrong_options': len(reviews),
        'categories': dict(Counter(r['value']['category'] for r in reviews)),
        'key_concerns': sum(r['value']['key_concern'] for r in reviews),
        'evidence_exact': sum(r['value']['evidence_exact_match'] for r in reviews)}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (OUTPUT / 'item-results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows({**r, 'tags': json.dumps(r['tags'], ensure_ascii=False)} for r in rows)
    public_reviews = [{'item_id': r['item_id'], 'selected_option': r['selected_for_review'], **r['value']} for r in reviews]
    (OUTPUT / 'error-review.json').write_text(json.dumps(public_reviews, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result, rows


def percent(x): return f'{100*x:.1f}%'


def render(result, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    j = np.array([r['jev_correct'] for r in rows]); c = np.array([r['comparator_correct'] for r in rows]); p = np.array([r['selected_probability'] for r in rows]); n = len(rows)
    bins = result['calibration']['bins']; ax = axes[0, 0]
    means = [b['accuracy'] for b in bins]; errors = np.array([[b['accuracy'] - b['ci95'][0], b['ci95'][1] - b['accuracy']] for b in bins]).T
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    ax.errorbar([b['mean_probability'] for b in bins], means, yerr=errors, fmt='o-', color='#16697a', capsize=3)
    for b in bins: ax.annotate(f"n={b['n']}", (b['mean_probability'], b['accuracy']), xytext=(4, 6), textcoords='offset points', fontsize=8)
    ax.set(xlim=(0, 1.02), ylim=(0, 1.05), xlabel='Mean selected-answer probability', ylabel='Observed accuracy', title='A  Jev calibration (95% Wilson intervals)')
    ax = axes[0, 1]
    ax.hist([p[j], p[~j]], bins=np.linspace(0, 1, 21), label=['Correct', 'Incorrect'], color=['#16697a', '#c75b39'], alpha=.85)
    ax.set(xlabel='Selected-answer probability', ylabel='Question count', title='B  Does uncertainty identify errors?'); ax.legend(frameon=False)
    ax = axes[1, 0]; order = np.argsort(-p, kind='stable'); count = np.arange(1, n+1)
    risk = np.cumsum(~j[order]) / count
    ax.plot(count/n, risk, color='#16697a', label='Retain highest probability')
    ax.axhline((~j).mean(), linestyle='--', color='gray', label='Random retention, expected')
    ax.set(xlabel='Fraction of questions retained', ylabel='Error fraction among retained', title='C  Error versus coverage', ylim=(0, max(.4, float(risk.max()) + .03))); ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]; order = np.argsort(p, kind='stable'); gains = (c.astype(int) - j.astype(int))[order]
    accuracy = np.r_[j.mean(), j.mean() + np.cumsum(gains)/n]; fractions = np.arange(n+1)/n
    ax.plot(fractions, accuracy, color='#16697a', label='Refer lowest probability')
    ax.plot(fractions, j.mean() + fractions*(c.mean()-j.mean()), '--', color='gray', label='Random referral, expected')
    ax.axhline(c.mean(), color='#c75b39', linestyle=':', label='Always Sol')
    r = result['routing_frozen']; ax.scatter(r['referral_rate'], r['accuracy'], marker='D', color='black', label='Frozen development threshold')
    ax.set(xlabel='Fraction referred to Sol', ylabel='Combined accuracy', title='D  Offline escalation simulation', ylim=(max(0, min(j.mean(), c.mean())-.04), 1.01)); ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f'KorMedMCQA doctor test | {n} paired questions | exploratory in silico study', fontsize=13)
    for extension in ['png', 'svg', 'pdf']: fig.savefig(OUTPUT / f'overview.{extension}', dpi=170)
    plt.close(fig)
    r = result['routing_frozen']; comp = result['comparison']; cal = result['calibration']; unc = result['uncertainty']['selected_probability']; repeat = result['repeatability']
    lines = ['# Jev on Korean medical licensing questions', '',
        'Initial in silico study, September 17, 2026. Local research report. Not submitted or externally reviewed.', '',
        f"**{result['paired_test']} of {result['planned_test']} test questions have paired valid answers.** Official benchmark keys determine correctness. Model-assisted annotations are exploratory and have no clinician validation.", '',
        '## Main findings', '', '| Model | Correct / planned | Accuracy (95% CI) | Median / p95 request latency |', '|---|---:|---:|---:|']
    for role, label in [('jev', 'Jev 1.13.0'), ('comparator', 'GPT-5.6 Sol, reasoning none')]:
        v = result['models'][role]; lo, hi = v['operational_ci95']
        lines.append(f"| {label} | {v['correct']} / {result['planned_test']} | {percent(v['operational_accuracy'])} ({percent(lo)}, {percent(hi)}) | {v['median_latency_ms']:.0f} / {v['p95_latency_ms']:.0f} ms |")
    d = comp['paired_difference']; lines += ['', f"The paired Sol minus Jev accuracy difference is {100*d['difference']:+.1f} percentage points (95% question-bootstrap interval {100*d['ci95'][0]:+.1f} to {100*d['ci95'][1]:+.1f}). These are descriptive exploratory intervals.", '',
        '![Study overview](overview.png)', '',
        'Figure 1. Calibration uses fixed probability bins with Wilson intervals. Histograms count source questions. Coverage and escalation curves rank test probabilities, with deterministic ID tie-breaking, and are descriptive. The black diamond uses the independently frozen development threshold. Routing is simulated from separately recorded answers. Four concurrent requests were used, so latency includes contention and network effects.', '',
        '## Native probabilities and errors', '',
        f"Error-detection AUROC using one minus selected-answer probability is **{unc['error_detection_auc']:.3f}** (95% bootstrap interval {unc['ci95'][0]:.3f} to {unc['ci95'][1]:.3f}). Median selected probability is {unc['correct_median']:.3f} for correct answers and {unc['wrong_median']:.3f} for incorrect answers.", '',
        f"Mean selected probability is {cal['mean_selected_probability']:.3f}, compared with observed accuracy {cal['accuracy']:.3f}. Multiclass Brier score is {cal['multiclass_brier']:.4f} (sum convention, range 0 to 2), natural-log loss {cal['log_loss']:.4f}, and fixed-bin ECE {cal['ece_fixed_bins']:.4f}. Ranking and calibration answer different questions. A useful ranking does not establish accurate probability estimates.", '',
        f"At selected probability >= 0.90, there are {cal['high_probability']['errors']} errors among {cal['high_probability']['n']} answers. {cal['renormalized_n']} probability vectors required bounded normalization. Raw-vector Brier and log loss are {cal['raw_brier']:.4f} and {cal['raw_log_loss']:.4f}; raw responses are retained.", '',
        '| Uncertainty signal | Error-detection AUROC (95% CI) |', '|---|---:|']
    for name, v in result['uncertainty'].items(): lines.append(f"| {name} | {v['error_detection_auc']:.3f} ({v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}) |")
    lines += ['', '## Shared and recoverable errors', '',
        '| Outcome | Questions |', '|---|---:|',
        f"| Both correct | {comp['both_correct']} |", f"| Jev wrong, Sol correct | {comp['a_wrong_b_correct']} |",
        f"| Jev correct, Sol wrong | {comp['a_correct_b_wrong']} |", f"| Both wrong | {comp['both_wrong']} |",
        f"| Same wrong option (subset of both wrong) | {comp['same_wrong_option']} |", '',
        'Shared distractors are observable agreement, not evidence that the models reasoned the same way. The complementary errors determine whether escalation helps.', '',
        '## Frozen-threshold escalation', '',
        f"The 20th percentile of development selected probability was **{result['threshold']:.4f}**. Applying probability <= this threshold refers **{r['referred']}/{r['n']} ({percent(r['referral_rate'])})** test questions. Ties and split differences mean the referral fraction need not equal 20%.", '',
        f"The simulated workflow achieves **{percent(r['accuracy'])}** accuracy. It rescues {r['rescued']} Jev errors and loses {r['lost']} previously correct answers, a net change of {r['net_correct']:+d} answers. Random referral of the same number has expected accuracy {percent(r['random_referral_expected_accuracy'])}.", '',
        f"The gain over random expected referral is {100*r['vs_random_expected']['difference']:+.1f} percentage points (95% question-bootstrap interval {100*r['vs_random_expected']['ci95'][0]:+.1f} to {100*r['vs_random_expected']['ci95'][1]:+.1f}). This contrast measures the value of the ranking at the observed referral rate; it is not a randomized clinical comparison.", '',
        f"Estimated simulated API cost is ${r['simulated_cost_usd']:.4f}, compared with ${r['always_comparator_cost_usd']:.4f} for always using Sol. Simulated sequential median/p95 latency is {r['simulated_median_latency_ms']:.0f}/{r['simulated_p95_latency_ms']:.0f} ms, calculated by adding Jev and Sol request times on referred questions. This was not a live cascade latency measurement.", '',
        '| Test-ranked referral budget | Actual referred | Combined accuracy | Rescued | Lost | Random expected accuracy |', '|---|---:|---:|---:|---:|---:|']
    for v in result['routing_budgets']: lines.append(f"| {percent(v['target_fraction'])} | {v['referred']} | {percent(v['accuracy'])} | {v['rescued']} | {v['lost']} | {percent(v['random_referral_expected_accuracy'])} |")
    lines += ['', 'These test-ranked budgets describe discrimination only. They are not independent threshold validations.', '',
        '## Sensitivity analyses', '', '| Subset | n | Jev accuracy | Sol accuracy | Jev error AUROC |', '|---|---:|---:|---:|---:|']
    for name, v in result['sensitivity'].items(): lines.append(f"| {name} | {v['n']} | {percent(v['jev_accuracy'])} | {percent(v['comparator_accuracy'])} | {v['auc']:.3f} |")
    lines += ['', 'The original pilot exposed 100 selected questions before this study was designed. Remaining items are new to this evaluation, but public-benchmark training exposure is unknown. Automated complete-input selection is a sensitivity analysis, not a clinician-adjudicated cohort.', '',
        '## Automated question and error characterization', '',
        f"Terra assigned blinded tags to {result['automated_tags']['completed']} test questions. {result['automated_tags']['evidence_exact']} supplied excerpts match source text exactly. Exact quotation validates provenance, not correctness of the category. These annotations have no independent human validation.", '',
        '| Automated category | n | Jev errors | Error rate (95% CI) | Sol errors |', '|---|---:|---:|---:|---:|']
    for name, v in result['automated_tags']['groups'].items():
        lo, hi = v['error_ci95']; lines.append(f"| {name} | {v['n']} | {v['jev_errors']} | {percent(v['jev_error_rate'])} ({percent(lo)}, {percent(hi)}) | {v['comparator_errors']} |")
    ar = result['automated_reviews']
    lines += ['', f"Sol with low reasoning characterized {ar['completed_unique_wrong_options']} unique incorrect item/option pairs, with identities and confidence withheld. {ar['key_concerns']} were flagged for possible key or input concerns. No official keys were changed. Review uses the same model family as the comparator and is exploratory.", '',
        '[Readable error review](error-review.md) and [structured review records](error-review.json) contain the descriptions, evidence excerpts and uncertainty labels. These describe apparent distinctions, not hidden model reasoning or validated clinical harms.', '',
        '## Repeatability and cost', '',
        f"Across {repeat['items_complete']} randomly selected development items and {repeat['extra_calls']} additional identical calls, there were {repeat['answer_flips']} answer flips affecting {repeat['items_with_flip']} items. {repeat['items_with_exact_outputs']} items had exactly identical answer objects in both repeats. Maximum probability total-variation distance was {repeat['maximum_total_variation']:.3f}. Repeats are not independent questions.", '',
        f"Applying the subsequently frozen threshold to these same repeat responses changes {repeat['referral_decision_flips']} referral decisions across {repeat['items_with_referral_flip']} items. This is a descriptive check of routing sensitivity to repeat variation, not a separate threshold validation.", '',
        f"Total estimated new API charges across development, testing, repeats and annotation are **${result['estimated_cost_usd']:.4f}**. The conservative budget ledger is **${result['budget_upper_usd']:.4f}**, below the $5 authorization. Estimates use recorded usage rather than invoices; the conservative ledger includes possible cache-write pricing and reserves for unpriced failures.", '',
        '## Interpretation and limitations', '',
        'This study evaluates whether a single native probability vector can help allocate additional model calls. It does not establish clinical safety or calibrated clinical risk. Stronger-comparator performance is measured, not assumed. Limited sample size, public data exposure, historical keys, possible residual incomplete inputs, development/test shift, and automated annotations constrain interpretation. No retrieval or reasoning supplementation was evaluated.', '',
        'All results are exploratory. Question-level bootstrap intervals condition on this benchmark and do not capture uncertainty across model versions, clinical settings, or guideline eras. No multiplicity-adjusted claims are made. Full-set scores provide context for published results, but the original paper used five-shot prompting whereas this study used zero-shot typed answers.', '',
        '## Reproduction and evidence', '',
        '- [Frozen protocol](protocol.md) and [protocol metadata](protocol-lock.json)',
        '- [Frozen development threshold](threshold.json)',
        '- [Machine-readable summary](summary.json) and [item results](item-results.csv)',
        '- [Figure PDF](overview.pdf) and [SVG](overview.svg)',
        '- Local raw requests and responses are retained under `runs/kormed-study-v1/`. Inputs are fetched from the pinned licensed dataset and are not bundled in this report.', '',
        'Recompute without API calls using `python -m jevbench.study_analysis report` from the repository root. This report was not published or submitted to medRxiv.', '',
        '## References', '',
        '1. [KorMedMCQA](https://arxiv.org/abs/2403.01469)',
        '2. [TypeSafe confidence](https://docs.typesafe.ai/confidence)',
        '3. [Token probabilities and medical confidence, JMIR 2025](https://www.jmir.org/2025/1/e64348/)',
        '4. [Medical selective prediction, Nature 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10396962/)',
        '5. [KorMedMCQA answer-extraction issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/3103)', '']
    text = '\n'.join(lines); (OUTPUT / 'report.md').write_text(text, encoding='utf-8')
    import markdown
    rendered = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    (OUTPUT / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Jev KorMedMCQA initial study</title><style>body{max-width:1050px;margin:32px auto;padding:0 22px;font:16px/1.6 system-ui;color:#172b36}h1,h2{line-height:1.2}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #d8e1e5;text-align:left}th{background:#edf3f5}img{max-width:100%}a{color:#12687b}code{overflow-wrap:anywhere} @media(max-width:600px){table{display:block;overflow-x:auto}body{margin-top:16px}}</style><body>' + rendered + '</body></html>', encoding='utf-8')
    reviews = read(OUTPUT / 'error-review.json')
    by_id = {r['id']: r for r in rows}
    details = ['# Automated error characterization', '',
        'Generated by GPT-5.6 Sol with low reasoning, without model identities or probabilities. Official keys are retained. These interpretations have no clinician validation and do not reveal model reasoning. Model attribution below was joined after review.', '']
    for review in sorted(reviews, key=lambda r: (r['category'], r['item_id'], r['selected_option'])):
        row = by_id[review['item_id']]
        models = [name for name, field in [('Jev', 'jev_answer'), ('Sol', 'comparator_answer')] if row[field] == review['selected_option']]
        details += [f"## {review['item_id']} | {' and '.join(models)}", '',
            f"Selected {review['selected_option']}; official key {row['gold']}. Category `{review['category']}`. Annotation certainty `{review['certainty']}`. Key/input concern `{review['key_concern']}`.", '',
            review['description'], '', f"Source excerpt: {review['evidence']}", '',
            f"Exact source match: {review['evidence_exact_match']}. Jev selected-answer probability: {row['selected_probability']:.3f}.", '']
    (OUTPUT / 'error-review.md').write_text('\n'.join(details), encoding='utf-8')


def verify():
    m = manifest(); rr = all_records(); items = {i['id']: i for i in m['items']}
    checks = {'source_hashes': True, 'original_files_unchanged': True, 'records_integrity': True}
    for s in m['sources']:
        if filehash(ROOT / s['local']) != s['sha256']: raise ValueError('Source changed')
    for p, h in m['preserved_files'].items():
        if filehash(ROOT / p) != h: raise ValueError('Original evidence changed: ' + p)
    for r in rr:
        if r['manifest_hash'] != m['sha256']: raise ValueError('Foreign response')
        stem = r['job'] + '__a' + str(r['attempt']) + '.json'; intent = read(DIRECTORY / 'intents' / stem)
        if digest(intent['request']) != r['request_hash']: raise ValueError('Request changed')
        if r['status'] == 'success' and r['role'] in ['jev', 'comparator']:
            if r['value']['correct'] != (r['value']['prediction'] == items[r['item_id']]['gold']): raise ValueError('Scoring inconsistency')
    unresolved = [p.name for p in (DIRECTORY / 'intents').glob('*.json') if not (DIRECTORY / 'attempts' / p.name).exists()]
    checks['unresolved_intents'] = unresolved
    checks['budget_upper_usd'] = sum(r['budget_charge_usd'] for r in rr)
    if checks['budget_upper_usd'] > 5: raise ValueError('Budget exceeded')
    threshold = read(DIRECTORY / 'threshold.json')
    test_times = [r['started_at'] for r in rr if r['split'] == 'test' and r['role'] in ['jev', 'comparator']]
    checks['threshold_before_test'] = bool(test_times) and threshold['created_at'] < min(test_times)
    checks['manifest_before_inference'] = m['created_at'] < min(r['started_at'] for r in rr)
    good = good_records()
    for uid, h in threshold['development_response_hashes'].items():
        if digest(good[('jev', uid, 0, None)]) != h: raise ValueError('Threshold evidence changed')
    checks['threshold_development_hashes'] = True
    by_content = {}
    duplicates = []
    for item in m['items']:
        key = digest({'question': item['question'], 'options': item['options']})
        if key in by_content: duplicates.append([by_content[key], item['id']])
        by_content[key] = item['id']
    checks['exact_content_duplicates'] = duplicates
    checks['completion'] = {}
    for split in ['dev', 'test']:
        expected = [i for i in m['items'] if i['split'] == split]
        for role in ['jev', 'comparator']:
            checks['completion'][role + '/' + split] = {'expected': len(expected), 'successful': sum((role, i['id'], 0, None) in good for i in expected)}
    expected_review = {(r['item_id'], r['value']['prediction']) for r in rr if r['split'] == 'test' and r['repeat'] == 0 and r['role'] in ['jev', 'comparator'] and r['status'] == 'success' and not r['value']['correct']}
    checks['completion']['review'] = {'expected': len(expected_review), 'successful': sum(('review', uid, 0, selected) in good for uid, selected in expected_review)}
    checks['completion']['tags'] = {'expected': 435, 'successful': sum(('tags', i['id'], 0, None) in good for i in m['items'] if i['split'] == 'test')}
    checks['completion']['repeat'] = {'expected': 60, 'successful': sum(('jev', uid, r, None) in good for uid in m['repeat_ids'] for r in [1,2])}
    checks['record_count'] = len(rr); checks['generated_at'] = now()
    (OUTPUT / 'verification.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(checks, indent=2))


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['threshold', 'report', 'verify']); a = p.parse_args()
    if a.command == 'threshold': print(json.dumps(freeze_threshold(), ensure_ascii=False)[:400])
    elif a.command == 'verify': verify()
    else:
        result, rows = analyze(); render(result, rows)
        print(json.dumps({k: result[k] for k in ['paired_test', 'models', 'comparison', 'routing_frozen', 'repeatability', 'budget_upper_usd']}, indent=2))


if __name__ == '__main__': main()
