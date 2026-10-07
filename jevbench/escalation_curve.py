"""Offline Jev escalation curves: refer the lowest-probability share of questions to one destination.

Starting from Jev answers, questions are sorted by ascending selected-option probability and the
first k are replaced with the destination answer, for every k from 0 to all valid Jev answers.
Tied probabilities are handled by the expected value under random order within the tie, so each
curve is deterministic. Invalid Jev answers stay incorrect and are never referred, matching the
full-denominator cascade rule. Costs are standardized token-price equivalents per 1,000 questions:
Jev on every question plus the destination on referred questions.
"""
import csv
import json
from collections import defaultdict
from .common import ROOT, filehash

SOURCE = ROOT / 'docs/comparator-refresh-v1/unified/item-results.csv'
SOL_MEDIUM = ROOT / 'docs/sol-medium-v1/item-results.csv'
OUT = ROOT / 'docs/escalation-curve-v1'
COHORTS = [('kormed', 'KorMedMCQA (Korean), n = 435'), ('medqa', 'MedQA (English), n = 1,273')]
# Curves are still computed and tabulated for these, but they are internal-only and never plotted.
INTERNAL_ONLY = {'haiku', 'solmed'}
# Plotted destinations with fixed colors; this six-hue set passes the all-pairs palette check,
# and every line is also labeled directly.
ALL = [('sol', 'GPT-6.1 Sol low', '#2a78d6'), ('opus', 'Claude Opus 5.5 low', '#e34948'),
       ('gemini', 'Gemini 3.8 Flash low', '#4a3aa7'), ('lunamax', 'GPT-6 Luna max', '#eda100'),
       ('sonnet', 'Claude Sonnet 5.5 low', '#008300'), ('luna', 'GPT-6 Luna none', '#1baf7a'),
       ('solmed', 'GPT-6.1 Sol medium', None), ('haiku', 'Claude Haiku 4.5 off', None)]
THRESHOLDS = {'kormed': 0.626, 'medqa': 0.770}
MARKS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
MARKER_SHARE = 20


def load():
    rows = defaultdict(dict)
    for source in [SOURCE, SOL_MEDIUM]:
        if not source.exists():
            continue
        with source.open(encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                rows[(r['cohort'], r['configuration_id'])][r['question_id']] = r
    return rows


def curve(jev, dest):
    n = len(jev)
    assert set(jev) == set(dest)
    base_correct = sum(r['valid'] == 'True' and r['correct'] == 'True' for r in jev.values())
    base_cost = sum(float(r['standardized_cost_usd']) for r in jev.values())
    groups = defaultdict(list)
    for q, r in jev.items():
        if r['valid'] != 'True':
            continue
        gain = (dest[q]['correct'] == 'True') - (r['correct'] == 'True')
        groups[float(r['confidence'])].append((gain, float(dest[q]['standardized_cost_usd'])))
    points = [dict(k=0, p_max=None, correct=base_correct, cost=base_cost)]
    correct, cost, k = base_correct, base_cost, 0
    for p in sorted(groups):
        items = groups[p]
        g = sum(i[0] for i in items) / len(items)
        c = sum(i[1] for i in items) / len(items)
        for _ in items:
            k += 1; correct += g; cost += c
            points.append(dict(k=k, p_max=p, correct=correct, cost=cost))
    for pt in points:
        pt.update(share=pt['k'] / n, accuracy=pt['correct'] / n, cost_per_1000=pt['cost'] / n * 1000)
    return points


def standalone(rows):
    n = len(rows)
    return dict(accuracy=sum(r['correct'] == 'True' for r in rows.values()) / n,
                cost_per_1000=sum(float(r['standardized_cost_usd']) for r in rows.values()) / n * 1000)


def at_share(points, n, pct):
    k = min(round(n * pct / 100), points[-1]['k'])
    return points[k]


def build():
    rows = load()
    result = {}
    for cohort, _ in COHORTS:
        jev = rows[(cohort, 'jev')]
        n = len(jev)
        threshold_k = sum(r['valid'] == 'True' and float(r['confidence']) <= THRESHOLDS[cohort] for r in jev.values())
        entry = dict(n=n, threshold=THRESHOLDS[cohort], threshold_referred=threshold_k,
                     standalone={'jev': standalone(jev)}, curves={})
        for dest, *_ in ALL:
            if (cohort, dest) not in rows:
                continue
            entry['standalone'][dest] = standalone(rows[(cohort, dest)])
            entry['curves'][dest] = curve(jev, rows[(cohort, dest)])
        result[cohort] = entry
    return result


def check(result):
    # Frozen-threshold operating points must reproduce the unified cascade table.
    expected = {('kormed', 'sol'): 409, ('kormed', 'opus'): 409, ('medqa', 'sol'): 1207, ('medqa', 'opus'): 1206,
                ('kormed', 'luna'): 395, ('kormed', 'lunamax'): 407, ('kormed', 'gemini'): 407,
                ('kormed', 'haiku'): 372, ('kormed', 'sonnet'): 400, ('medqa', 'luna'): 1152,
                ('medqa', 'lunamax'): 1198, ('medqa', 'gemini'): 1205, ('medqa', 'haiku'): 1105,
                ('medqa', 'sonnet'): 1187}
    for (cohort, dest), correct in expected.items():
        e = result[cohort]
        pt = e['curves'][dest][e['threshold_referred']]
        assert abs(pt['correct'] - correct) < 1e-9, (cohort, dest, pt['correct'])


def plot(result, stem=None, publication=False):
    """Working figure by default; publication=True drops the embedded title and footnote for use with a caption."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    ink, muted, grid = '#1f2328', '#5b636a', '#e3e5e6'
    present = [d for d in ALL if d[0] not in INTERNAL_ONLY]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2) if publication else (14, 6.4), sharey=True)
    fig.subplots_adjust(left=.07, right=.99, top=.78 if publication else .79, bottom=.12 if publication else .17, wspace=.05)
    for ax, (cohort, title) in zip(axes, COHORTS):
        e = result[cohort]
        n = e['n']
        for dest, label, color in present:
            pts = e['curves'][dest]
            ax.plot([p['cost_per_1000'] for p in pts], [p['accuracy'] * 100 for p in pts], color=color, lw=1.9, zorder=2)
            m = at_share(pts, n, MARKER_SHARE)
            ax.scatter(m['cost_per_1000'], m['accuracy'] * 100, s=46, color=color, edgecolor='white', linewidth=1.3, zorder=4)
        j = e['standalone']['jev']
        ax.scatter(j['cost_per_1000'], j['accuracy'] * 100, s=55, marker='D', color='#6f7a80', edgecolor='white', zorder=5)
        ax.annotate('Jev alone', (j['cost_per_1000'], j['accuracy'] * 100), xytext=(7, -4), textcoords='offset points',
                    fontsize=9, color=muted, va='top')
        ax.set_xscale('log')
        ax.set_xlim(.02, 5)
        ax.set_ylim(85.5, 100)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
        ax.set_title(title, fontsize=12, color=ink, loc='left')
        ax.set_xlabel('Estimated USD per 1,000 questions (log scale)', fontsize=10.5, color=ink)
        ax.grid(color=grid, lw=.7); ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#9aa1a6')
        ax.tick_params(colors=muted, labelsize=9.5)
    axes[0].set_ylabel('Accuracy (%)', fontsize=10.5, color=ink)
    if not publication:
        fig.suptitle('Accuracy and cost when escalating the least confident Jev answers', x=.06, ha='left',
                     fontsize=15, color=ink, y=.97)
    handles = [Line2D([], [], color=color, lw=2.2, label=f'Jev → {label}') for _, label, color in present]
    handles += [Line2D([], [], marker='o', color='#9aa1a6', markeredgecolor='white', ls='', markersize=8,
                       label=f'{MARKER_SHARE}% of questions escalated'),
                Line2D([], [], marker='D', color='#6f7a80', markeredgecolor='white', ls='', markersize=7, label='Jev alone')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.065, .995) if publication else (.055, .935),
               ncol=3 if publication else 4, frameon=False, fontsize=9.5)
    if not publication:
        fig.text(.06, .03, 'Each line starts at Jev alone and escalates questions in order of ascending Jev selected-option '
                 'probability, ending with every valid Jev answer replaced.\nFull planned denominators; invalid answers count as '
                 'incorrect. Costs are standardized token-price estimates at 30 September 2026 rates.',
                 fontsize=8.8, color=muted, linespacing=1.5)
    stem = stem or OUT / 'escalation-curve'
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(f'{stem}.{ext}', dpi=300 if publication else 200)
    plt.close(fig)


def write_table(result):
    with (OUT / 'escalation-curve.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cohort', 'destination', 'escalated', 'escalated_share', 'jev_p_at_or_below', 'expected_correct',
                    'accuracy', 'usd_per_1000'])
        for cohort, e in result.items():
            for dest, pts in e['curves'].items():
                for p in pts:
                    w.writerow([cohort, dest, p['k'], f"{p['share']:.6f}", '' if p['p_max'] is None else p['p_max'],
                                f"{p['correct']:.4f}", f"{p['accuracy']:.6f}", f"{p['cost_per_1000']:.6f}"])
    summary = {'sources': {p.relative_to(ROOT).as_posix(): filehash(p) for p in [SOURCE, SOL_MEDIUM] if p.exists()}, 'cohorts': {}}
    for cohort, e in result.items():
        s = {'n': e['n'], 'threshold': e['threshold'], 'threshold_referred': e['threshold_referred'],
             'standalone': e['standalone'], 'by_share': {}}
        for dest, pts in e['curves'].items():
            s['by_share'][dest] = {m: {k: at_share(pts, e['n'], m)[k] for k in ['k', 'accuracy', 'cost_per_1000']}
                                   for m in [0] + MARKS}
        summary['cohorts'][cohort] = s
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = build()
    check(result)
    plot(result)
    write_table(result)


if __name__ == '__main__':
    main()
