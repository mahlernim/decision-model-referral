"""Publication figures from evidence.json, drawn at final print size (6.9 in wide) so fonts print at their stated size.

Figure 1. Errors caught by referring each model's least confident answers, with random and oracle references.
Figure 2. Accuracy and cost when referring the least confident answers to GPT-6.1 Sol, with reference models.
Figure 3. Answer and referral changes under option rotation and identical repetition.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures'
WIDTH = 6.9
INK, MUTED, GRID, AXIS = '#1f2328', '#4f575e', '#e8eaec', '#9aa1a6'
ORACLE, RANDOM, REPEAT = '#6f777d', '#b9bec2', '#c9cdd0'
COLORS = {'jev': '#2a78d6', 'decisions': '#c4521f', 'clef': '#1a8a5a', 'luna_letter': '#6b5aa6'}
NAMES = {'jev': 'Jev', 'decisions': 'OpenAI Decisions', 'clef': 'Clef', 'luna_letter': 'GPT-6 Luna letter probabilities'}
SHORT_NAMES = {**NAMES, 'luna_letter': 'Luna letter probabilities'}
COHORTS = [('kormed', 'Korean (KorMedMCQA), n = 435'), ('medqa', 'English (MedQA), n = 1,273')]
TIERS = ['jev', 'decisions', 'clef', 'luna_letter']
DECISION = ['jev', 'decisions', 'clef']

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.edgecolor': AXIS, 'axes.linewidth': .6, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.labelcolor': INK, 'text.color': INK, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
})


def style(ax, grid_axis='both'):
    ax.grid(axis=grid_axis, color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


def save(fig, stem):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=600 if ext == 'png' else None, facecolor='white')
    plt.close(fig)


def figure1(ev):
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 3.85), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.99, top=.825, bottom=.11, wspace=.1, hspace=.6)
    for row, (cohort, title) in enumerate(COHORTS):
        for col, t in enumerate(TIERS):
            ax = axes[row, col]; x = ev['cohorts'][cohort]['first_tiers'][t]
            cc = x['capture_curve']; s = x['shares']['20']
            share_e = s['errors'] / x['n']
            ax.plot([0, 100], [0, 100], color=RANDOM, lw=1, ls=(0, (2.5, 2)), zorder=2)
            ax.plot([0, 100 * share_e, 100], [0, 100, 100], color=ORACLE, lw=1, zorder=2)
            ax.plot([100 * p['share'] for p in cc], [100 * p['capture'] for p in cc], color=COLORS[t], lw=1.6, zorder=3)
            ax.axvline(20, color=AXIS, lw=.6, ls=(0, (1, 2)), zorder=1)
            ax.scatter(20, 100 * s['capture'], s=24, color='white', edgecolor=COLORS[t], lw=1.3, zorder=4)
            ax.annotate(f"{100*s['capture']:.0f}%", (20, 100 * s['capture']), xytext=(5, -9), textcoords='offset points',
                        fontsize=7.5, color=COLORS[t], fontweight='bold')
            ax.set_xlim(0, 60); ax.set_ylim(0, 104)
            ax.xaxis.set_major_locator(FixedLocator([0, 20, 40, 60])); ax.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
            style(ax)
            ax.set_title(SHORT_NAMES[t], loc='left', fontweight='bold', color=COLORS[t], pad=3)
            if col == 0:
                ax.set_ylabel('Errors caught (%)')
            if row == 1:
                ax.set_xlabel('Answers referred (%)')
        top = axes[row, 0].get_position().y1
        fig.text(.012, top + .052, f"{'AB'[row]}", fontsize=10, fontweight='bold', va='bottom')
        fig.text(.04, top + .052, title, fontsize=8.5, fontweight='bold', va='bottom')
    handles = [Line2D([], [], color=INK, lw=1.6, label='Least confident answers first'),
               Line2D([], [], color=ORACLE, lw=1, label='Oracle'),
               Line2D([], [], color=RANDOM, lw=1, ls=(0, (2.5, 2)), label='Random referral'),
               Line2D([], [], marker='o', ls='', markersize=4.5, markerfacecolor='white', markeredgecolor=INK, mew=1.1, label='20% referred')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=4, frameon=False, handlelength=2.2, columnspacing=1.8)
    save(fig, 'figure-1-error-capture')


SHORT = {'sol': 'GPT-6.1 Sol', 'gemini': 'Gemini 3.8 Flash', 'lunamax': 'GPT-6 Luna max', 'opus': 'Claude Opus 5.5',
         'sonnet': 'Claude Sonnet 5.5'}
COST_MAX = 1.0  # Linear cost axis in US$ per 1,000 questions. Reference models beyond it are marked at the right edge.
# Label positions in data coordinates (cost, accuracy) and alignment, chosen so that no label touches a curve.
LABELS = {
    'kormed': {'lunamax': (.42, 99.55, 'center'), 'sol': (.76, 99.6, 'center'), 'gemini': (.74, 97.25, 'center')},
    'medqa': {'lunamax': (.3, 97.6, 'center'), 'sol': (.62, 98.25, 'center'), 'gemini': (.95, 98.9, 'right')},
}


def figure2(ev):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.45), sharey=True)
    fig.subplots_adjust(left=.075, right=.91, top=.78, bottom=.135, wspace=.27)
    for i, (ax, (cohort, title)) in enumerate(zip(axes, COHORTS)):
        e = ev['cohorts'][cohort]
        for t in TIERS:
            x = e['first_tiers'][t]; cv = x['curve']
            ax.plot([p['cost_per_1000'] for p in cv], [100 * p['accuracy'] for p in cv], color=COLORS[t],
                    lw=1.5 if t != 'luna_letter' else 1.3, ls='-' if t != 'luna_letter' else (0, (4, 1.6)), zorder=3)
            ax.scatter(cv[0]['cost_per_1000'], 100 * cv[0]['accuracy'], s=24, color=COLORS[t], edgecolor='white', lw=.7, zorder=5)
            ax.scatter(x['share_20_cost_per_1000'], 100 * x['shares']['20']['accuracy'], s=28, color='white', edgecolor=COLORS[t], lw=1.3, zorder=6)
        for k, r in e['reference'].items():
            if k == 'luna':
                continue
            if r['cost_per_1000'] <= COST_MAX:
                ax.scatter(r['cost_per_1000'], 100 * r['accuracy'], s=20, marker='D', color=ORACLE, edgecolor='white', lw=.5, zorder=4)
                tx, ty, ha = LABELS[cohort][k]
                ax.annotate(SHORT[k], (r['cost_per_1000'], 100 * r['accuracy']), xytext=(tx, ty), textcoords='data', fontsize=7,
                            color=MUTED, ha=ha, va='center', arrowprops=dict(arrowstyle='-', color=AXIS, lw=.5, shrinkA=1.5, shrinkB=2.5))
            else:
                # Off-scale reference model: an arrow at the right edge at its accuracy, labelled with its actual cost.
                ax.scatter(COST_MAX, 100 * r['accuracy'], s=26, marker='>', color=ORACLE, clip_on=False, zorder=4)
                ax.annotate(f"{SHORT[k].replace('Claude ', '')}\nUS${r['cost_per_1000']:.2f}", (COST_MAX, 100 * r['accuracy']), xytext=(7, 0),
                            textcoords='offset points', fontsize=6.8, color=MUTED, ha='left', va='center', annotation_clip=False, linespacing=1.05)
        ax.set_xlim(0, COST_MAX)
        ax.set_ylim(82, 100); ax.set_yticks(range(82, 101, 2))
        ax.set_title(f"{'AB'[i]}   {title}", loc='left', fontweight='bold', fontsize=8.5, pad=5)
        ax.set_xlabel('Estimated cost, US$ per 1,000 questions')
        style(ax)
    axes[0].set_ylabel('Accuracy (%)')
    handles = [Line2D([], [], color=COLORS[t], lw=1.5, ls='-' if t != 'luna_letter' else (0, (4, 1.6)), label=NAMES[t]) for t in TIERS] + \
              [Line2D([], [], marker='o', ls='', markersize=4.5, color=INK, markeredgecolor='white', label='Model alone'),
               Line2D([], [], marker='o', ls='', markersize=4.5, markerfacecolor='white', markeredgecolor=INK, mew=1.1, label='20% referred to GPT-6.1 Sol'),
               Line2D([], [], marker='D', ls='', markersize=4, color=ORACLE, markeredgecolor='white', label='Reference model')]
    order = [0, 4, 1, 5, 2, 6, 3]  # Matplotlib fills legend columns first, so interleave to fill rows.
    fig.legend(handles=[handles[i] for i in order], loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=4, frameon=False,
               handlelength=2.4, columnspacing=1.6, labelspacing=.45)
    save(fig, 'figure-2-escalation')


def figure3(ev):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.1), sharey='row')
    fig.subplots_adjust(left=.085, right=.99, top=.875, bottom=.07, wspace=.07, hspace=.45)
    w = .36
    panels = (('answer', 'Answer changed', 13.5), ('referral', 'Referral decision changed', 21))
    for row, (key, label, ymax) in enumerate(panels):
        for col, (cohort, title) in enumerate(COHORTS):
            ax = axes[row, col]
            for i, t in enumerate(DECISION):
                o = ev['order'][t][cohort][key]
                for j, (kind, color) in enumerate((('rotation', COLORS[t]), ('repeat', REPEAT))):
                    m, (lo, hi) = 100 * o[kind]['mean'], [100 * v for v in o[kind]['ci95']]
                    x = i + (j - .5) * (w + .04)
                    ax.bar(x, m, width=w, color=color, zorder=3)
                    if hi > lo:
                        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt='none', ecolor=INK, elinewidth=.7, capsize=2, capthick=.7, zorder=4)
                    ax.text(x, max(hi, m) + ymax * .02, f'{m:.1f}', ha='center', va='bottom', fontsize=7, color=INK)
            ax.set_xticks(range(3), [NAMES[t] for t in DECISION])
            ax.tick_params(axis='x', length=0, labelsize=7.5, colors=INK)
            ax.set_xlim(-.6, 2.6); ax.set_ylim(0, ymax)
            ax.set_yticks(range(0, int(ymax) + 1, 4 if key == 'answer' else 5))
            ax.set_title(f"{'ABCD'[2 * row + col]}   {label}, {title.split(' (')[0]}", loc='left', fontweight='bold', fontsize=8.5, pad=5)
            style(ax, 'y')
        axes[row, 0].set_ylabel('Mean change per question (%)')
    rotation = tuple(Rectangle((0, 0), 1, 1, color=COLORS[t]) for t in DECISION)
    handles = [rotation, Rectangle((0, 0), 1, 1, color=REPEAT)]
    fig.legend(handles=handles, labels=['Rotated answer options', 'Identical repeat'], handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
               loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=2, frameon=False, handlelength=2.6, columnspacing=2.4)
    save(fig, 'figure-3-option-order')


def main():
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob('*'):
        old.unlink()
    ev = json.loads((HERE / 'evidence.json').read_text(encoding='utf-8'))
    figure1(ev); figure2(ev); figure3(ev)
    print(json.dumps(sorted(p.name for p in OUT.glob('*.png'))))


if __name__ == '__main__':
    main()
