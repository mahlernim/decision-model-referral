"""Publication figures from evidence.json, drawn at final print size (6.9 in wide) so fonts print at their stated size.

Figure 1. Study design, drawn as a schematic with illustrative values.
Figure 2. Errors caught by referring each model's least confident answers, with random and oracle references.
Figure 3. Accuracy and cost when referring the least confident answers to GPT-6.1 Sol, with reference models.
Figure 4. Answer and referral changes under option rotation and identical repetition.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
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


# Figure 1 schematic. Twenty illustrative answers ordered from least to most confident, four of them wrong.
DESIGN_HEIGHT = 4.45
ACCENT, ACCENT_BG, REFERRED_BG = COLORS['jev'], '#e3eefb', '#c3daf6'
WRONG, RIGHT = '#d1495b', '#d5d9dc'
OPTION_TEXT = ['#f2c14e', '#7cc6a4', '#9db7e8', '#e59a8c', '#c4a7de']

N = 20
ERRORS = {0, 1, 3, 10}
RANDOM_PICK = {2, 7, 10, 16}


def box(ax, x, y, w, h, fc='white', ec=AXIS, lw=.7, r=.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw))


def arrow(ax, a, b, color=MUTED, lw=.9, ms=7):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw, shrinkA=0, shrinkB=0))


def title(ax, x, y, letter, text):
    ax.text(x, y, letter, fontsize=9, fontweight='bold', va='top')
    ax.text(x + .2, y, text, fontsize=8.5, fontweight='bold', va='top')


def squares(ax, x0, y, pitch, size, referred=()):
    """Answers as squares. Referred answers sit on a shaded tile, so neighbouring tiles join into one band."""
    for i in referred:
        ax.add_patch(Rectangle((x0 + i * pitch - (pitch - size) / 2, y - .055), pitch, size + .11, fc=REFERRED_BG, ec='none'))
    for i in range(N):
        ax.add_patch(Rectangle((x0 + i * pitch, y), size, size, fc=WRONG if i in ERRORS else RIGHT, ec='none'))


def panel_a(ax):
    title(ax, 0, 4.42, 'A', 'A decision model scores every option')
    box(ax, .02, 2.3, 1.12, 1.74)
    ax.text(.1, 3.94, 'Question', fontsize=7.5, fontweight='bold', va='top')
    for k, w in enumerate([.92, .86, .6]):
        ax.add_patch(Rectangle((.1, 3.66 - k * .1), w, .04, fc=GRID, ec='none'))
    for k, L in enumerate('ABCDE'):
        y = 3.27 - k * .17
        ax.text(.12, y, L, fontsize=7, va='center', color=INK)
        ax.add_patch(Rectangle((.25, y - .05), .22, .1, fc=OPTION_TEXT[k], ec='none'))
        ax.add_patch(Rectangle((.52, y - .02), .5 - .06 * (k % 3), .04, fc=GRID, ec='none'))
    arrow(ax, (1.16, 3.23), (1.36, 3.23))
    box(ax, 1.38, 2.96, .72, .54, fc=ACCENT_BG, ec=ACCENT, lw=.9)
    ax.text(1.74, 3.23, 'Decision\nmodel', ha='center', va='center', fontsize=7.5, color=ACCENT, fontweight='bold', linespacing=1.1)
    arrow(ax, (2.12, 3.23), (2.32, 3.23))
    probs = [.05, .62, .21, .08, .04]
    ax.text(2.36, 4.0, 'Option probabilities', fontsize=7.5, va='top', fontweight='bold')
    for k, (L, p) in enumerate(zip('ABCDE', probs)):
        y = 3.67 - k * .17
        ax.text(2.4, y, L, fontsize=7, va='center')
        ax.add_patch(Rectangle((2.52, y - .055), .82, .11, fc=GRID, ec='none'))
        ax.add_patch(Rectangle((2.52, y - .055), .82 * p, .11, fc=ACCENT if k == 1 else '#9aa1a6', ec='none'))
        ax.text(2.52 + .82 * p + .04, y, f'{p:.2f}', fontsize=6.5, va='center', color=INK if k == 1 else MUTED,
                fontweight='bold' if k == 1 else 'normal')
    ax.text(2.36, 2.88, 'Selected B, confidence 0.62', fontsize=7.2, va='top', color=ACCENT, fontweight='bold')
    # Generative comparison: the letter probability is usually saturated.
    arrow(ax, (1.16, 2.5), (1.36, 2.5))
    box(ax, 1.38, 2.3, .72, .4, fc='#f1f2f3', ec=AXIS, lw=.7)
    ax.text(1.74, 2.5, 'Generative\nmodel', ha='center', va='center', fontsize=7, color=MUTED, linespacing=1.05)
    arrow(ax, (2.12, 2.5), (2.32, 2.5))
    ax.text(2.36, 2.5, 'Writes "B", and its letter probability\nis 1.00 for most answers', fontsize=6.8, va='center', color=MUTED, linespacing=1.15)


def panel_b(ax):
    x0, pitch, size, y = 3.95, .145, .117, 3.36
    title(ax, x0 - .02, 4.42, 'B', 'The least confident 20% are referred')
    cut = x0 + 4 * pitch - (pitch - size) / 2
    ax.add_patch(Rectangle((x0 - .05, y - .07), cut - x0 + .05, size + .14, fc=REFERRED_BG, ec='none'))
    squares(ax, x0, y, pitch, size)
    ax.plot([cut, cut], [y - .12, y + size + .33], color=ACCENT, lw=.9, ls=(0, (2.5, 1.6)))
    ax.text(cut + .05, y + size + .3, "20th percentile of the model's own confidence,\nset without answer keys", fontsize=6.8, va='top',
            color=ACCENT, linespacing=1.15)
    ax.text(x0 - .02, y + size + .3, 'Answers', fontsize=6.8, va='top', color=MUTED)
    arrow(ax, (x0, y - .17), (x0 + N * pitch - .03, y - .17), color=AXIS, lw=.7, ms=6)
    ax.text(x0, y - .22, 'Least confident', fontsize=6.8, va='top', color=MUTED)
    ax.text(x0 + N * pitch - .03, y - .22, 'Most confident', fontsize=6.8, va='top', ha='right', color=MUTED)
    mid_ref = (x0 + cut) / 2
    arrow(ax, (mid_ref, y - .42), (mid_ref, 2.79), color=ACCENT)
    box(ax, x0 - .03, 2.3, 1.3, .48, fc=ACCENT_BG, ec=ACCENT, lw=.8)
    ax.text(x0 + .62, 2.54, 'Referred to GPT-6.1 Sol\n(a clinician in practice)', ha='center', va='center', fontsize=7, linespacing=1.15)
    mid_acc = 6.155
    arrow(ax, (mid_acc, y - .42), (mid_acc, 2.79))
    box(ax, 5.45, 2.3, 1.41, .48)
    ax.text(6.155, 2.54, "Accepted: the decision\nmodel's answer is kept", ha='center', va='center', fontsize=7, linespacing=1.15)
    for k, (c, lab) in enumerate([(RIGHT, 'Correct answer'), (WRONG, 'Wrong answer')]):
        lx = 5.3 + k * .82
        ax.add_patch(Rectangle((lx, 4.2), .09, .09, fc=c, ec='none'))
        ax.text(lx + .13, 4.245, lab, fontsize=6.8, va='center', color=MUTED)


def panel_c(ax):
    title(ax, 0, 1.98, 'C', 'Referral rules compared at the same share')
    x0, pitch, size = .86, .13, .1
    rows = [('Random', 'floor', RANDOM_PICK), ('Confidence', 'this study', set(range(4))), ('Oracle', 'ceiling', ERRORS)]
    ax.text(3.52, 1.7, 'Errors\ncaught', fontsize=6.8, va='top', ha='left', color=MUTED, linespacing=1.1)
    for k, (name, role, pick) in enumerate(rows):
        y = 1.32 - k * .38
        hi = name == 'Confidence'
        ax.text(.02, y + size / 2 + .05, name, fontsize=7.5, va='center', fontweight='bold' if hi else 'normal', color=ACCENT if hi else INK)
        ax.text(.02, y + size / 2 - .1, role, fontsize=6.8, va='center', color=MUTED)
        squares(ax, x0, y, pitch, size, referred=pick)
        ax.text(3.52, y + size / 2, f'{len(pick & ERRORS)} of 4', fontsize=7.5, va='center', color=ACCENT if hi else INK,
                fontweight='bold' if hi else 'normal')
    ax.text(.02, .12, 'Shaded answers are referred. Outcomes were errors caught,\naccuracy after referral and cost.', fontsize=6.8,
            va='center', color=MUTED, linespacing=1.2)


def panel_d(ax):
    x0 = 3.95
    title(ax, x0 - .02, 1.98, 'D', 'Stability when options are reordered')
    cx, pitch, s = 5.2, .25, .19
    for j, L in enumerate('ABCDE'):
        ax.text(cx + j * pitch + s / 2, 1.66, L, ha='center', va='center', fontsize=7, color=MUTED)
    rows = [('Published order', 0, 1), ('Rotation 1', 1, 1), ('Rotation 2', 2, 2), ('Identical repeat', 0, 1)]
    for k, (name, shift, chosen) in enumerate(rows):
        y = 1.36 - k * .29 - (.1 if k == 3 else 0)
        ax.text(x0, y + s / 2, name, fontsize=7, va='center')
        for j in range(5):
            t = (j + shift) % 5
            ax.add_patch(Rectangle((cx + j * pitch, y), s, s, fc=OPTION_TEXT[t], ec=INK if t == chosen else 'none', lw=1.1))
        if k == 2:
            ax.text(cx + 5 * pitch + .02, y + s / 2, 'Answer\nchanged', fontsize=6.5, va='center', color=WRONG, linespacing=1.05)
    ax.text(cx + 2 * pitch + s / 2, .66, '...', ha='center', va='center', fontsize=7, color=MUTED)
    ax.text(x0, .12, 'Colors are option texts and the outline is the chosen text.\nChanges in answer and referral were counted.',
            fontsize=6.8, va='center', color=MUTED, linespacing=1.2)


def figure1():
    fig = plt.figure(figsize=(WIDTH, DESIGN_HEIGHT))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, WIDTH); ax.set_ylim(0, DESIGN_HEIGHT); ax.axis('off')
    panel_a(ax); panel_b(ax); panel_c(ax); panel_d(ax)
    ax.plot([0, WIDTH], [2.08, 2.08], color=GRID, lw=.8)
    save(fig, 'figure-1-study-design')


def figure2(ev):
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
    save(fig, 'figure-2-error-capture')


SHORT = {'sol': 'GPT-6.1 Sol', 'gemini': 'Gemini 3.8 Flash', 'lunamax': 'GPT-6 Luna max', 'opus': 'Claude Opus 5.5',
         'sonnet': 'Claude Sonnet 5.5'}
COST_MAX = 1.0  # Linear cost axis in US$ per 1,000 questions. Reference models beyond it are marked at the right edge.
# Label positions in data coordinates (cost, accuracy) and alignment, chosen so that no label touches a curve.
LABELS = {
    'kormed': {'lunamax': (.42, 99.55, 'center'), 'sol': (.76, 99.6, 'center'), 'gemini': (.74, 97.25, 'center')},
    'medqa': {'lunamax': (.3, 97.6, 'center'), 'sol': (.62, 98.25, 'center'), 'gemini': (.95, 98.9, 'right')},
}


def figure3(ev):
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
    save(fig, 'figure-3-escalation')


def figure4(ev):
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
    save(fig, 'figure-4-option-order')


def main():
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob('*'):
        old.unlink()
    ev = json.loads((HERE / 'evidence.json').read_text(encoding='utf-8'))
    figure1(); figure2(ev); figure3(ev); figure4(ev)
    print(json.dumps(sorted(p.name for p in OUT.glob('*.png'))))


if __name__ == '__main__':
    main()
