"""Offline checks for the medRxiv draft: every table cell and reported number against evidence.json, citations,
wording rules, internal-only terms, preserved frozen artifacts and the DOCX."""
import hashlib
import json
import re
from pathlib import Path

from docx import Document

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
main = (HERE / 'manuscript.md').read_text(encoding='utf-8')
supplement = (HERE / 'supplementary.md').read_text(encoding='utf-8')
body = main.split('## References')[0]
ev = json.loads((HERE / 'evidence.json').read_text(encoding='utf-8'))
checks = 0
C = {'Korean': 'kormed', 'English': 'medqa'}
N = {'Jev 1.13.0': 'jev', 'OpenAI Decisions': 'decisions', 'Clef': 'clef', 'GPT-6 Luna letter probabilities': 'luna_letter'}
REF = {'GPT-6.1 Sol low': 'sol', 'Claude Opus 5.5 low': 'opus', 'Gemini 3.8 Flash low': 'gemini', 'GPT-6 Luna max': 'lunamax',
       'Claude Sonnet 5.5 low': 'sonnet', 'GPT-6 Luna none': 'luna'}
DM = ['jev', 'decisions', 'clef']


def equal(actual, expected):
    global checks
    assert actual == expected, (actual, expected)
    checks += len(actual) if isinstance(actual, list) else 1


def contains(text, *needles):
    global checks
    for s in needles:
        assert s in text, s
        checks += 1


def tables(text):
    result, rows = [], []
    for line in text.splitlines() + ['']:
        if line.startswith('|'):
            row = [x.strip() for x in line.strip('|').split('|')]
            if not all(re.fullmatch(':?-+:?', x) for x in row):
                rows.append(row)
        elif rows:
            result.append(rows)
            rows = []
    return result


p1 = lambda x: f'{100*x:.1f}'
p2 = lambda x: f'{100*x:.2f}'
ci1 = lambda x, lo, hi: f'{100*x:.1f} ({100*lo:.1f} to {100*hi:.1f})'
ci3 = lambda x, lo, hi: f'{x:.3f} ({lo:.3f} to {hi:.3f})'
rng = lambda vals, fmt=p1: f'{fmt(min(vals))}% to {fmt(max(vals))}%'
raw_mt, raw_st = tables(main), tables(supplement)


def filled(table):
    """Repeat group labels that the published tables show only once per group."""
    out, prev = [], None
    for i, row in enumerate(table):
        row = list(row)
        if i and prev is not None:
            for c in (0, 1):
                if row[c] == '' and prev[c]:
                    row[c] = prev[c]
        out.append(row); prev = row
    return out


mt, st = [filled(t) for t in raw_mt], [filled(t) for t in raw_st]
FT = {c: ev['cohorts'][c]['first_tiers'] for c in C.values()}
BS = {c: ev['cohorts'][c]['bootstrap'] for c in C.values()}

# Table 1, decision models and letter probabilities.
for row in mt[0][1:]:
    c, t = C[row[0]], N[row[1]]; x = FT[c][t]
    equal(row[2:], [ci1(x['accuracy'], *x['accuracy_ci95']), ci3(x['error_auroc'], *x['error_auroc_ci95']), f"{x['ece']:.3f}",
                    p1(x['share_probability_one']), f"{x['cost_per_1000']:.4f}"])
equal(len(mt[0]) - 1, 8)
# Table 2, reference models.
for row in mt[1][1:]:
    k = REF[row[0]]; exp = []
    for c in C.values():
        r = ev['cohorts'][c]['reference'][k]
        exp += [ci1(r['accuracy'], *r['accuracy_ci95']), f"{r['cost_per_1000']:.4f}"]
    equal(row[1:], exp)
# Table 3, the 20% share.
for row in mt[2][1:]:
    c, t = C[row[0]], N[row[1]]; s = FT[c][t]['shares']['20']; b = BS[c]
    equal(row[2:], [ci1(s['capture'], *b[t + '_capture']), ci1(s['accuracy'], *b[t + '_accuracy']), p1(s['random_accuracy']),
                    p1(s['oracle_accuracy']), f"{100*s['achievable']:.0f}", f"{FT[c][t]['share_20_cost_per_1000']:.3f}"])
    equal([s['k'], s['oracle_capture']], [87 if c == 'kormed' else 255, 1.0])
sol_ = {c: ev['cohorts'][c]['reference']['sol'] for c in C.values()}
contains(main, f"GPT-6.1 Sol alone answered {p1(sol_['kormed']['accuracy'])}% of Korean and {p1(sol_['medqa']['accuracy'])}% of English questions correctly at US${sol_['kormed']['cost_per_1000']:.3f} and US${sol_['medqa']['cost_per_1000']:.3f} per 1,000 questions")
# Table 4 and Supplementary Table 7, error groups.
def bands(table, c, keys):
    t = {r[0]: r[1:] for r in table[1:]}; E = ev['cohorts'][c]['errors']
    assert set(t) == {label for label, _ in keys}, set(t)
    for label, k in keys:
        equal(t[label], [str(E[m]['bands'][b][k]) if k == 'n' else p1(E[m]['bands'][b][k]) for m in DM for b in ('cautious', 'confident')])


common = [('Errors, n', 'n'), ('Answer changed under rotation, %', 'flip'), ('Corrected by GPT-6.1 Sol, %', 'sol'),
          ('Mean reference accuracy, %', 'panel_accuracy'), ('GPT-6 Luna none chose the same wrong option, %', 'luna_same')]
bands(mt[3], 'medqa', common + [('Physician-majority flagged, %', 'flagged'), ("Physician majority chose the model's answer, %", 'physicians_chose')])
bands(st[6], 'kormed', common + [('Law and policy content, %', 'law')])
thr = lambda c: [f"{ev['cohorts'][c]['errors'][m]['threshold']:.3f}" for m in DM]
k_thr, e_thr = thr('kormed'), thr('medqa')
contains(main, f"{e_thr[0]} or less for Jev, {e_thr[1]} or less for OpenAI Decisions and {e_thr[2]} or less for Clef")
contains(supplement, f"with thresholds of {k_thr[0]}, {k_thr[1]} and {k_thr[2]}",
         f"Referral thresholds were {k_thr[0]}, {k_thr[1]} and {k_thr[2]} in Korean and {e_thr[0]}, {e_thr[1]} and {e_thr[2]} in English")
assert all(abs(ev['cohorts'][c]['errors'][m]['threshold'] - FT[c][m]['threshold_20']) < 1e-12 for c in C.values() for m in DM)

# Supplementary Table 2, standalone counts.
for row in st[1][1:]:
    c = C[row[0]]
    if row[1] in N:
        x = FT[c][N[row[1]]]; equal(row[2:], [f"{x['correct']}/{x['n']}", str(x['valid'])])
    else:
        r = ev['cohorts'][c]['reference'][REF[row[1]]]; equal(row[2:], [f"{r['correct']}/{ev['cohorts'][c]['n']}", str(r['valid'])])
equal(len(st[1]) - 1, 20)
# Supplementary Table 3, the separately reported confidence field.
NC = {c: ev['cohorts'][c]['native_confidence'] for c in C.values()}
for row in st[2][1:]:
    x = NC[C[row[0]]][N[row[1]]]; p_, q_ = x['probability'], x['confidence']
    equal(row[2:], [f"{x['mean_difference']:.3f}", f"{p_['error_auroc']:.3f} / {q_['error_auroc']:.3f}", f"{p_['ece']:.3f} / {q_['ece']:.3f}",
                    f"{100*p_['capture']:.1f} / {100*q_['capture']:.1f}"])
gaps = [abs(x['probability']['error_auroc'] - x['confidence']['error_auroc']) for c in NC.values() for x in c.values()]
worse = sum(x['confidence']['ece'] > x['probability']['ece'] for c in NC.values() for x in c.values())
assert 0.002 < max(gaps) <= 0.003 and worse == 5 and all(x['mean_difference'] < 0 for c in NC.values() for x in c.values())
contains(body, 'ranked errors almost identically but was systematically lower than the selected probability and generally less well calibrated (Supplementary Table 3)')
# Supplementary Table 4, AUROC differences.
lab = {'OpenAI Decisions minus GPT-6 Luna letter probabilities': 'decisions_minus_luna_letter', 'Jev minus OpenAI Decisions': 'jev_minus_decisions',
       'Jev minus Clef': 'jev_minus_clef', 'OpenAI Decisions minus Clef': 'decisions_minus_clef',
       'Jev minus GPT-6 Luna letter probabilities': 'jev_minus_luna_letter'}
for row in st[3][1:]:
    b, k = BS[C[row[0]]], lab[row[1]]
    equal(row[2], ci3(b[f'auroc_{k}_estimate'], *b[f'auroc_{k}']))
# Supplementary Table 5, shares.
for row in st[4][1:]:
    s = FT[C[row[0]]][N[row[1]]]['shares'][row[2]]
    equal(row[3:], [p1(s['capture']), p1(s['oracle_capture']), p1(s['accuracy']), p1(s['random_accuracy']), p1(s['oracle_accuracy'])])
equal(len(st[4]) - 1, 32)
# Supplementary Table 6, option order.
for row in st[5][1:]:
    o = ev['order'][N[row[1]]][C[row[0]]]; x = o['answer' if row[2] == 'Answer' else 'referral']
    equal(row[3:], [str(o['complete']), p2(x['rotation']['mean']), p2(x['repeat']['mean']),
                    f"{p2(x['excess']['mean'])} ({p2(x['excess']['ci95'][0])} to {p2(x['excess']['ci95'][1])})"])
# Supplementary Table 8, content groups.
for row in st[7][1:]:
    k = N.get(row[0]) or REF[row[0]]
    a, b = ev['cohorts']['kormed']['content'][k], ev['cohorts']['medqa']['content'][k]
    equal(row[1:], [p1(a['inside_accuracy']), p1(a['other_accuracy']), p1(a['share_of_errors']),
                    p1(b['inside_accuracy']), p1(b['other_accuracy']), p1(b['share_of_errors'])])
equal([ev['cohorts']['kormed']['content']['n_inside'], ev['cohorts']['medqa']['content']['n_inside'], ev['labels']['sol_positive']], [71, 192, 71])
# Supplementary Table 9 is descriptive. Supplementary Table 10, flag sensitivity.
equal(len(st[8]) - 1, 9)
F = ev['flag_sensitivity']
for row in st[9][1:]:
    x = F[N[row[0]]]
    equal(row[1:], [p1(x['accuracy']), f"{x['error_auroc']:.3f}", p1(x['capture']), p1(x['accuracy_20']), p1(x['random_accuracy_20']), p1(x['oracle_accuracy_20'])])
equal(F['n'], 1081)
contains(supplement, f"answered {p1(F['sol'])}% of these questions correctly")
equal(len(st), 10)

# Every table and figure is cited in the main text, numbered by first mention and placed after that mention.
cited = re.sub(r'^\*\*(?:Supplementary )?(?:Table|Figure) \d+\..*$', '', body, flags=re.M)
for kind, count in (('Table', 4), ('Figure', 4), ('Supplementary Table', 10)):
    pattern = r'(?<!Supplementary )' + kind + r' (\d+)' if kind != 'Supplementary Table' else r'Supplementary Table (\d+)'
    first = []
    for m in re.finditer(pattern, cited):
        if int(m.group(1)) not in first:
            first.append(int(m.group(1)))
    equal(first, list(range(1, count + 1)))
    if kind != 'Supplementary Table':
        for n in range(1, count + 1):
            mention = body.index(re.search(r'(?<!Supplementary )' + kind + rf' {n}\b', cited).group(0))
            assert body.index(f'**{kind} {n}.') > mention, (kind, n)
for n in range(1, 11):
    assert supplement.index(f'**Supplementary Table {n}.') > supplement.index('## Supplementary references')
assert supplement.index('## Supplementary tables') > supplement.index('## Supplementary methods')
for line in (main + supplement).splitlines():
    assert not re.match(r'^#+ (?:Supplementary )?(?:Table|Figure) ', line), line

# Numbers in the main text and abstract.
K, E = FT['kormed'], FT['medqa']
dm = lambda c, key: [FT[c][m][key] for m in DM]
s20 = lambda c, key: [FT[c][m]['shares']['20'][key] for m in DM]
irng = lambda vals: f"{100*min(vals):.0f}% to {100*max(vals):.0f}%"
auc_rng = lambda c: f"{min(dm(c, 'error_auroc')):.3f} to {max(dm(c, 'error_auroc')):.3f}"
contains(body, f"answered {rng(dm('kormed', 'accuracy'))} of Korean and {rng(dm('medqa', 'accuracy'))} of English questions correctly, with error-detection",
         f"of {auc_rng('kormed')} and {auc_rng('medqa')}",
         f"held {irng(s20('kormed', 'capture'))} of Korean and {irng(s20('medqa', 'capture'))} of English errors")
d = lambda c, k: (BS[c][f'auroc_{k}_estimate'], *BS[c][f'auroc_{k}'])
dk, de = d('kormed', 'decisions_minus_luna_letter'), d('medqa', 'decisions_minus_luna_letter')
contains(body, f"by {dk[0]:.3f} (95% CI {dk[1]:.3f} to {dk[2]:.3f}) in Korean and {de[0]:.3f} ({de[1]:.3f} to {de[2]:.3f}) in English",
         f"higher by {dk[0]:.3f} (95% CI {dk[1]:.3f} to {dk[2]:.3f}) in Korean and {de[0]:.3f} ({de[1]:.3f} to {de[2]:.3f}) in English",
         f"exactly one for {p1(K['luna_letter']['share_probability_one'])}% of Korean and {p1(E['luna_letter']['share_probability_one'])}% of English answers")
assert dk[1] > 0 and de[1] > 0
gains = [FT[c][m]['shares']['20']['accuracy'] - FT[c][m]['accuracy'] for c in C.values() for m in DM]
achs = [FT[c][m]['shares']['20']['achievable'] for c in C.values() for m in DM]
contains(body, f"caught {irng(s20('kormed', 'capture') + s20('medqa', 'capture'))} of errors")
contains(body, f"raised accuracy by {100*min(gains):.0f} to {100*max(gains):.0f} percentage points, achieving {irng(achs)} of the gain available to an oracle",
         f"raised accuracy by {100*min(gains):.1f} to {100*max(gains):.1f} percentage points, more than random referral",
         f"raised accuracy by {100*min(gains):.0f} to {100*max(gains):.0f} percentage points at a quarter to a third")
assert .45 < min(achs) < .55 and .85 < max(achs) < .95
contains(body, f"achieved {irng(s20('kormed', 'achievable'))} of the oracle's gain in Korean and {irng(s20('medqa', 'achievable'))} in English")
assert all(FT[c][m]['shares']['20']['accuracy'] > FT[c][m]['shares']['20']['random_accuracy'] for c in C.values() for m in DM)
assert min(FT[c][m]['shares']['50']['capture'] for c in C.values() for m in DM) > .9
sol = {c: ev['cohorts'][c]['reference']['sol'] for c in C.values()}
ratios = [FT[c][m]['share_20_cost_per_1000'] / sol[c]['cost_per_1000'] for c in C.values() for m in DM]
assert .25 <= min(ratios) and max(ratios) <= .36, ratios
contains(body, 'cost about a quarter to a third as much as using GPT-6.1 Sol alone', 'at a quarter to a third of Sol\'s cost')
lm = {c: ev['cohorts'][c]['reference']['lunamax'] for c in C.values()}
assert all(lm[c]['accuracy'] >= max(s20(c, 'accuracy')) - 1e-9 for c in C.values())
assert all(min(FT[c][m]['share_20_cost_per_1000'] for m in DM) - .02 <= lm[c]['cost_per_1000'] <= max(FT[c][m]['share_20_cost_per_1000'] for m in DM) + .04 for c in C.values())
contains(body, f"catching {100*K['luna_letter']['shares']['20']['capture']:.0f}% of Korean and {100*E['luna_letter']['shares']['20']['capture']:.0f}% of English errors",
         f"the {100 - 100*K['luna_letter']['share_probability_one']:.0f}% to {100 - 100*E['luna_letter']['share_probability_one']:.0f}% of Luna answers below one")
contains(body, f"did for {irng([FT[c][m]['share_probability_one'] for c in C.values() for m in ('jev', 'decisions')])} of answers")
assert K['clef']['share_probability_one'] == 0 and E['clef']['share_probability_one'] == 0
assert K['decisions']['n'] - K['decisions']['valid'] == 1 and E['decisions']['n'] - E['decisions']['valid'] == 6
refs_ = [ev['cohorts'][c]['reference'][k] for c in C.values() for k in REF.values()]
paid = [r['cost_per_1000'] for c in C.values() for k, r in ev['cohorts'][c]['reference'].items() if k != 'luna']
contains(body, f"at {rng([r['accuracy'] for r in refs_])}, but apart from GPT-6 Luna without reasoning they cost {min(paid):.2f} to {max(paid):.2f} US dollars per 1,000 questions against {min(dm('kormed', 'cost_per_1000') + dm('medqa', 'cost_per_1000')):.2f} to {max(dm('kormed', 'cost_per_1000') + dm('medqa', 'cost_per_1000')):.2f} for the decision models")
assert min(r['accuracy'] for r in refs_) > max(dm('kormed', 'accuracy') + dm('medqa', 'accuracy')) - .03
# Option order in the text.
O = ev['order']
allc = lambda k, kind: [O[m][c][k][kind]['mean'] for m in DM for c in C.values()]
contains(body, f"Rotating options changed {irng(allc('referral', 'rotation'))} of referral decisions",
         f"changed answers on {100*min(allc('answer', 'rotation')):.1f}% to {100*max(allc('answer', 'rotation')):.1f}% of rotations and referral decisions on {100*min(allc('referral', 'rotation')):.1f}% to {100*max(allc('referral', 'rotation')):.1f}%",
         f"by {100*min(allc('referral', 'excess')):.1f} to {100*max(allc('referral', 'excess')):.1f} percentage points, with every 95% confidence interval above zero")
assert all(O[m][c]['referral']['excess']['ci95'][0] > 0 for m in DM for c in C.values())
assert all(1.5 < 100 * O['jev'][c]['answer']['repeat']['mean'] < 2.5 for c in C.values())
assert all(O[m][c]['referral']['repeat']['mean'] == 0 and O[m][c]['answer']['repeat']['mean'] == 0 for m in ('decisions', 'clef') for c in C.values())
assert max(abs(O[m][c]['accuracy_original'] - O[m][c]['accuracy_rotation']) for m in DM for c in C.values()) < .015
# Error groups in the text.
EE, EK = ev['cohorts']['medqa']['errors'], ev['cohorts']['kormed']['errors']
band = lambda E_, m, b, k: E_[m]['bands'][b][k]
cm = lambda E_, m, k: E_[m]['cautious_minus_confident'][k]
f1 = lambda x: f"{100*x['estimate']:.1f}"
fi = lambda x: f"{100*x['ci95'][0]:.1f} to {100*x['ci95'][1]:.1f}"
contains(body, f"({irng([band(EE, m, 'confident', 'flip') for m in DM])} of rotations, against {irng([band(EE, m, 'cautious', 'flip') for m in DM])} for cautious errors)",
         f"OpenAI Decisions left only {band(EK, 'decisions', 'confident', 'n')} confident errors")
for m in DM:
    assert band(EE, m, 'confident', 'panel_accuracy') < band(EE, m, 'cautious', 'panel_accuracy')
    assert band(EE, m, 'confident', 'luna_same') > band(EE, m, 'cautious', 'luna_same')
    assert band(EE, m, 'confident', 'sol') < band(EE, m, 'cautious', 'sol')
    assert band(EE, m, 'confident', 'physicians_chose') > band(EE, m, 'cautious', 'physicians_chose')
    assert cm(EE, m, 'flip')['ci95'][0] > 0
assert cm(EK, 'jev', 'flip')['ci95'][0] > 0 and band(EK, 'clef', 'confident', 'flip') > .8 * band(EK, 'clef', 'cautious', 'flip')
s_cont = lambda c, k: [ev['cohorts'][c]['content'][m][k] for m in DM]
contains(body, f"formed {100*71/435:.0f}% of the Korean cohort but {irng(s_cont('kormed', 'share_of_errors'))} of the decision models' errors",
         f"formed {100*192/1273:.0f}% of the English cohort but held {irng(s_cont('medqa', 'share_of_errors'))} of the decision models' errors")
for m in DM:  # most law and policy errors fall within the least confident 20%, all of them for OpenAI Decisions
    lc, lf = band(EK, m, 'cautious', 'law') * band(EK, m, 'cautious', 'n'), band(EK, m, 'confident', 'law') * band(EK, m, 'confident', 'n')
    assert lc / (lc + lf) > .5 and (m != 'decisions' or lf == 0)
assert sum(ev['cohorts']['kormed']['content'][k]['inside_accuracy'] < ev['cohorts']['kormed']['content'][k]['other_accuracy'] for k in REF.values()) >= 4
assert all(F[t]['error_auroc'] > FT['medqa'][t]['error_auroc'] and F[t]['accuracy'] > FT['medqa'][t]['accuracy'] for t in N.values())
sh = ev['cohorts']['medqa']['shared']
contains(body, f"all wrong on {sh['wrong_all']} English questions, and a physician majority had flagged half of them")
assert .45 < sh['content'] / sh['wrong_all'] < .55
# Supplementary text numbers.
lb = ev['labels']; cmiss = ev['consensus_missed']
contains(supplement, f"{lb['sol_positive']} of 435 questions were labeled", f"blinded sample of {lb['review_n']} questions",
         f"and {lb['review_random_negative']} random others", f"agreed with the label on {100*lb['author_sol_agree']/lb['review_n']:.1f}% of them (Cohen's kappa {lb['author_sol_kappa']:.2f})",
         f"{cmiss['missed']} English questions".replace('24 English', 'Twenty-four English'),
         f"flagged {cmiss['majority_concern']['consensus_missed']['pct']}% of them, against {cmiss['majority_concern']['others']['pct']}%",
         f"{cmiss['majority_concern']['risk_ratio']['rr']} times as often (95% CI {cmiss['majority_concern']['risk_ratio']['ci95'][0]} to {cmiss['majority_concern']['risk_ratio']['ci95'][1]}",
         f"{cmiss['no_endorsed_key']['consensus_missed']['pct']}% of them, against {cmiss['no_endorsed_key']['others']['pct']}%",
         f"risk ratio {cmiss['no_endorsed_key']['risk_ratio']['rr']}, 95% CI {cmiss['no_endorsed_key']['risk_ratio']['ci95'][0]:.2f} to {cmiss['no_endorsed_key']['risk_ratio']['ci95'][1]}",
         f"wrong option on {cmiss['same_wrong_option']} of the 24", f"selected that option on {cmiss['physicians_endorse']['at_least_one']}")
assert lb['author_yes_sol_no'] == 5 and lb['author_no_sol_yes'] == 0 and lb['review_terra_positive'] + lb['review_random_negative'] == lb['review_n']
contains(supplement, *[f"{f1(cm(EK, m, 'flip'))} ({'95% CI ' if m == 'jev' else ''}{fi(cm(EK, m, 'flip'))})".replace(' (95% CI', ' percentage points (95% CI') for m in DM])

# Citations appear in first-use order and every listed reference is cited.
refs = re.findall(r'^(\d+)\. ', main.split('## References')[1].split('## Tables')[0], re.M)
order = []
for group in re.findall(r'\[(\d+(?:,\d+)*)\]', body):
    for n in map(int, group.split(',')):
        if n not in order:
            order.append(n)
equal(order, list(range(1, len(refs) + 1)))
for n in re.findall(r'\[(\d+)\]', supplement):
    assert int(n) <= len(refs), n

# Wording rules and internal-only content.
text = main + supplement
assert '—' not in text
assert not re.search(r'GPT-5', text), 'superseded model name'
for term in ('Haiku', 'Sol medium', 'medium effort', 'Astra', 'routing', 'blend', 'two-tier', 'repaired', 'parser', 'gateway',
             'frozen', 'freez', 'sha256', 'PENDING', 'early-access', 'early access', '[Author', '[author', 'matched-cost', 'Luna-first', 'Jev-first'):
    assert term not in text, term
assert not re.search(r'\bquota\b', text, re.I)
prose = re.sub(r'^\d+\. .*$', '', text, flags=re.M)  # reference titles may contain "our"
for word in (r'\b[Ww]e\b', r'\b[Oo]ur\b', r'\bus\b'):
    assert not re.search(word, prose), word


# DOCX mirrors the Markdown tables and figures.
doc = Document(HERE / 'medrxiv-manuscript.docx')
equal([[[c.text for c in r.cells] for r in t.rows] for t in doc.tables], raw_mt + raw_st)
equal(len(doc.inline_shapes), 4)
print(json.dumps({'status': 'passed', 'checks': checks, 'main_tables': len(mt), 'supplementary_tables': len(st),
                  'references': len(refs),
                  'main_text_words_including_abstract': len(body.split())}, indent=2))
