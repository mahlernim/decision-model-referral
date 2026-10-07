"""Option-order stability for Jev, OpenAI Decisions and Clef, summarized with the Jev order-analysis definitions."""
import json
from collections import defaultdict

from .common import ROOT, read
from .decision_models import OUTPUT
from .medical_order_analysis import estimate

SOURCES = {'jev': ROOT / 'runs/medical-option-order-v1/attempts',
           'decisions': ROOT / 'runs/decision-models-order-v1/decisions/attempts',
           'clef': ROOT / 'runs/decision-models-order-v1/clef/attempts'}
METRICS = ('semantic_flip', 'referral_flip', 'absolute_selected_probability_change', 'total_variation')


def load(model):
    last = {}
    for p in sorted(SOURCES[model].glob('*.json')):
        r = read(p)
        if r['terminal'] and (r['eval_id'] not in last or r['attempt'] > last[r['eval_id']]['attempt']):
            last[r['eval_id']] = r
    values = {}
    for r in last.values():
        values[(r['cohort'], r['item_id'], r['variant'])] = r['value'] if r['status'] == 'success' else None
    return values


def compare(a, b):
    return dict(semantic_flip=float(a['semantic_prediction'] != b['semantic_prediction']),
                referral_flip=float(a['referred'] != b['referred']),
                absolute_selected_probability_change=abs(a['selected_probability'] - b['selected_probability']),
                total_variation=.5 * sum(abs(a['semantic_probabilities'][k] - b['semantic_probabilities'][k])
                                         for k in a['semantic_probabilities']))


def summarize(values, cohort):
    nopt = 5 if cohort == 'kormed' else 4
    ids = sorted({i for c, i, _ in values if c == cohort})
    rows, full = [], defaultdict(list)
    for q in ids:
        base, rep = values.get((cohort, q, 'rot0')), values.get((cohort, q, 'repeat'))
        rot = [values.get((cohort, q, f'rot{k}')) for k in range(1, nopt)]
        full['original'].append(float(bool(base and base['correct'])))
        full['rotation_average'].append(sum(float(bool(v and v['correct'])) for v in rot) / len(rot))
        if base is None or rep is None or any(v is None for v in rot):
            continue
        comps = [compare(base, v) for v in rot]; ctl = compare(base, rep)
        row = {}
        for m in METRICS:
            row[f'rotation_{m}'] = sum(c[m] for c in comps) / len(comps)
            row[f'repeat_{m}'] = ctl[m]; row[f'excess_{m}'] = row[f'rotation_{m}'] - ctl[m]
        row['any_rotation_semantic_flip'] = float(any(c['semantic_flip'] for c in comps))
        row['rotation_accuracy'] = sum(float(v['correct']) for v in rot) / len(rot)
        row['original_correct'] = float(base['correct'])
        rows.append(row)
    return dict(planned_items=len(ids), complete_items=len(rows),
                metrics={m: {k: estimate([r[f'{k}_{m}'] for r in rows]) for k in ('rotation', 'repeat', 'excess')} for m in METRICS},
                any_rotation_semantic_flip=estimate([r['any_rotation_semantic_flip'] for r in rows]),
                full_accuracy={k: estimate(v) for k, v in full.items()},
                complete_accuracy=dict(original=estimate([r['original_correct'] for r in rows]),
                                       rotation_average=estimate([r['rotation_accuracy'] for r in rows])))


def analyze():
    result = {}
    for model in SOURCES:
        values = load(model)
        result[model] = {c: summarize(values, c) for c in ('kormed', 'medqa')}
    (OUTPUT / 'order-summary.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    return result


def brief(result):
    pct = lambda e: f"{e['mean']*100:.2f} [{e['ci95'][0]*100:.2f}, {e['ci95'][1]*100:.2f}]"
    for c in ('kormed', 'medqa'):
        print(c)
        for model, s in result.items():
            s = s[c]; m = s['metrics']
            print(f"  {model:10s} complete {s['complete_items']}/{s['planned_items']} | answer rot {pct(m['semantic_flip']['rotation'])} "
                  f"rep {pct(m['semantic_flip']['repeat'])} excess {pct(m['semantic_flip']['excess'])} | referral rot "
                  f"{pct(m['referral_flip']['rotation'])} rep {pct(m['referral_flip']['repeat'])} excess {pct(m['referral_flip']['excess'])} | "
                  f"TV excess {pct(m['total_variation']['excess'])} | acc orig {s['full_accuracy']['original']['mean']*100:.2f} "
                  f"rot {s['full_accuracy']['rotation_average']['mean']*100:.2f}")


if __name__ == '__main__':
    brief(analyze())
