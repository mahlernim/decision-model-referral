"""Frozen option-order stability collection for OpenAI Decisions and Clef, mirroring the Jev order experiment."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import threading
import time
from collections import Counter

from .common import ROOT, canonical, digest, filehash, now, read, write_new
from .decision_models import (ENGLISH_INSTRUCTION, KOREAN_INSTRUCTION, MODELS, OUTPUT, TRANSIENT, credentials,
                              endpoint, inspect, items, upper, usage)
from .medical_order import make_evaluations

DIRECTORY = ROOT / 'runs/decision-models-order-v1'
PROTOCOL = OUTPUT / 'order-protocol.md'
SEED = 2026100703
BUDGET = {'decisions': 2.0, 'clef': 3.0}
GATEWAY = 'clef-gateway'
QUOTA = 'daily free allocation'
CODE = ['jevbench/decision_models_order.py', 'jevbench/decision_models.py', 'jevbench/medical_order.py', 'jevbench/common.py']


class QuotaReached(RuntimeError):
    pass


def to_model(model, jev_request):
    criteria = jev_request['questions']['answer']['criteria']; inst = jev_request['questions']['answer']['instructions']
    state = jev_request['state']
    if model == 'clef':
        return {'model': 'clef', 'state': state, 'questions': {'answer': {'type': 'choice', 'instructions': inst, 'criteria': criteria}}}
    return {'model': MODELS[model], 'input': canonical(state), 'questions': [{
        'type': 'choice', 'name': 'answer', 'instructions': inst,
        'choices': [{'value': k, 'description': v} for k, v in criteria.items()]}]}


def evaluations(model, thresholds):
    out = []
    for item in [i for i in items() if i['split'] == 'test']:
        inst = KOREAN_INSTRUCTION if item['cohort'] == 'kormed' else ENGLISH_INSTRUCTION
        for e in make_evaluations(item, inst, threshold=thresholds[item['cohort']]):
            req = to_model(model, e['request'])
            out.append({**{k: v for k, v in e.items() if k not in ('request', 'request_hash')},
                        'eval_id': f"{model}__{e['eval_id']}", 'request': req, 'request_hash': digest(req)})
    return out


def prepare():
    path = DIRECTORY / 'manifest.json'
    if path.exists():
        return manifest()
    summary = read(OUTPUT / 'summary.json')
    thresholds = {m: {c: summary[c][m]['threshold'] for c in ('kormed', 'medqa')} for m in MODELS}
    evals = {m: evaluations(m, thresholds[m]) for m in MODELS}
    assert all(len(v) == 8975 for v in evals.values())
    rng = random.Random(SEED)
    blocks = sorted({e['item_id'] for e in evals['decisions']}); rng.shuffle(blocks)
    order = {}
    for b in blocks:
        variants = sorted({e['variant'] for e in evals['decisions'] if e['item_id'] == b}); rng.shuffle(variants); order[b] = variants
    m = dict(experiment='decision-models-order-v1', created_at=now(), models=MODELS, thresholds=thresholds, seed=SEED,
             budget_usd=BUDGET, gateway=GATEWAY, block_order=blocks, condition_order=order, evaluations=evals,
             summary_sha256=filehash(OUTPUT / 'summary.json'), protocol_sha256=filehash(PROTOCOL),
             code_hashes={p: filehash(ROOT / p) for p in CODE})
    m['sha256'] = digest(m)
    write_new(path, m)
    write_new(OUTPUT / 'order-protocol-lock.json', {**{k: v for k, v in m.items() if k not in ('evaluations', 'condition_order', 'block_order')},
                                                    'evaluation_ids_sha256': digest({k: [e['eval_id'] for e in v] for k, v in evals.items()})})
    return m


def manifest():
    m = read(DIRECTORY / 'manifest.json'); expected = m.pop('sha256')
    if digest(m) != expected:
        raise ValueError('Manifest integrity failure')
    m['sha256'] = expected
    if filehash(PROTOCOL) != m['protocol_sha256'] or any(filehash(ROOT / p) != h for p, h in m['code_hashes'].items()):
        raise ValueError('Frozen protocol or code changed')
    return m


def score(model, raw, e):
    shown = {'options': e['request']['questions']['answer']['criteria'] if model == 'clef'
             else {c['value']: c['description'] for c in e['request']['questions'][0]['choices']}, 'gold': e['gold']}
    v, failure = inspect(model, raw, shown)
    if v is None:
        return None, failure
    mapping = e['display_to_original']; p = v['selected_probability']
    return {**v, 'semantic_prediction': mapping[v['answer']],
            'semantic_probabilities': {mapping[k]: x for k, x in v['probabilities'].items()},
            'correct': mapping[v['answer']] == e['original_gold'], 'referred': p <= e['threshold']}, None


class Runner:
    def __init__(self, model, m):
        self.model, self.m = model, m
        self.d = DIRECTORY / model; self.keys = credentials(); self.lock = threading.Lock()
        self.records = [read(p) for p in sorted((self.d / 'attempts').glob('*.json'))]
        self.spent = sum(r['budget_charge_usd'] for r in self.records)
        import httpx
        self.client = httpx.Client(timeout=httpx.Timeout(120, connect=15), follow_redirects=False)
        self.stop = threading.Event()

    def call(self, e):
        prior = sorted([r for r in self.records if r['eval_id'] == e['eval_id']], key=lambda r: r['attempt'])
        if any(r['request_hash'] != e['request_hash'] for r in prior):
            raise RuntimeError('Saved request differs from frozen request')
        if prior and prior[-1]['terminal']:
            return prior[-1]
        url, headers = endpoint(self.model, self.keys)
        if self.model == 'clef':
            headers = {**headers, 'cf-aig-gateway-id': GATEWAY}
        first = len(prior) + 1
        for attempt in range(first, first + 3):
            if self.stop.is_set():
                raise QuotaReached('stopped')
            stem = f"{e['eval_id']}__a{attempt}"
            intent, output = self.d / 'intents' / f'{stem}.json', self.d / 'attempts' / f'{stem}.json'
            if intent.exists() and not output.exists():
                raise RuntimeError('Unresolved intent; reconcile before running')
            reserve = upper(self.model, e['request'])
            with self.lock:
                if self.spent + reserve > self.m['budget_usd'][self.model]:
                    raise RuntimeError('Budget ceiling reached')
                self.spent += reserve
            rec = dict(eval_id=e['eval_id'], item_id=e['item_id'], cohort=e['cohort'], variant=e['variant'], model=self.model,
                       attempt=attempt, manifest_hash=self.m['sha256'], request_hash=e['request_hash'], started_at=now(),
                       budget_charge_usd=reserve)
            write_new(intent, {**rec, 'request': e['request']})
            tick = time.perf_counter(); retry = quota = False
            try:
                r = self.client.post(url, headers=headers, content=canonical(e['request']).encode('utf-8'))
                rec.update(http_status=r.status_code, latency_ms=(time.perf_counter() - tick) * 1000,
                           request_id=r.headers.get('x-request-id') or r.headers.get('cf-ray'))
                if r.status_code != 200:
                    quota = r.status_code == 429 and QUOTA in r.text
                    retry = r.status_code in TRANSIENT and not quota
                    rec.update(status='error', error_type=f'http_{r.status_code}', error=r.text[:500])
                else:
                    raw = r.json(); rec['raw_response'] = raw; rec['usage'] = usage(self.model, raw)
                    rec['value'], rec['failure'] = score(self.model, raw, e)
                    rec['status'] = 'success' if rec['value'] is not None else 'invalid'; rec['terminal'] = True
            except Exception as ex:
                retry = True; rec.update(status='error', error_type=type(ex).__name__)
            if rec.get('status') == 'error':
                rec['terminal'] = not quota and (not retry or attempt == first + 2)
            rec['finished_at'] = now()
            with self.lock:
                self.spent += (rec['usage']['standardized_usd'] if rec.get('usage') else 0) - reserve
                rec['budget_charge_usd'] = rec['usage']['standardized_usd'] if rec.get('usage') else 0.
                write_new(output, rec); self.records.append(rec)
            if quota:
                self.stop.set(); raise QuotaReached(e['eval_id'])
            if rec['terminal']:
                return rec
            time.sleep(2 ** (attempt - first + 1) * 2)

    def block(self, evals):
        return [self.call(e)['status'] for e in evals]

    def run(self, workers=4):
        by_id = {e['eval_id']: e for e in self.m['evaluations'][self.model]}
        blocks = [[by_id[f"{self.model}__{next(e['cohort'] for e in self.m['evaluations'][self.model] if e['item_id'] == b)}__{b}__{v}"]
                   for v in self.m['condition_order'][b]] for b in self.m['block_order']]
        done = Counter(); n = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for f in concurrent.futures.as_completed([pool.submit(self.block, b) for b in blocks]):
                done.update(f.result()); n += 1
                if n % 200 == 0:
                    print(json.dumps({'model': self.model, 'blocks': n, 'statuses': done, 'spent_usd': round(self.spent, 4)}), flush=True)
        print(json.dumps({'model': self.model, 'blocks': n, 'statuses': done, 'spent_usd': round(self.spent, 4)}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'run', 'status']); p.add_argument('--model', choices=list(MODELS))
    a = p.parse_args(); m = prepare()
    if a.command == 'prepare':
        print(json.dumps({'manifest': m['sha256'], 'evaluations': {k: len(v) for k, v in m['evaluations'].items()},
                          'thresholds': m['thresholds']})); return
    for model in [a.model] if a.model else list(MODELS):
        r = Runner(model, m)
        if a.command == 'status':
            last = {}
            for x in r.records:
                last[x['eval_id']] = x
            print(json.dumps({'model': model, 'terminal': sum(x['terminal'] for x in last.values()),
                              'statuses': Counter(x['status'] for x in last.values()), 'spent_usd': r.spent})); continue
        r.run()


if __name__ == '__main__':
    main()
