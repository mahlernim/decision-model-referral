"""Frozen GPT-6 Luna answer-letter probability baseline, with resumable evidence.

Uses the exact current-panel Luna none request and adds token log probabilities. The
selected-option probability is the renormalized probability of the generated answer
letter among the valid option letters at the answer token.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import threading
import time
from collections import Counter

import httpx

from .common import ROOT, digest, filehash, now, read, write_new
from .luna import credential
from .medical_extension import api_request

DIRECTORY = ROOT / 'runs/luna-logprob-v1'
OUTPUT = ROOT / 'docs/luna-logprob-v1'
MODEL = 'gpt-6-luna'
CAP = 128
TOP_LOGPROBS = 20
MIN_VALID_MASS = 0.5
# USD per million tokens: input, cached input, output. Current-panel table, verified 30 September 2026.
PRICE = (.10, .01, .50)
BUDGET_USD = 5
TRANSIENT = {408, 429, 500, 502, 503, 504, 524}
CODE = ['jevbench/luna_logprob.py', 'jevbench/medical_extension.py', 'jevbench/common.py']


def items():
    kormed = [{**i, 'cohort': 'kormed'} for i in read(ROOT / 'runs/kormed-study-v1/manifest.json')['items']]
    medqa = [i for i in read(ROOT / 'runs/medical-extension-v1/manifest.json')['items'] if i['cohort'] == 'medqa']
    out = sorted(kormed + medqa, key=lambda i: (i['cohort'], i['split'], i['id']))
    assert Counter((i['cohort'], i['split']) for i in out) == {('kormed', 'dev'): 164, ('kormed', 'test'): 435,
                                                               ('medqa', 'dev'): 1272, ('medqa', 'test'): 1273}
    return out


def request(item):
    req = api_request('luna', item)
    req.update(model=MODEL, reasoning={'effort': 'none'}, max_output_tokens=CAP,
               include=['message.output_text.logprobs'], top_logprobs=TOP_LOGPROBS)
    return req


def inspect(raw, item):
    """Return (value, failure reason). Value holds the answer and its letter distribution."""
    if raw.get('status') != 'completed' or raw.get('incomplete_details'):
        return None, 'not_completed'
    parts = [c for o in raw.get('output', []) if o.get('type') == 'message'
             for c in o.get('content', []) if c.get('type') == 'output_text']
    if len(parts) != 1:
        return None, 'expected_one_answer'
    try:
        answer = json.loads(parts[0]['text'])['answer']
    except (json.JSONDecodeError, KeyError, TypeError):
        return None, 'invalid_json'
    if answer not in item['options']:
        return None, 'invalid_answer'
    tokens = parts[0].get('logprobs') or []
    prefix, position = '', None
    for n, t in enumerate(tokens):
        if t['token'] == answer and prefix.endswith('"answer":"'):
            position = n
            break
        prefix += t['token']
    if position is None:
        return None, 'answer_token_not_found'
    raw_probs = {}
    for alt in tokens[position].get('top_logprobs', []):
        if alt['token'] in item['options'] and alt['token'] not in raw_probs:
            raw_probs[alt['token']] = math.exp(alt['logprob'])
    mass = sum(raw_probs.values())
    if answer not in raw_probs or mass < MIN_VALID_MASS:
        return None, 'insufficient_letter_mass'
    probs = {k: raw_probs.get(k, 0.0) / mass for k in item['options']}
    return dict(answer=answer, correct=answer == item['gold'], probabilities=probs, valid_letter_mass=mass,
                selected_probability=probs[answer], argmax_matches=max(probs, key=probs.get) == answer), None


def usage(raw):
    u = raw.get('usage') or {}
    ni, no = u.get('input_tokens'), u.get('output_tokens')
    nc = (u.get('input_tokens_details') or {}).get('cached_tokens', 0)
    if not all(type(x) is int and x >= 0 for x in [ni, no, nc]):
        return None
    return dict(input_tokens=ni, output_tokens=no, cached_tokens=nc,
                billed_usd=((ni - nc) * PRICE[0] + nc * PRICE[1] + no * PRICE[2]) / 1e6,
                standardized_usd=(ni * PRICE[0] + no * PRICE[2]) / 1e6)


class Runner:
    """One terminal record per job. Transient HTTP failures are retried at most twice."""

    def __init__(self, directory, manifest, price_upper):
        self.d, self.m, self.price_upper = directory, manifest, price_upper
        self.key = credential(); self.lock = threading.Lock()
        self.records = [read(p) for p in sorted((directory / 'attempts').glob('*.json'))]
        self.spent = sum(r['budget_charge_usd'] for r in self.records); self.pending = 0.

    def call(self, job, req, check):
        prior = sorted([r for r in self.records if r['job'] == job], key=lambda r: r['attempt'])
        if any(r['request_hash'] != digest(req) for r in prior):
            raise RuntimeError('Saved request differs from frozen request')
        if prior and prior[-1]['terminal']:
            return prior[-1]
        for attempt in range(len(prior) + 1, 4):
            stem = f'{job}__a{attempt}'
            intent, output = self.d / 'intents' / f'{stem}.json', self.d / 'attempts' / f'{stem}.json'
            if intent.exists() and not output.exists():
                raise RuntimeError('Unresolved intent; reconcile before running')
            reserve = self.price_upper(req)
            with self.lock:
                if self.spent + self.pending + reserve > self.m['budget_usd']:
                    raise RuntimeError('Budget ceiling reached')
                self.pending += reserve
            rec = dict(job=job, attempt=attempt, manifest_hash=self.m['sha256'], request_hash=digest(req),
                       started_at=now(), budget_charge_usd=reserve)
            write_new(intent, {**rec, 'request': req})
            tick = time.perf_counter(); retry = False
            try:
                r = httpx.post('https://api.openai.com/v1/responses', json=req, timeout=600,
                               headers={'Authorization': 'Bearer ' + self.key})
                rec['http_status'] = r.status_code
                if r.status_code in TRANSIENT:
                    retry = True
                    raise RuntimeError(f'transient {r.status_code}')
                r.raise_for_status()
                raw = r.json()
                rec['latency_ms'] = (time.perf_counter() - tick) * 1000
                rec['raw_response'] = raw
                rec['usage'] = usage(raw)
                if rec['usage']:
                    rec['budget_charge_usd'] = rec['usage']['standardized_usd']
                rec['resolved_model'] = raw.get('model')
                rec['value'], rec['failure'] = check(raw)
                rec['status'] = 'success' if rec['value'] is not None else 'invalid'
                rec['terminal'] = True
            except (httpx.TimeoutException, httpx.TransportError) as e:
                retry = True; rec.update(status='error', error_type=type(e).__name__)
            except Exception as e:
                rec.update(status='error', error_type=type(e).__name__, error=str(e)[:300])
            if rec.get('status') == 'error':
                rec['terminal'] = not retry or attempt == 3
            rec['finished_at'] = now()
            write_new(output, rec)
            with self.lock:
                self.pending -= reserve; self.spent += rec['budget_charge_usd']; self.records.append(rec)
            if rec['terminal']:
                return rec
            time.sleep(2 ** attempt)

    def run(self, jobs, workers=4):
        done = Counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for f in concurrent.futures.as_completed([pool.submit(self.call, *j) for j in jobs]):
                done[f.result()['status']] += 1
                if sum(done.values()) % 200 == 0:
                    print(json.dumps({'done': sum(done.values()), 'statuses': done, 'spent_usd': round(self.spent, 4)}), flush=True)
        print(json.dumps({'jobs': len(jobs), 'statuses': done, 'spent_usd': round(self.spent, 4)}), flush=True)


def upper(req):
    return (len(json.dumps(req, ensure_ascii=False).encode('utf-8')) * PRICE[0] + CAP * PRICE[2]) / 1e6


def prepare():
    path = DIRECTORY / 'manifest.json'
    if path.exists():
        return manifest()
    its = items()
    m = dict(experiment='luna-logprob-v1', created_at=now(), model=MODEL, cap=CAP, top_logprobs=TOP_LOGPROBS,
             min_valid_mass=MIN_VALID_MASS, prices=PRICE, budget_usd=BUDGET_USD, items=its,
             requests={i['id']: digest(request(i)) for i in its},
             protocol_sha256=filehash(OUTPUT / 'protocol.md'), code_hashes={p: filehash(ROOT / p) for p in CODE},
             probe_files=sorted(p.name for p in (ROOT / 'runs/luna-logprob-probe-v1').glob('*.json')))
    m['sha256'] = digest(m)
    write_new(path, m)
    write_new(OUTPUT / 'protocol-lock.json', {**{k: v for k, v in m.items() if k not in ('items', 'requests')},
                                              'item_ids_sha256': digest([i['id'] for i in its])})
    return m


def manifest():
    m = read(DIRECTORY / 'manifest.json'); expected = m.pop('sha256')
    if digest(m) != expected:
        raise ValueError('Manifest integrity failure')
    m['sha256'] = expected
    if filehash(OUTPUT / 'protocol.md') != m['protocol_sha256'] or any(filehash(ROOT / p) != h for p, h in m['code_hashes'].items()):
        raise ValueError('Frozen protocol or code changed')
    return m


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare', 'dev', 'test', 'status']); a = p.parse_args()
    m = prepare()
    if a.command == 'prepare':
        print(json.dumps({'manifest': m['sha256'], 'items': len(m['items'])})); return
    runner = Runner(DIRECTORY, m, upper)
    if a.command == 'status':
        print(json.dumps({'records': len(runner.records), 'statuses': Counter(r['status'] for r in runner.records), 'spent_usd': runner.spent})); return
    if a.command == 'test' and any(not any(r['job'] == f"luna__{i['id']}" and r['terminal'] for r in runner.records)
                                   for i in m['items'] if i['split'] == 'dev'):
        raise RuntimeError('Development collection must finish before test')
    jobs = [(f"luna__{i['id']}", request(i), lambda raw, i=i: inspect(raw, i)) for i in m['items'] if i['split'] == a.command]
    runner.run(jobs)


if __name__ == '__main__':
    main()
