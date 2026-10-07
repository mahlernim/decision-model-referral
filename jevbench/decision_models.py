"""Frozen OpenAI Decisions and Cloudflare Clef collection on the medical examination cohorts.

Each request carries Jev's state, instruction, option letters, option texts and submitted
order, so only the decision model and its interface change. The selected-option probability is
the normalized probability of the returned choice, the same rule used for Jev.
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

from .common import ROOT, canonical, digest, filehash, now, read, write_new

DIRECTORY = ROOT / 'runs/decision-models-v1'
PROBE = ROOT / 'runs/decision-models-probe-v1'
OUTPUT = ROOT / 'docs/decision-models-v1'
KOREAN_INSTRUCTION = ('`exam.question`에 대한 정답을 선택하세요. 문항에서 요구하는 조건에 따라 '
                      '다섯 선택지 중 가장 적절한 하나를 고르세요.')
ENGLISH_INSTRUCTION = ('Select the correct answer to `exam.question`. Following the conditions requested in the '
                       'question, choose the single most appropriate of the four options.')
MODELS = {'decisions': 'gpt-6-luna', 'clef': '@cf/cloudflare/clef'}
# USD per million input tokens. Neither endpoint bills output. Verified 6 and 7 October 2026.
PRICE = {'decisions': 0.10, 'clef': 0.24}
# Returned probabilities are rounded to two (Decisions) and four (Clef) decimal places.
DECIMALS = {'decisions': 2, 'clef': 4}
BUDGET_USD = 5
TRANSIENT = {408, 409, 429, 500, 502, 503, 504, 524}
CODE = ['jevbench/decision_models.py', 'jevbench/common.py']


def items():
    kormed = [{**i, 'cohort': 'kormed'} for i in read(ROOT / 'runs/kormed-study-v1/manifest.json')['items']]
    medqa = [i for i in read(ROOT / 'runs/medical-extension-v1/manifest.json')['items'] if i['cohort'] == 'medqa']
    out = sorted(kormed + medqa, key=lambda i: (i['cohort'], i['split'], i['id']))
    assert Counter((i['cohort'], i['split']) for i in out) == {('kormed', 'dev'): 164, ('kormed', 'test'): 435,
                                                               ('medqa', 'dev'): 1272, ('medqa', 'test'): 1273}
    return out


def instruction(item):
    return KOREAN_INSTRUCTION if item['cohort'] == 'kormed' else ENGLISH_INSTRUCTION


def request(model, item):
    state = {'exam': {'question': item['question']}}
    if model == 'clef':
        return {'model': 'clef', 'state': state, 'questions': {'answer': {
            'type': 'choice', 'instructions': instruction(item), 'criteria': item['options']}}}
    return {'model': MODELS[model], 'input': canonical(state), 'questions': [{
        'type': 'choice', 'name': 'answer', 'instructions': instruction(item),
        'choices': [{'value': k, 'description': v} for k, v in item['options'].items()]}]}


def credentials():
    values = {}
    for line in (ROOT / 'typesafe.env').read_text(encoding='utf-8-sig').splitlines():
        if '=' in line and not line.strip().startswith('#'):
            k, v = line.strip().removeprefix('export ').split('=', 1)
            values[k.strip()] = v.strip().strip('"\'')
    token = values.get('CLOUDFLARE_API_TOKEN') or values.get('CLOUDFLARE_AUTH_TOKEN')
    if not (values.get('OPENAI_API_KEY') and values.get('CLOUDFLARE_ACCOUNT_ID') and token):
        raise RuntimeError('Required credential missing from credential file')
    return dict(openai=values['OPENAI_API_KEY'], account=values['CLOUDFLARE_ACCOUNT_ID'], cloudflare=token)


def endpoint(model, keys):
    if model == 'clef':
        return (f"https://api.cloudflare.com/client/v4/accounts/{keys['account']}/ai/run/{MODELS['clef']}",
                {'Authorization': 'Bearer ' + keys['cloudflare'], 'Content-Type': 'application/json'})
    return ('https://api.openai.com/v1/decisions',
            {'Authorization': 'Bearer ' + keys['openai'], 'Content-Type': 'application/json'})


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Out-of-range or nonfinite probability')
    return float(value)


def inspect(model, raw, item):
    """Return (value, failure reason). Invalid outputs are terminal failures on the full denominator."""
    try:
        if model == 'clef':
            if raw.get('success') is not True or raw.get('errors'):
                return None, 'envelope_not_success'
            body = raw.get('result') or {}
            if body.get('model') not in ('clef', MODELS['clef']):
                return None, 'unexpected_model'
            answers = body.get('answers') or {}
            if set(answers) != {'answer'} or answers['answer'].get('type') != 'choice':
                return None, 'unexpected_answer_fields'
            answer = answers['answer']; vector = answer.get('probabilities')
            if not isinstance(vector, dict):
                return None, 'malformed_vector'
            probs = {k: probability(v) for k, v in vector.items()}
            returned_order = list(vector)
        else:
            body = raw
            reported = body.get('model')
            if not isinstance(reported, str) or not (reported == MODELS[model] or reported.startswith(MODELS[model] + '-')):
                return None, 'unexpected_model'
            answers = body.get('answers')
            if not isinstance(answers, list) or len(answers) != 1:
                return None, 'expected_one_answer'
            answer = answers[0]
            if answer.get('type') != 'choice' or answer.get('name') != 'answer':
                return None, 'unexpected_answer_fields'
            vector = answer.get('probabilities')
            if not isinstance(vector, list) or not all(isinstance(p, dict) and set(p) == {'value', 'probability'} for p in vector):
                return None, 'malformed_vector'
            probs = {p['value']: probability(p['probability']) for p in vector}
            if len(probs) != len(vector):
                return None, 'duplicate_option'
            returned_order = [p['value'] for p in vector]
    except (ValueError, AttributeError, TypeError):
        return None, 'malformed_vector'
    options = list(item['options'])
    choice = answer.get('choice')
    if set(probs) != set(options) or choice not in options:
        return None, 'option_mismatch'
    mass = math.fsum(probs.values())
    if abs(mass - 1) > len(options) * 0.5 * 10 ** -DECIMALS[model] + 1e-12:
        return None, 'mass_outside_rounding_bound'
    if probs[choice] < max(probs.values()) - 1e-12:
        return None, 'choice_below_maximum'
    normalized = {k: v / mass for k, v in probs.items()}
    confidence = answer.get('confidence')
    return dict(answer=choice, correct=choice == item['gold'], probabilities=normalized, raw_probabilities=probs,
                probability_sum=mass, selected_probability=normalized[choice], returned_order=returned_order,
                native_confidence=confidence if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else None,
                tied_at_maximum=sum(v >= probs[choice] - 1e-12 for v in probs.values()) > 1), None


def usage(model, raw):
    u = (raw.get('result') if model == 'clef' else raw) or {}
    u = u.get('usage') or {}
    ni, no = u.get('input_tokens'), u.get('output_tokens', 0)
    if type(ni) is not int or ni <= 0 or (no is not None and (type(no) is not int or no < 0)):
        return None
    return dict(input_tokens=ni, output_tokens=no, standardized_usd=ni * PRICE[model] / 1e6)


def upper(model, req):
    # Byte count bounds the token count for these inputs. Only input is billed.
    return len(canonical(req).encode('utf-8')) * PRICE[model] / 1e6


class Runner:
    """One terminal record per job. Transient HTTP or transport failures are retried at most twice."""

    def __init__(self, model, directory, manifest_hash, budget):
        self.model, self.d, self.hash, self.budget = model, directory, manifest_hash, budget
        self.keys = credentials(); self.lock = threading.Lock()
        self.records = [read(p) for p in sorted((directory / 'attempts').glob('*.json'))]
        self.spent = sum(r['budget_charge_usd'] for r in self.records); self.pending = 0.
        self.client = httpx.Client(timeout=httpx.Timeout(120, connect=15), follow_redirects=False)

    def call(self, job, req, item):
        prior = sorted([r for r in self.records if r['job'] == job], key=lambda r: r['attempt'])
        if any(r['request_hash'] != digest(req) for r in prior):
            raise RuntimeError('Saved request differs from frozen request')
        if prior and prior[-1]['terminal']:
            return prior[-1]
        url, headers = endpoint(self.model, self.keys)
        for attempt in range(len(prior) + 1, 4):
            stem = f'{job}__a{attempt}'
            intent, output = self.d / 'intents' / f'{stem}.json', self.d / 'attempts' / f'{stem}.json'
            if intent.exists() and not output.exists():
                raise RuntimeError('Unresolved intent; reconcile before running')
            reserve = upper(self.model, req)
            with self.lock:
                if self.spent + self.pending + reserve > self.budget:
                    raise RuntimeError('Budget ceiling reached')
                self.pending += reserve
            rec = dict(job=job, model=self.model, attempt=attempt, manifest_hash=self.hash, request_hash=digest(req),
                       started_at=now(), budget_charge_usd=reserve)
            write_new(intent, {**rec, 'request': req})
            tick = time.perf_counter(); retry = False
            try:
                r = self.client.post(url, headers=headers, content=canonical(req).encode('utf-8'))
                rec['http_status'] = r.status_code
                rec['request_id'] = r.headers.get('x-request-id') or r.headers.get('cf-ray')
                rec['latency_ms'] = (time.perf_counter() - tick) * 1000
                if r.status_code != 200:
                    retry = r.status_code in TRANSIENT
                    rec.update(status='error', error_type=f'http_{r.status_code}', error=r.text[:500])
                else:
                    raw = r.json()
                    rec['raw_response'] = raw
                    rec['usage'] = usage(self.model, raw)
                    if rec['usage']:
                        rec['budget_charge_usd'] = rec['usage']['standardized_usd']
                    rec['value'], rec['failure'] = inspect(self.model, raw, item)
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
            time.sleep(2 ** attempt * 2)

    def run(self, jobs, workers):
        done = Counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for f in concurrent.futures.as_completed([pool.submit(self.call, *j) for j in jobs]):
                done[f.result()['status']] += 1
                if sum(done.values()) % 200 == 0:
                    print(json.dumps({'model': self.model, 'done': sum(done.values()), 'statuses': done,
                                      'spent_usd': round(self.spent, 4)}), flush=True)
        print(json.dumps({'model': self.model, 'jobs': len(jobs), 'statuses': done, 'spent_usd': round(self.spent, 4)}), flush=True)


def probe():
    """Two development questions per cohort and model, outside the manifest. No test question is sent."""
    picks = []
    for cohort in ['kormed', 'medqa']:
        dev = [i for i in items() if i['cohort'] == cohort and i['split'] == 'dev']
        picks += [dev[0], max(dev, key=lambda i: len(i['question']))]
    for model in MODELS:
        runner = Runner(model, PROBE / model, 'probe', 0.05)
        for i in picks:
            rec = runner.call(f"{model}__{i['id']}", request(model, i), i)
            v = rec.get('value') or {}
            print(json.dumps({'model': model, 'item': i['id'], 'status': rec['status'], 'failure': rec.get('failure'),
                              'error': rec.get('error_type'), 'latency_ms': round(rec.get('latency_ms') or 0),
                              'raw_probabilities': v.get('raw_probabilities'), 'native_confidence': v.get('native_confidence'),
                              'usage': rec.get('usage')}, ensure_ascii=False), flush=True)


def prepare():
    path = DIRECTORY / 'manifest.json'
    if path.exists():
        return manifest()
    its = items()
    m = dict(experiment='decision-models-v1', created_at=now(), models=MODELS, prices=PRICE, decimals=DECIMALS,
             budget_usd=BUDGET_USD, items=its,
             requests={model: {i['id']: digest(request(model, i)) for i in its} for model in MODELS},
             protocol_sha256=filehash(OUTPUT / 'protocol.md'), code_hashes={p: filehash(ROOT / p) for p in CODE},
             probe_files=sorted(str(p.relative_to(PROBE)).replace('\\', '/') for p in PROBE.rglob('attempts/*.json')))
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
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['probe', 'prepare', 'dev', 'test', 'status'])
    p.add_argument('--model', choices=list(MODELS))
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.command == 'probe':
        probe(); return
    m = prepare()
    if a.command == 'prepare':
        print(json.dumps({'manifest': m['sha256'], 'items': len(m['items'])})); return
    for model in [a.model] if a.model else list(MODELS):
        runner = Runner(model, DIRECTORY / model, m['sha256'], BUDGET_USD / len(MODELS))
        if a.command == 'status':
            print(json.dumps({'model': model, 'records': len(runner.records), 'terminal_jobs': len({r['job'] for r in runner.records if r['terminal']}),
                              'statuses': Counter(r['status'] for r in runner.records if r['terminal']), 'spent_usd': runner.spent}))
            continue
        if a.command == 'test' and any(not any(r['job'] == f"{model}__{i['id']}" and r['terminal'] for r in runner.records)
                                       for i in m['items'] if i['split'] == 'dev'):
            raise RuntimeError('Development collection must finish before test')
        jobs = [(f"{model}__{i['id']}", request(model, i), i) for i in m['items'] if i['split'] == a.command]
        for i in m['items']:
            if digest(request(model, i)) != m['requests'][model][i['id']]:
                raise RuntimeError('Request differs from manifest')
        runner.run(jobs, a.workers)


if __name__ == '__main__':
    main()
