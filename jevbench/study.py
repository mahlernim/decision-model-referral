"""Frozen, bounded in-silico KorMedMCQA study. No original pilot mutations."""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import math
import platform
import random
import threading
import time
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient

from .common import ROOT, canonical, digest, filehash, frozen, now, read, write_new
from .runner import credential as jev_key, validated, run_lock
from .luna import credential as openai_key

DIRECTORY = ROOT / 'runs/kormed-study-v1'
OUTPUT = ROOT / 'docs/kormed-study-v1'
SEED = 2026091703
INSTRUCTION = ('`exam.question`에 대한 정답을 선택하세요. 문항에서 요구하는 조건에 따라 '
               '다섯 선택지 중 가장 적절한 하나를 고르세요.')
MODELS = {'jev': 'jev-1.13.0', 'comparator': 'gpt-5.6-sol',
          'tags': 'gpt-5.6-terra', 'review': 'gpt-5.6-sol'}
# Conservative input ceiling includes the documented 1.25x cache-write rate.
PRICES = {'jev': (0.042, 0), 'gpt-5.6-sol': (5, 20), 'gpt-5.6-terra': (2.5, 12)}
TASKS = ['diagnosis', 'investigation', 'treatment', 'prevention', 'law_policy', 'other']
TAG_PROPERTIES = {
    'task': {'type': 'string', 'enum': TASKS},
    'laboratory': {'type': 'boolean'}, 'numeric': {'type': 'boolean'},
    'negative_wording': {'type': 'boolean'}, 'korean_law_policy': {'type': 'boolean'},
    'input_status': {'type': 'string', 'enum': ['complete', 'possibly_incomplete', 'indeterminate']},
    'evidence': {'type': 'string'},
}
TAG_INSTRUCTION = '''Characterize this Korean medical examination item without solving it.
Return only the requested tags. task is the main requested action. laboratory means actual laboratory results are supplied.
numeric means answering requires numerical interpretation or calculation, not merely that age or vitals are printed.
negative_wording means the question asks for an incorrect, inappropriate, absent or exception option.
korean_law_policy means answering depends on Korean legal/administrative rules.
input_status: complete if the supplied text and options appear self-contained; possibly_incomplete only if necessary
figure/table/ECG information is referenced but absent; indeterminate if you cannot tell. A phrase such as
'results are as follows' is NOT evidence of missing information when results are supplied later in the text.
Do not assume every image mention requires seeing an image if findings are described in words.
evidence must be a short exact excerpt (at most 60 characters) from the supplied question supporting the main tag or concern.
Do not provide an answer, diagnosis, confidence in an answer, or explanation of clinical correctness.'''
REVIEW_PROPERTIES = {
    'category': {'type': 'string', 'enum': ['clinical_distinction', 'numerical_lab',
        'negative_wording', 'law_policy', 'possible_item_defect', 'indeterminate']},
    'description': {'type': 'string'}, 'evidence': {'type': 'string'},
    'key_concern': {'type': 'boolean'},
    'certainty': {'type': 'string', 'enum': ['low', 'medium', 'high']},
}
REVIEW_INSTRUCTION = '''Describe an incorrect selected option in a Korean medical exam item.
The official key is the scoring reference, not independently validated clinical truth.
Classify the observable mismatch between the selected option and keyed option, not the model's hidden reasoning.
Use indeterminate when the reason cannot be inferred. Do not invent facts, citations, guidelines, or causal explanations.
description: at most 45 English words explaining what distinction the item appears to test, with uncertainty where appropriate.
evidence: one exact excerpt from the question or options, at most 80 characters.
key_concern: flag possible ambiguity, outdated key, or absent essential evidence; do not replace the answer key.
certainty concerns this characterization, not whether the candidate model should be trusted.
No model identity or model confidence is supplied. No external tools are available.'''


def schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def api_request(role, item, selected=None):
    state = {'exam': {'question': item['question']}}
    options = item['options']
    if role == 'jev':
        return {'model': MODELS[role], 'state': state,
                'questions': {'answer': {'type': 'choice', 'instructions': INSTRUCTION, 'criteria': options}}}
    if role == 'comparator':
        instructions, properties, effort, cap = INSTRUCTION, {'answer': {'type': 'string', 'enum': list('ABCDE')}}, 'none', 128
        content = instructions + '\n\n' + canonical(state) + '\n\n' + canonical(options)
    elif role == 'tags':
        instructions, properties, effort, cap = TAG_INSTRUCTION, TAG_PROPERTIES, 'none', 400
        content = instructions + '\n\n' + canonical({'question': item['question'], 'options': options})
    else:
        instructions, properties, effort, cap = REVIEW_INSTRUCTION, REVIEW_PROPERTIES, 'low', 2048
        content = instructions + '\n\n' + canonical({'question': item['question'], 'options': options,
                    'selected_option': selected, 'official_key': item['gold']})
    return {'model': MODELS[role], 'reasoning': {'effort': effort}, 'store': False,
        'service_tier': 'default', 'max_output_tokens': cap,
        'input': [{'role': 'user', 'content': content}],
        'text': {'format': {'type': 'json_schema', 'name': role, 'strict': True, 'schema': schema(properties)}}}


def prepare():
    if (DIRECTORY / 'manifest.json').exists():
        return manifest()
    source = read(ROOT / 'results/source-lock.json')['kormed']
    files = [e for e in source['files'] if e['local'].endswith('.parquet')]
    dev_name = 'doctor/dev-00000-of-00001-57f53ad03eca2262.parquet'
    url = f"https://huggingface.co/datasets/{source['repository']}/resolve/{source['revision']}/{dev_name}"
    p = DIRECTORY / 'sources/dev.parquet'
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        r = httpx.get(url, follow_redirects=True, timeout=60); r.raise_for_status()
        with p.open('xb') as f: f.write(r.content)
    files.append({'local': p.relative_to(ROOT).as_posix(), 'url': url, 'sha256': filehash(p)})
    pilot = read(ROOT / 'data/pilot-v1/manifest.json')
    pilot_ids = {e['case_id'].replace('kormed-', 'doctor-') for e in pilot['evaluations'] if e['stage'] == 2}
    items = []
    for f in files:
        if filehash(ROOT / f['local']) != f['sha256']: raise ValueError('Source hash mismatch')
        split = 'dev' if 'dev' in f['local'] else 'test'
        for r in pq.read_table(ROOT / f['local']).to_pylist():
            uid = f"doctor-{r['year']}-{r['period']}-{r['q_number']}"
            if not r['question'].strip() or not all(r[k].strip() for k in 'ABCDE') or r['answer'] not in range(1, 6):
                raise ValueError('Structurally invalid item')
            items.append({'id': uid, 'split': split, 'year': r['year'], 'period': r['period'],
                'q_number': r['q_number'], 'question': r['question'], 'options': {k: r[k] for k in 'ABCDE'},
                'gold': 'ABCDE'[r['answer'] - 1], 'pilot_exposed': uid in pilot_ids})
    items.sort(key=lambda x: (x['split'], x['id']))
    if len({i['id'] for i in items}) != len(items): raise ValueError('Duplicate item IDs')
    dev = [i for i in items if i['split'] == 'dev']
    repeated = random.Random(SEED).sample([i['id'] for i in dev], 30)
    preserved = {p.relative_to(ROOT).as_posix(): filehash(p) for d in ['results', 'docs']
                 for p in (ROOT / d).rglob('*') if p.is_file() and 'kormed-study-v1' not in str(p)}
    m = {'experiment': 'kormed-study-v1', 'created_at': now(), 'seed': SEED, 'budget_usd': 5,
        'scope': 'Initial 90-minute in silico study; exploratory extension after pilot exposure',
        'sources': files, 'source_revision': source['revision'], 'items': items,
        'repeat_ids': repeated, 'models': MODELS, 'prices_upper_per_million': PRICES,
        'instruction': INSTRUCTION, 'tag_instruction': TAG_INSTRUCTION, 'review_instruction': REVIEW_INSTRUCTION,
        'primary': 'Full doctor test accuracy and Jev error-detection AUROC from 1 - selected probability',
        'routing': 'Freeze 20th percentile selected-probability threshold on dev; route test p <= threshold',
        'secondary': ['risk coverage at 10/20/30 percent referral', 'multiclass Brier and log loss',
            'same wrong option and shared failures', 'random referral at equal count',
            'automated tags and complete-input sensitivity', 'pilot-exposed versus unexposed questions'],
        'bootstrap': {'replicates': 4000, 'unit': 'source question', 'seed': SEED},
        'repeats': '30 randomly selected dev items, baseline plus two identical calls; no option changes',
        'preserved_files': preserved, 'git_head': None,
        'roles': {'comparator': 'Sol none, decision only', 'tags': 'Terra none, blinded to keys and predictions',
                  'review': 'Sol low, all unique incorrect options from either model; blinded identities/probabilities'},
        'review_limits': 'Automated descriptive annotation only; no clinician evaluation or causal error attribution',
        'incomplete_items': 'Full-set primary keeps all official items; automated complete-input subset is sensitivity only',
        'concurrency': 4, 'retry': 'At most 2 transient retries; no retry for incomplete/invalid outputs',
        'deadline_utc': '2026-09-17T04:44:41+00:00'}
    m['sha256'] = digest(m)
    write_new(DIRECTORY / 'manifest.json', m)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    public = {k: v for k, v in m.items() if k not in ['items', 'preserved_files']}
    public['item_index'] = [{k: v for k, v in i.items() if k not in ['question', 'options']} for i in items]
    write_new(OUTPUT / 'protocol-lock.json', public)
    return m


def manifest():
    m = read(DIRECTORY / 'manifest.json'); expected = m.pop('sha256')
    if digest(m) != expected: raise ValueError('Manifest integrity failure')
    m['sha256'] = expected
    if m['models'] != MODELS or m['instruction'] != INSTRUCTION or m['tag_instruction'] != TAG_INSTRUCTION or m['review_instruction'] != REVIEW_INSTRUCTION:
        raise ValueError('Frozen configuration changed')
    return m


def all_records():
    return [read(p) for p in sorted((DIRECTORY / 'attempts').glob('*.json'))]


def output_value(raw, role, item):
    if raw.get('status') != 'completed': raise ValueError('Response not completed')
    texts = [c['text'] for o in raw.get('output', []) if o.get('type') == 'message'
             for c in o.get('content', []) if c.get('type') == 'output_text']
    if len(texts) != 1: raise ValueError('Expected one output')
    value = json.loads(texts[0])
    props = {'answer': {'type': 'string', 'enum': list('ABCDE')}} if role == 'comparator' else TAG_PROPERTIES if role == 'tags' else REVIEW_PROPERTIES
    if set(value) != set(props): raise ValueError('Output fields mismatch')
    for k, p in props.items():
        if p['type'] == 'boolean' and type(value[k]) is not bool: raise ValueError('Invalid boolean')
        if p['type'] == 'string' and not isinstance(value[k], str): raise ValueError('Invalid string')
        if 'enum' in p and value[k] not in p['enum']: raise ValueError('Invalid enum')
    if role == 'comparator': return {'prediction': value['answer'], 'correct': value['answer'] == item['gold']}
    evidence = value['evidence']
    value['evidence_exact_match'] = bool(evidence) and any(evidence in t for t in [item['question'], *item['options'].values()])
    return value


class StudyRunner:
    def __init__(self, m):
        self.m = m; self.mutex = threading.Lock(); self.pending = 0.; self.completed = 0
        self.records = all_records(); self.spent = sum(r['budget_charge_usd'] for r in self.records)
        self.resolved = {}
        for r in self.records:
            if r['manifest_hash'] != m['sha256']: raise ValueError('Foreign record')
            if r.get('status') == 'success':
                role = r['role']; model = r['raw_response']['model']
                if role in self.resolved and self.resolved[role] != model: raise ValueError('Mixed models')
                self.resolved[role] = model
        self.oa_key = openai_key(); self.ts_key = jev_key(ROOT / 'typesafe.env')

    def evaluate(self, role, item, repeat=0, selected=None):
        request = api_request(role, item, selected)
        job = f"{role}__{item['id']}__r{repeat}" + (f'__{selected}' if selected else '')
        prior = sorted([r for r in self.records if r['job'] == job], key=lambda r: r['attempt'])
        if any(r['request_hash'] != digest(request) for r in prior):
            raise RuntimeError('Saved job request differs from frozen request')
        if prior and any(r['terminal'] for r in prior): return prior[-1]
        request_hash = digest(request)
        for attempt in range(len(prior) + 1, 4):
            stem = job + f'__a{attempt}'
            intent = DIRECTORY / 'intents' / (stem + '.json')
            output = DIRECTORY / 'attempts' / (stem + '.json')
            if intent.exists() and not output.exists(): raise RuntimeError('Uncertain request; manual reconciliation required')
            pi, po = PRICES['jev' if role == 'jev' else request['model']]
            reserve = ((len(canonical(request).encode('utf-8')) + 2048) * pi + request.get('max_output_tokens', 0) * po) / 1e6
            from datetime import datetime, timezone
            if datetime.now(timezone.utc) >= datetime.fromisoformat(self.m['deadline_utc']):
                raise RuntimeError('Study inference time limit reached')
            with self.mutex:
                if self.spent + self.pending + reserve > self.m['budget_usd']: raise RuntimeError('Study budget ceiling reached')
                self.pending += reserve
            rec = {'job': job, 'role': role, 'item_id': item['id'], 'split': item['split'], 'repeat': repeat,
                'selected_for_review': selected, 'attempt': attempt, 'manifest_hash': self.m['sha256'],
                'request_hash': request_hash, 'started_at': now(), 'budget_charge_usd': reserve}
            write_new(intent, {**rec, 'request': request})
            tick = time.perf_counter(); retry = False
            try:
                if role == 'jev':
                    with TypeSafeClient(api_key=self.ts_key, model=MODELS['jev'], retry=RetryPolicy(max_retries=0), timeout=45) as client:
                        q = request['questions']['answer']
                        tick = time.perf_counter()
                        response = client.system_one(state=request['state'], questions={'answer': Choice(instructions=q['instructions'], criteria=q['criteria'])}, model=MODELS['jev'])
                        raw = response.raw_http_response.json()
                    rec['latency_ms'] = (time.perf_counter() - tick) * 1000
                    rec['raw_response'] = raw
                    score = validated({'kind': 'choice', 'gold': item['gold'], 'request': request}, raw)
                    rec['value'] = score; rec['budget_charge_usd'] = score['estimated_cost_usd']
                else:
                    with httpx.Client(timeout=90) as client:
                        tick = time.perf_counter()
                        response = client.post('https://api.openai.com/v1/responses', headers={'Authorization': 'Bearer ' + self.oa_key}, json=request)
                        response.raise_for_status(); raw = response.json()
                    rec['latency_ms'] = (time.perf_counter() - tick) * 1000
                    rec['raw_response'] = raw
                    u = raw['usage']; ni, no = u['input_tokens'], u['output_tokens']
                    nc = u.get('input_tokens_details', {}).get('cached_tokens', 0)
                    nr = u.get('output_tokens_details', {}).get('reasoning_tokens', 0)
                    if any(type(v) is not int or v < 0 for v in [ni, no, nc, nr]) or nc > ni or nr > no: raise ValueError('Invalid usage')
                    # Report uncached base-price estimate plus conservative cache-write upper bound.
                    base_pi = pi / 1.25
                    rec['estimated_cost_usd'] = ((ni - nc) * base_pi + nc * base_pi * .1 + no * po) / 1e6
                    rec['budget_charge_usd'] = (ni * pi + no * po) / 1e6
                    rec['value'] = output_value(raw, role, item)
                with self.mutex:
                    if role in self.resolved and self.resolved[role] != raw['model']: raise ValueError('Resolved model changed')
                    self.resolved[role] = raw['model']
                rec.update(status='success', terminal=True)
            except Exception as e:
                status = getattr(getattr(e, 'response', None), 'status_code', None) or getattr(e, 'status_code', None)
                retry = status in [408, 429, 500, 502, 503, 504] or isinstance(e, (httpx.TimeoutException, httpx.ConnectError)) or type(e).__name__ in ['TypeSafeAPIConnectionError', 'TypeSafeAPITimeoutError']
                rec.update(status='error', terminal=not retry or attempt == 3, error_type=type(e).__name__, http_status=status)
                rec.setdefault('latency_ms', (time.perf_counter() - tick) * 1000)
            rec['finished_at'] = now()
            write_new(output, rec)
            with self.mutex:
                self.pending -= reserve; self.spent += rec['budget_charge_usd']; self.records.append(rec); self.completed += 1
                if self.completed % 20 == 0:
                    print(json.dumps({'completed_this_run': self.completed, 'budget_upper_usd': round(self.spent, 4), 'last_role': role}), flush=True)
            if rec['terminal']: return rec
            time.sleep(2 ** attempt)

    def run(self, phase, limit=None):
        items = self.m['items']; jobs = []
        if phase in ['dev', 'test']:
            selected = [i for i in items if i['split'] == phase]
            if limit is not None: selected = selected[:limit]
            for i in selected:
                roles = ['jev', 'comparator']
                random.Random(SEED + int(digest(i['id'])[:8], 16)).shuffle(roles)
                jobs.extend((r, i, 0, None) for r in roles)
        elif phase == 'repeat':
            jobs = [('jev', i, r, None) for i in items if i['id'] in self.m['repeat_ids'] for r in [1, 2]]
        elif phase == 'tags':
            jobs = [('tags', i, 0, None) for i in items if i['split'] == 'test']
        elif phase == 'review':
            wrong = {(r['item_id'], r['value']['prediction']) for r in self.records if r['role'] in ['jev', 'comparator']
                     and r['split'] == 'test' and r['repeat'] == 0 and r['status'] == 'success' and not r['value']['correct']}
            jobs = [('review', i, 0, a) for i in items for uid, a in sorted(wrong) if uid == i['id']]
        else: raise ValueError('Unknown phase')
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self.evaluate, *j) for j in jobs]
            for f in concurrent.futures.as_completed(futures): f.result()
        print(json.dumps({'phase': phase, 'jobs': len(jobs), 'upper_cost_usd': self.spent}), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare', 'dev', 'test', 'repeat', 'tags', 'review', 'status'])
    p.add_argument('--limit', type=int); a = p.parse_args()
    if a.command == 'prepare':
        m = prepare(); print(json.dumps({'manifest_hash': m['sha256'], 'n': len(m['items'])})); return
    m = manifest()
    if a.command == 'status':
        from collections import Counter
        rr = all_records(); print(json.dumps({'counts': dict(Counter(r['role'] + '/' + r['split'] + '/' + r['status'] for r in rr)),
            'upper_cost_usd': sum(r['budget_charge_usd'] for r in rr)}, indent=2)); return
    if a.command == 'test':
        threshold = read(DIRECTORY / 'threshold.json')
        if threshold['manifest_hash'] != m['sha256'] or threshold['n_dev'] != 164:
            raise RuntimeError('Development threshold not frozen for this manifest')
    with run_lock(DIRECTORY):
        write_new(DIRECTORY / 'invocations' / (now().replace(':', '-') + '.json'), {'started_at': now(),
            'command': a.command, 'limit': a.limit, 'python': platform.python_version(),
            'sdk': importlib.metadata.version('typesafe-sdk'), 'code_sha256': filehash(__file__),
            'concurrency': 4, 'manifest_hash': m['sha256']})
        StudyRunner(m).run(a.command, a.limit)


if __name__ == '__main__': main()
