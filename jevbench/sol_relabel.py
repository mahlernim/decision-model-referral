"""Frozen relabeling of Korean test questions with GPT-6.1 Sol at default settings.

Uses the original content-label instruction and schema from the KorMedMCQA study. Only
the labeling model and its settings change. No answer key or model answers are supplied.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from .common import ROOT, digest, filehash, now, read, write_new
from .luna_logprob import Runner
from .study import TAG_INSTRUCTION, TAG_PROPERTIES, api_request, output_value

DIRECTORY = ROOT / 'runs/sol-relabel-v1'
OUTPUT = ROOT / 'docs/sol-relabel-v1'
MODEL = 'gpt-6.1-sol'
CAP = 16000
# USD per million tokens: input, cached input, output. Current-panel table, verified 30 September 2026.
PRICE = (2.0, .10, 10.0)
BUDGET_USD = 25
CODE = ['jevbench/sol_relabel.py', 'jevbench/luna_logprob.py', 'jevbench/study.py', 'jevbench/common.py']


def request(item):
    req = api_request('tags', item)
    req.pop('reasoning')  # Default settings: the reasoning effort is left to the service.
    req.update(model=MODEL, max_output_tokens=CAP)
    return req


def check(raw, item):
    if raw.get('incomplete_details'):
        return None, 'incomplete:' + str(raw['incomplete_details'].get('reason'))
    if any(c.get('type') == 'refusal' for o in raw.get('output', []) for c in o.get('content', []) or []):
        return None, 'refusal'
    try:
        return {**output_value(raw, 'tags', item), 'resolved_reasoning_effort': (raw.get('reasoning') or {}).get('effort')}, None
    except Exception as e:
        return None, 'invalid:' + type(e).__name__


def upper(req):
    return (len(json.dumps(req, ensure_ascii=False).encode('utf-8')) * PRICE[0] + CAP * PRICE[2]) / 1e6


def prepare():
    path = DIRECTORY / 'manifest.json'
    if path.exists():
        return manifest()
    its = [i for i in read(ROOT / 'runs/kormed-study-v1/manifest.json')['items'] if i['split'] == 'test']
    assert len(its) == 435
    m = dict(experiment='sol-relabel-v1', created_at=now(), model=MODEL, cap=CAP, prices=PRICE, budget_usd=BUDGET_USD,
             tag_instruction=TAG_INSTRUCTION, tag_properties=TAG_PROPERTIES, items=its,
             requests={i['id']: digest(request(i)) for i in its},
             protocol_sha256=filehash(OUTPUT / 'protocol.md'), code_hashes={p: filehash(ROOT / p) for p in CODE})
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
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare', 'run', 'status']); a = p.parse_args()
    m = prepare()
    runner = Runner(DIRECTORY, m, upper)
    if a.command == 'prepare':
        print(json.dumps({'manifest': m['sha256'], 'items': len(m['items'])})); return
    if a.command == 'status':
        print(json.dumps({'records': len(runner.records), 'statuses': Counter(r['status'] for r in runner.records), 'spent_usd': runner.spent})); return
    runner.run([(f"tags__{i['id']}", request(i), lambda raw, i=i: check(raw, i)) for i in m['items']])


if __name__ == '__main__':
    main()
