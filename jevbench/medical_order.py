"""Frozen Jev medical option-position audit with contemporary repeat controls."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

from .common import ROOT, PRICE, canonical, digest, filehash, frozen, now, read, write_new
from .runner import credential, run_lock, validated

EXPERIMENT = 'medical-option-order-v1'
DIRECTORY = ROOT / 'runs' / EXPERIMENT
OUTPUT = ROOT / 'docs' / EXPERIMENT
SEED = 2026092201
MODEL = 'jev-1.13.0'
BUDGET_USD = 1.0
CONCURRENCY = 4


def checked_json(path):
    value = read(path)
    expected = value.pop('sha256')
    if digest(value) != expected:
        raise ValueError(f'Manifest hash mismatch: {Path(path).name}')
    value['sha256'] = expected
    return value


def make_evaluations(item, instruction, model=MODEL, threshold=0.626):
    labels = list(item['options'])
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid referral threshold')
    if {'kormed': 5, 'medqa': 4}.get(item['cohort']) != len(labels):
        raise ValueError('Cohort option count mismatch')
    if labels not in [list('ABCDE'), list('ABCD')] or item['gold'] not in labels:
        raise ValueError('Expected four or five ordered options and a valid key')
    if len(set(item['options'].values())) != len(labels):
        raise ValueError('Duplicate option text needs explicit handling')
    result = []
    for variant, shift in [(f'rot{k}', k) for k in range(len(labels))] + [('repeat', 0)]:
        mapping = {label: labels[(j + shift) % len(labels)] for j, label in enumerate(labels)}
        request = {'model': model, 'state': {'exam': {'question': item['question']}},
                   'questions': {'answer': {'type': 'choice', 'instructions': instruction,
                     'criteria': {label: item['options'][original] for label, original in mapping.items()}}}}
        gold = next(label for label, original in mapping.items() if original == item['gold'])
        result.append({'eval_id': f"{item['cohort']}__{item['id']}__{variant}",
                       'item_id': item['id'], 'cohort': item['cohort'], 'variant': variant,
                       'rotation': shift, 'kind': 'choice', 'gold': gold,
                       'original_gold': item['gold'], 'display_to_original': mapping,
                       'threshold': threshold, 'request': request, 'request_hash': digest(request)})
    return result


def score_response(e, raw):
    mapping = e['display_to_original']
    labels = set(e['request']['questions']['answer']['criteria'])
    if set(mapping) != labels or set(mapping.values()) != labels or mapping.get(e['gold']) != e['original_gold']:
        raise ValueError('Invalid semantic mapping')
    if raw.get('model') != e['request']['model']:
        raise ValueError('Response model mismatch')
    v = validated(e, raw)
    semantic = {mapping[k]: p for k, p in v['probabilities'].items()}
    prediction = v['prediction']
    p = v['probabilities'][prediction]
    raw_p = raw['answers']['answer']['probabilities'][prediction]
    return {**v, 'semantic_prediction': mapping[prediction], 'semantic_probabilities': semantic,
            'selected_probability': p, 'raw_selected_probability': raw_p,
            'reported_confidence': v['rank_confidence'], 'correct': mapping[prediction] == e['original_gold'],
            'referred': p <= e['threshold'], 'raw_referred': raw_p <= e['threshold'],
            'renormalized': v['probabilities_renormalized']}


def preserved_inventory():
    files = []
    for name in ('runs', 'results', 'docs', 'jevbench', 'tests'):
        for p in (ROOT / name).rglob('*'):
            if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:
                continue
            if DIRECTORY in p.parents or OUTPUT in p.parents:
                continue
            if p.name.startswith(('medical_order', 'test_medical_order')):
                continue
            files.append(p)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(filehash, files))
    return dict(zip((p.relative_to(ROOT).as_posix() for p in files), hashes))


def historical_record(directory, item, request_hash, threshold, index=None):
    candidates = []
    rows = index[item['id']] if index is not None else [(p, read(p)) for p in sorted((directory / 'attempts').glob(f"jev__{item['id']}__r0__a*.json"))]
    for p, r in rows:
        if r['request_hash'] != request_hash:
            raise ValueError('Contemporary original does not match the historical request')
        candidates.append((p, r))
    if not candidates:
        raise ValueError('Missing historical item')
    p, r = candidates[-1]
    v = r.get('value', {})
    selected = v.get('probabilities', {}).get(v.get('prediction'))
    return {'record_path': p.relative_to(ROOT).as_posix(), 'record_sha256': filehash(p),
            'status': r['status'], 'prediction': v.get('prediction'),
            'selected_probability': selected, 'referred': selected <= threshold if selected is not None else None,
            'correct': bool(v.get('correct', False)), 'request_hash': request_hash,
            'estimated_cost_usd': v.get('estimated_cost_usd', 0.0)}


PROTOCOL = '''# Medical answer-option order robustness

Exploratory amendment approved on 22 September 2026 after inspecting the earlier medical study. This protocol and its manifest are frozen before new inference. Existing study results, prompts, keys and referral thresholds remain unchanged.

## Scope and design

Evaluate Jev 1.13.0 on all 435 frozen KorMedMCQA doctor test questions and all 1,273 frozen English MedQA test questions. Korean is primary and English is a separate replication. There is no language-effect comparison. Each five-option item receives five cyclic rotations and one identical original repeat. Each four-option item receives four rotations and one repeat. Total planned calls are 8,975. No new comparator or OpenAI requests are made.

Rotation zero is a fresh contemporary original-order reference. Every original answer text appears exactly once under each displayed letter and position across rotations. Question text, instruction, option content and model version are unchanged. Predictions and probability vectors are mapped back to original semantic option identities before comparison. The repeat has exactly the same request and request hash as rotation zero.

The manipulation jointly changes option position and its letter label. It cannot isolate position from label effects, and cyclic rotations do not exhaust all possible permutations. A deterministic text screen with manual review of flagged candidates found no option-relative references that prevent rotation. Ordinary references to figure panels and clinical abbreviations were retained. This screen is not a clinical item-quality audit. No items are excluded.

## Outcomes and analysis

Primary descriptive outcomes are each source question's mean semantic answer-disagreement rate across nonzero rotations versus its contemporary original, the identical-repeat disagreement rate, and their paired difference. Analyze the two cohorts separately. Use 4,000 paired source-question bootstrap resamples with seed 2026092201. Rotations are repeated measurements, not independent questions. Also report the proportion with any rotation disagreement descriptively, without equating its multiple opportunities to a single repeat.

Secondary outcomes are referral-decision flips, selected-answer probability change, semantically aligned probability-vector total variation, confidence changes, accuracy changes, rescued and lost answers, and exploratory correctness by displayed gold position. The historical original is a separate drift comparison, never the control for the rotation effect. Accuracy uses all planned calls in each condition, counting terminal invalid or failed outcomes as incorrect. Paired answer/probability metrics use complete valid item blocks, with excluded IDs and reasons reported. An operational sensitivity treats invalid outputs as failures separately from semantic flips.

Use the original normalized selected-answer probability rule, referring p <= 0.626 in Korean and p <= 0.770 in English. Preserve raw probabilities and report raw-vector routing sensitivity. No threshold is refitted. No answer-key, prompt or exclusion changes based on new outcomes are allowed. No best-performing rotation is selected as a new primary result.

## Execution and evidence

Use the existing authorized TypeSafe credential without copying or exposing it. Hard new-charge ceiling is USD 1, with atomic cost reservations, concurrency four, and the existing USD 0.042 per million input-token accounting convention. Charges are estimates from usage, not invoices. Shuffle source-item blocks and within-item condition order using the frozen seed. Blocks execute sequentially per worker, with no simultaneous variants for the same item. Keep model, transport and concurrency constant across conditions. A seeded ten-item pilot (five per cohort, all conditions) is part of the planned cohort and only checks transport, mapping and output integrity.

Record immutable intents before calls and immutable outcomes, raw response bytes, request and response hashes, resolved model, request latency, input tokens, and cost afterward. Disable SDK retries. Retry only explicit HTTP 408, 429 or 5xx responses, at most twice. A connection timeout or interrupted intent is uncertain and stops the run without automatic resubmission. Completed invalid answers are terminal and count as failures. Authentication or model-integrity errors stop further work. Missing-usage failures retain a conservative cost reservation. Never retry a wrong answer.

Freeze prior artifact checksums and verify preservation after the audit. Publishable reports contain IDs and derived measurements, not source question collections or credentials. Raw licensed inputs remain local. Report latency and cost as observed service measurements. No public posting, deployment or clinical-safety claim is part of this audit.

Sources for interface semantics are https://docs.typesafe.ai/primitives/choice and https://docs.typesafe.ai/confidence. Source data and original thresholds inherit the frozen study manifests listed in protocol-lock.json.
'''


def prepare():
    if (DIRECTORY / 'manifest.json').exists():
        return manifest()
    km = checked_json(ROOT / 'runs/kormed-study-v1/manifest.json')
    em = checked_json(ROOT / 'runs/medical-extension-v1/manifest.json')
    kt = read(ROOT / 'runs/kormed-study-v1/threshold.json')['threshold']
    et = read(ROOT / 'runs/medical-extension-v1/threshold.json')['threshold']
    if (kt, et) != (.626, .77):
        raise ValueError('Historical thresholds changed')
    items = [{**i, 'cohort': 'kormed'} for i in km['items'] if i['split'] == 'test']
    items += [copy.deepcopy(i) for i in em['items'] if i['split'] == 'test' and i['cohort'] == 'medqa']
    if Counter(i['cohort'] for i in items) != {'kormed': 435, 'medqa': 1273}:
        raise ValueError('Source cohort sizes changed')
    evaluations = []
    history_indexes = {}
    for cohort, source_name in [('kormed', 'kormed-study-v1'), ('medqa', 'medical-extension-v1')]:
        index = defaultdict(list)
        for p in sorted((ROOT / 'runs' / source_name / 'attempts').glob('jev__*__r0__a*.json')):
            r = read(p)
            index[r['item_id']].append((p, r))
        history_indexes[cohort] = index
    for item in items:
        item['source_item_hash'] = digest({k: item[k] for k in ('id', 'question', 'options', 'gold')})
        threshold = kt if item['cohort'] == 'kormed' else et
        instruction = km['instruction'] if item['cohort'] == 'kormed' else em['instruction']
        es = make_evaluations(item, instruction, threshold=threshold)
        source_dir = ROOT / 'runs' / ('kormed-study-v1' if item['cohort'] == 'kormed' else 'medical-extension-v1')
        item['historical'] = historical_record(source_dir, item, es[0]['request_hash'], threshold, history_indexes[item['cohort']])
        evaluations.extend(es)
    rng = random.Random(SEED)
    pilot_ids = []
    for cohort in ('kormed', 'medqa'):
        pilot_ids.extend(rng.sample(sorted(i['id'] for i in items if i['cohort'] == cohort), 5))
    block_ids = sorted(i['id'] for i in items)
    rng.shuffle(block_ids)
    condition_order = {}
    for item in items:
        variants = [e['eval_id'] for e in evaluations if e['item_id'] == item['id']]
        rng.shuffle(variants)
        condition_order[item['id']] = variants
    sources = {p: filehash(ROOT / p) for p in (
        'runs/kormed-study-v1/manifest.json', 'runs/medical-extension-v1/manifest.json',
        'runs/kormed-study-v1/threshold.json', 'runs/medical-extension-v1/threshold.json')}
    m = {'experiment': EXPERIMENT, 'created_at': now(), 'seed': SEED, 'model': MODEL,
         'budget_usd': BUDGET_USD, 'input_price_usd': PRICE, 'concurrency': CONCURRENCY,
         'thresholds': {'kormed': kt, 'medqa': et}, 'source_files': sources,
         'items': items, 'evaluations': evaluations, 'block_order': block_ids,
         'condition_order': condition_order, 'pilot_ids': pilot_ids,
         'protocol_sha256': hashlib.sha256(PROTOCOL.encode('utf-8')).hexdigest(),
         'preserved_files': preserved_inventory()}
    m['sha256'] = digest(m)
    write_new(DIRECTORY / 'manifest.json', m)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'protocol.md').write_text(PROTOCOL, encoding='utf-8', newline='\n')
    write_new(OUTPUT / 'protocol-lock.json', {k: v for k, v in m.items()
              if k not in ('items', 'evaluations', 'preserved_files')})
    return manifest()


def manifest():
    m = checked_json(DIRECTORY / 'manifest.json')
    if m['model'] != MODEL or m['budget_usd'] != BUDGET_USD or m['input_price_usd'] != PRICE:
        raise ValueError('Frozen configuration changed')
    if hashlib.sha256((OUTPUT / 'protocol.md').read_bytes()).hexdigest() != m['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    for path, expected in m['source_files'].items():
        if filehash(ROOT / path) != expected:
            raise ValueError(f'Source changed: {path}')
    if len({e['eval_id'] for e in m['evaluations']}) != len(m['evaluations']):
        raise ValueError('Duplicate evaluations')
    for e in m['evaluations']:
        if digest(e['request']) != e['request_hash']:
            raise ValueError('Request hash mismatch')
    return m


def records():
    return [read(p) for p in sorted((DIRECTORY / 'attempts').glob('*.json'))]


def attempt_history(m):
    expected = {e['eval_id']: e for e in m['evaluations']}
    hist = defaultdict(list)
    for r in records():
        e = expected.get(r['eval_id'])
        if e is None or r['manifest_hash'] != m['sha256'] or r['request_hash'] != e['request_hash']:
            raise ValueError('Foreign attempt')
        hist[r['eval_id']].append(r)
    for values in hist.values():
        values.sort(key=lambda r: r['attempt'])
        if [r['attempt'] for r in values] != list(range(1, len(values) + 1)):
            raise ValueError('Noncontiguous attempts')
        if any(r['terminal'] for r in values[:-1]):
            raise ValueError('A terminal answer was retried')
    return hist


class Runner:
    def __init__(self, m):
        self.m = m
        self.history = attempt_history(m)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.pending = 0.0
        self.spent = sum(r['budget_charge_usd'] for rs in self.history.values() for r in rs)
        self.completed = sum(bool(rs and rs[-1]['terminal']) for rs in self.history.values())
        self.key = credential(ROOT / 'typesafe.env')
        self.evaluations = {e['eval_id']: e for e in m['evaluations']}
        for p in (DIRECTORY / 'intents').glob('*.json'):
            if not (DIRECTORY / 'attempts' / p.name).exists():
                raise RuntimeError('Unresolved request intent, manual reconciliation required')
        if any(r.get('fatal') for rs in self.history.values() for r in rs):
            raise RuntimeError('Prior fatal failure requires review')

    def evaluate(self, client, e):
        hist = self.history.get(e['eval_id'], [])
        if hist and hist[-1]['terminal']:
            return
        for attempt in range(len(hist) + 1, 4):
            if self.stop.is_set():
                return
            reserve = (len(canonical(e['request']).encode('utf-8')) + 4096) * PRICE * 1.25
            with self.lock:
                if self.spent + self.pending + reserve > self.m['budget_usd']:
                    self.stop.set()
                    raise RuntimeError('Budget ceiling reached')
                self.pending += reserve
            stem = e['eval_id'] + f'__a{attempt}'
            intent_path = DIRECTORY / 'intents' / (stem + '.json')
            result_path = DIRECTORY / 'attempts' / (stem + '.json')
            if intent_path.exists() or result_path.exists():
                self.stop.set()
                raise RuntimeError('Unexpected pre-existing request evidence')
            r = {k: e[k] for k in ('eval_id', 'item_id', 'cohort', 'variant', 'request_hash')}
            r.update(attempt=attempt, manifest_hash=self.m['sha256'], started_at=now(),
                     budget_charge_usd=reserve, estimated_cost_usd=None, input_tokens=None)
            write_new(intent_path, {**r, 'request': e['request']})
            tick = time.perf_counter()
            raw = None
            try:
                from typesafe_sdk import Choice
                q = e['request']['questions']['answer']
                response = client.system_one(state=e['request']['state'], model=self.m['model'],
                    questions={'answer': Choice(instructions=q['instructions'], criteria=q['criteria'])})
                r['latency_ms'] = (time.perf_counter() - tick) * 1000
                wire = response.raw_http_response.request.content
                wire_value = json.loads(wire)
                if wire_value != e['request'] or list(wire_value['questions']['answer']['criteria']) != list(q['criteria']):
                    raise RuntimeError('SDK changed planned request or criteria order')
                r['wire_request_sha256'] = hashlib.sha256(wire).hexdigest()
                r['wire_request_verified'] = True
                body = response.raw_http_response.content
                body_path = DIRECTORY / 'raw' / (stem + '.json')
                body_path.parent.mkdir(parents=True, exist_ok=True)
                with body_path.open('xb') as stream:
                    stream.write(body)
                r['response_body_sha256'] = hashlib.sha256(body).hexdigest()
                r['response_body_path'] = body_path.relative_to(ROOT).as_posix()
                raw = json.loads(body)
                r['raw_response'] = raw
                n = raw.get('usage', {}).get('input_tokens')
                if type(n) is int and n >= 0:
                    r['input_tokens'] = n
                    r['estimated_cost_usd'] = n * PRICE
                    r['budget_charge_usd'] = n * PRICE * 1.25
                if raw.get('model') != self.m['model']:
                    raise RuntimeError('Resolved model changed')
                r['value'] = score_response(e, raw)
                r.update(status='success', terminal=True, fatal=False)
            except Exception as exc:
                code = getattr(exc, 'status', getattr(exc, 'status_code', None))
                explicit_transient = code in (408, 429) or (type(code) is int and 500 <= code <= 599)
                invalid = raw is not None and isinstance(exc, (ValueError, KeyError, TypeError))
                fatal = not (explicit_transient or invalid)
                r.update(status='invalid' if invalid else 'error', terminal=not explicit_transient or attempt == 3,
                         fatal=fatal, error_type=type(exc).__name__, http_status=code,
                         error_reason=str(exc) if invalid else ('explicit_http_failure' if code else 'uncertain_transport_or_integrity_failure'),
                         latency_ms=r.get('latency_ms', (time.perf_counter() - tick) * 1000))
                if fatal:
                    self.stop.set()
            r['finished_at'] = now()
            write_new(result_path, r)
            with self.lock:
                self.pending -= reserve
                self.spent += r['budget_charge_usd']
                self.history.setdefault(e['eval_id'], []).append(r)
                if r['terminal']:
                    self.completed += 1
                if self.completed % 100 == 0 or r.get('fatal'):
                    print(json.dumps({'completed': self.completed, 'planned': len(self.m['evaluations']),
                                      'conservative_usd': round(self.spent, 6), 'last_status': r['status']}), flush=True)
            if r.get('fatal'):
                raise RuntimeError(f"Stopped after {r['error_type']}; inspect immutable attempt {stem}")
            if r['terminal']:
                return
            time.sleep(2 ** attempt)

    def block(self, item_id):
        from typesafe_sdk import TypeSafeClient, RetryPolicy
        if self.stop.is_set():
            return
        with TypeSafeClient(api_key=self.key, model=self.m['model'], retry=RetryPolicy(max_retries=0), timeout=45,
                            base_url='https://api.typesafe.ai') as client:
            for eval_id in self.m['condition_order'][item_id]:
                self.evaluate(client, self.evaluations[eval_id])


def run(phase='pilot'):
    m = manifest()
    with run_lock(DIRECTORY):
        runner = Runner(m)
        if phase == 'full':
            pilot_evals = [e for e in m['evaluations'] if e['item_id'] in m['pilot_ids']]
            if not all(runner.history.get(e['eval_id']) and runner.history[e['eval_id']][-1]['status'] == 'success' for e in pilot_evals):
                raise RuntimeError('Complete and inspect the pilot before full collection')
        ids = m['pilot_ids'] if phase == 'pilot' else m['block_order']
        ids = [uid for uid in ids if any(not runner.history.get(eid) or not runner.history[eid][-1]['terminal']
               for eid in m['condition_order'][uid])]
        stamp = now().replace(':', '-')
        write_new(DIRECTORY / 'invocations' / f'{stamp}.json', {
            'started_at': now(), 'phase': phase, 'manifest_hash': m['sha256'],
            'runner_sha256': filehash(__file__), 'validator_sha256': filehash(ROOT / 'jevbench/runner.py'),
            'sdk_version': importlib.metadata.version('typesafe-sdk'), 'python': platform.python_version(),
            'concurrency': CONCURRENCY, 'blocks': len(ids), 'timeout_seconds': 45})
        tick = time.perf_counter()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                futures = [pool.submit(runner.block, uid) for uid in ids]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        runner.stop.set()
                        for pending in futures:
                            pending.cancel()
                        raise
        finally:
            write_new(DIRECTORY / 'timing' / f'{stamp}.json', {'phase': phase, 'finished_at': now(),
                      'wall_seconds': time.perf_counter() - tick, 'completed_total': runner.completed,
                      'budget_charge_usd': runner.spent})
        return {'phase': phase, 'completed': runner.completed, 'planned': len(m['evaluations']),
                'budget_charge_usd': runner.spent}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'pilot', 'full', 'status'])
    args = p.parse_args()
    if args.command == 'prepare':
        m = prepare()
        result = {'manifest_hash': m['sha256'], 'items': len(m['items']),
                  'evaluations': len(m['evaluations']), 'preserved_files': len(m['preserved_files']),
                  'budget_usd': m['budget_usd'], 'pilot_items': len(m['pilot_ids'])}
    elif args.command == 'status':
        m = manifest(); rs = records()
        terminal = [r for r in rs if r['terminal']]
        result = {'planned': len(m['evaluations']), 'terminal': len(terminal),
                  'status_counts': dict(Counter(r['status'] for r in terminal)),
                  'observed_usd': sum(r.get('estimated_cost_usd') or 0 for r in rs),
                  'conservative_usd': sum(r['budget_charge_usd'] for r in rs)}
    else:
        result = run(args.command)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
