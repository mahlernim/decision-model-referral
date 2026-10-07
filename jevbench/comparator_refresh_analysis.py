"""Offline current-comparator analysis with fixed cohorts and explicit missing costs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .common import ROOT, digest, filehash, now, read, write_new
from .report import wilson

SEED = 2026100101
RESAMPLES = 4000
JEV_INPUT_PRICE = .042
OUTPUT = ROOT / 'docs/comparator-refresh-v1/analysis'
VIEWS = ('corrected_original', 'active')


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _cost(values, n):
    known = [float(x) for x in values if _number(x)]
    missing = len(values) - len(known)
    total = sum(known) if not missing else None
    return dict(known_components_usd=sum(known), unknown_components=missing,
                total_usd=total, per_1000_usd=total / n * 1000 if total is not None and n else None,
                complete=missing == 0, basis='standardized token-price estimate, not invoice')


def _latency(values):
    known = [float(x) for x in values if _number(x)]
    return dict(n=len(known), median_ms=float(np.median(known)) if known else None,
                p95_ms=float(np.quantile(known, .95)) if known else None)


def _ci(values):
    return [float(x) for x in np.quantile(values, [.025, .975])] if len(values) else None


def _accuracy(correct):
    n, k = len(correct), int(sum(correct))
    return dict(n=n, correct=k, accuracy=k / n if n else None,
                ci95=wilson(k, n) if n else None)


def error_cross_table(rows):
    return dict(n=len(rows),
                jev_correct=dict(destination_correct=sum(r['jev_correct'] and r['destination_correct'] for r in rows),
                                 destination_incorrect=sum(r['jev_correct'] and not r['destination_correct'] for r in rows)),
                jev_incorrect=dict(destination_correct=sum(not r['jev_correct'] and r['destination_correct'] for r in rows),
                                   destination_incorrect=sum(not r['jev_correct'] and not r['destination_correct'] for r in rows)))


def paired_difference(a, b, *, seed=SEED, resamples=RESAMPLES):
    if len(a) != len(b):
        raise ValueError('Paired arrays differ in length')
    if not len(a):
        return dict(n=0, difference=None, ci95=None)
    delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, resamples, 250):
        ix = rng.integers(0, len(delta), size=(min(250, resamples - start), len(delta)))
        samples.extend(delta[ix].mean(axis=1))
    return dict(n=len(a), difference=float(delta.mean()), ci95=_ci(samples))


def cascade(rows, threshold, *, seed=SEED, resamples=RESAMPLES):
    n = len(rows)
    if not n:
        return dict(n=0, correct=0, accuracy=None, ci95=None, referred=0,
                    reason='No valid paired observations')
    j = np.asarray([r['jev_correct'] for r in rows], dtype=bool)
    d = np.asarray([r['destination_correct'] for r in rows], dtype=bool)
    eligible = np.asarray([r['jev_valid'] for r in rows], dtype=bool)
    route = np.asarray([r['jev_valid'] and r['jev_p'] <= threshold for r in rows], dtype=bool)
    hybrid = np.where(route, d, j)
    change = (d.astype(float) - j.astype(float)) * eligible
    ne, k = int(eligible.sum()), int(route.sum())
    q_count = k / ne if ne else 0.0
    random_count = float(j.mean() + q_count * change.mean())
    costs = [r['destination_cost_usd'] for r in rows]
    missing_eligible = [r['id'] for r, flag, value in zip(rows, eligible, costs) if flag and not _number(value)]
    matched_available = not missing_eligible
    # Ineligible destination costs never enter the random-referral population.
    c = np.asarray([float(value) if flag and _number(value) else 0.0
                    for flag, value in zip(eligible, costs)])
    selected_cost, eligible_cost = float(c[route].sum()), float(c.sum())
    q_cost = selected_cost / eligible_cost if eligible_cost else 0.0
    random_cost = float(j.mean() + q_cost * change.mean()) if matched_available else None
    count_lifts, cost_lifts, jev_gains = [], [], []
    rng = np.random.default_rng(seed)
    for start in range(0, resamples, 250):
        ix = rng.integers(0, n, size=(min(250, resamples - start), n))
        baseline = j[ix].mean(axis=1)
        combined = hybrid[ix].mean(axis=1)
        changes = change[ix].mean(axis=1)
        eligible_n = eligible[ix].sum(axis=1)
        referred_n = route[ix].sum(axis=1)
        qc = np.divide(referred_n, eligible_n, out=np.zeros(len(ix)), where=eligible_n > 0)
        jev_gains.extend(combined - baseline)
        count_lifts.extend(combined - baseline - qc * changes)
        if matched_available:
            denominator = c[ix].sum(axis=1)
            numerator = (c[ix] * route[ix]).sum(axis=1)
            qm = np.divide(numerator, denominator, out=np.zeros(len(ix)), where=denominator > 0)
            cost_lifts.extend(combined - baseline - qm * changes)
    result = _accuracy(hybrid)
    result.update(threshold=threshold, eligible_n=ne, referred=k, referral_rate=k / n,
                  rescued=int((route & ~j & d).sum()), lost=int((route & j & ~d).sum()),
                  net_correct=int(hybrid.sum() - j.sum()), referred_ids=[r['id'] for r, flag in zip(rows, route) if flag],
                  vs_jev=dict(difference=float(hybrid.mean() - j.mean()), ci95=_ci(jev_gains)),
                  standardized_cost=_cost([r['jev_cost_usd'] for r in rows]
                                           + [r['destination_cost_usd'] for r, flag in zip(rows, route) if flag], n))
    result['same_count_random'] = dict(eligible_n=ne, referral_probability=q_count,
                                      expected_referred=k, expected_accuracy=random_count,
                                      gain=float(hybrid.mean() - random_count), gain_ci95=_ci(count_lifts))
    result['matched_cost_random'] = dict(
        available=matched_available, eligible_n=ne, unknown_cost_item_ids=missing_eligible,
        referral_probability=q_cost if matched_available else None,
        expected_referred=q_cost * ne if matched_available else None,
        destination_budget_usd=selected_cost if matched_available else None,
        expected_accuracy=random_cost,
        gain=float(hybrid.mean() - random_cost) if matched_available else None,
        gain_ci95=_ci(cost_lifts) if matched_available else None,
        definition='Uniform independent referral among valid Jev responses. Probability is selected destination cost divided by all eligible destination cost, recomputed within every question bootstrap. Zero eligible cost sets probability to zero. Invalid Jev responses are never referred.')
    valid_times = []
    for r, flag in zip(rows, route):
        if r['jev_valid'] and (not flag or r['destination_valid']):
            jt, dt = r['jev_latency_ms'], r['destination_latency_ms'] if flag else 0
            if _number(jt) and _number(dt):
                valid_times.append(jt + dt)
    result['simulated_valid_workflow_latency'] = _latency(valid_times)
    return result


def analyze_arm(rows, threshold, *, seed=SEED, resamples=RESAMPLES):
    if not rows or resamples < 1 or not _number(threshold) or threshold > 1:
        raise ValueError('Nonempty cohort, positive resamples and valid threshold required')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate item IDs')
    for r in rows:
        if not r['jev_valid'] and r['jev_correct']:
            raise ValueError('Invalid Jev result cannot be correct')
        if r['jev_valid'] and (not _number(r['jev_p']) or r['jev_p'] > 1):
            raise ValueError('Valid Jev result needs a probability')
        if not r['destination_valid'] and r['destination_correct']:
            raise ValueError('Invalid destination result cannot be correct')
        for key in ('jev_cost_usd', 'destination_cost_usd'):
            if r[key] is not None and not _number(r[key]):
                raise ValueError('Invalid cost value')
    valid = [r for r in rows if r['destination_valid']]
    standalone = _accuracy([r['destination_correct'] for r in rows])
    standalone.update(valid_n=len(valid), invalid_n=len(rows) - len(valid),
                      standardized_cost=_cost([r['destination_cost_usd'] for r in rows], len(rows)),
                      valid_response_latency=_latency([r['destination_latency_ms'] for r in valid]),
                      all_recorded_outcome_latency=_latency([r['destination_latency_ms'] for r in rows]))
    pair = [r for r in rows if r['jev_valid'] and r['destination_valid']]
    return dict(bootstrap=dict(resamples=resamples, seed=seed, unit='question within cohort'),
                standalone=standalone,
                standalone_vs_jev=paired_difference([r['jev_correct'] for r in rows],
                                                    [r['destination_correct'] for r in rows], seed=seed, resamples=resamples),
                cascade_full=cascade(rows, threshold, seed=seed, resamples=resamples),
                cascade_valid_pairs=cascade(pair, threshold, seed=seed, resamples=resamples),
                error_cross_table_full=error_cross_table(rows),
                error_cross_table_valid_pairs=error_cross_table(pair),
                valid_pair_excluded_ids=[r['id'] for r in rows if not r['jev_valid'] or not r['destination_valid']],
                shared_valid_wrong_ids=[r['id'] for r in pair if not r['jev_correct'] and not r['destination_correct']],
                same_wrong_option_ids=[r['id'] for r in pair if not r['jev_correct'] and not r['destination_correct']
                                       and r['jev_prediction'] == r['destination_prediction']],
                joint_incorrect_ids=[r['id'] for r in rows if not r['jev_correct'] and not r['destination_correct']])


def load_snapshot(path):
    path = Path(path).resolve()
    payload = read(path)
    if 'snapshot' in payload:
        target = (path.parent / payload['snapshot']).resolve()
        if target.parent != path.parent or filehash(target) != payload['sha256']:
            raise ValueError('Resolved snapshot pointer integrity failure')
        expected = payload['dataset_sha256']
        path, payload = target, read(target)
        if payload.get('sha256') != expected:
            raise ValueError('Resolved snapshot dataset hash mismatch')
    snapshot_hash = payload.get('sha256')
    if snapshot_hash != digest({k: v for k, v in payload.items() if k != 'sha256'}):
        raise ValueError('Resolved snapshot integrity failure')
    return payload


def load_jev_baseline(jev_input_price=JEV_INPUT_PRICE):
    """Reuse established historical joins and normalized probabilities, reprice raw input usage."""
    from .extension_analysis import assemble
    _, records, good, old_rows = assemble()
    latest = {}
    for record in sorted(records, key=lambda r: r['attempt']):
        if record['role'] == 'jev' and record['split'] == 'test' and record.get('repeat', 0) == 0:
            latest[record['item_id']] = record
    result = {}
    for row in old_rows:
        record = good.get(('jev', row['id'], 0)) or latest.get(row['id'])
        if record is None or not record.get('terminal'):
            raise ValueError('Historical Jev outcome missing: ' + row['id'])
        ni = (record.get('raw_response', {}).get('usage') or {}).get('input_tokens')
        result[row['id']] = dict(id=row['id'], cohort=row['cohort'], gold=row['gold'], source_hash=row['source_hash'],
                                jev_valid=row['jev_valid'], jev_correct=row['jev_correct'], jev_prediction=row['jev_answer'],
                                jev_p=row['p'], jev_cost_usd=ni * jev_input_price / 1e6 if type(ni) is int and ni >= 0 else None,
                                jev_input_tokens=ni, jev_latency_ms=row['jev_latency_ms'],
                                source_record_sha256=digest(record))
    return result


def analyze_snapshot(snapshot, manifest, baseline, *, seed=SEED, resamples=RESAMPLES,
                     jev_input_price=JEV_INPUT_PRICE, jev_price_verified=False):
    if snapshot.get('complete') is not True or snapshot.get('status') != 'complete':
        raise ValueError('Final analysis requires a complete resolved collection, not necessarily all valid answers')
    if snapshot['manifest_hash'] != manifest['sha256']:
        raise ValueError('Snapshot manifest mismatch')
    items = {i['id']: i for i in manifest['items'] if i['split'] == 'test'}
    expected = {(arm, i['id']) for arm, config in manifest['arms'].items()
                for i in items.values() if i['cohort'] in config['cohorts']}
    actual = [(r['arm'], r['item_id']) for r in snapshot['rows']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Resolved rows do not exactly cover planned arm/item pairs')
    attempt_groups = defaultdict(list)
    for attempt in snapshot['attempts']:
        attempt_groups[attempt['job']].append(attempt)
    result = dict(created_at=now(), snapshot_sha256=snapshot.get('sha256'), manifest_hash=manifest['sha256'],
                  complete=True, bootstrap=dict(seed=seed, resamples=resamples, unit='question within cohort'),
                  thresholds=manifest['thresholds'], collection=snapshot['summary']['collection'],
                  jev_price=dict(input_per_million=jev_input_price, output_per_million=0,
                                 status='verified_current_by_caller' if jev_price_verified else 'historical_rate_pending_current_verification'),
                  first_attempt_audit=dict(status_counts=dict(Counter(r['first_attempt']['status'] for r in snapshot['rows']))),
                  views={}, limitations=[
                      'Active results are post hoc repaired diagnostic results with mixed caps, not a fixed deployed policy.',
                      'Selected-response standardized costs omit preceding failed calls. Recovery-inclusive costs and all-attempt collection exposure are reported separately.',
                      'Unknown usage is not zero and budget reservations are not exact policy costs.',
                      'Intervals are conditional on frozen questions and thresholds. Latency is descriptive across separate collection periods.',
                      'This report covers fresh arms and historical Jev only. Historical-generation comparisons and complete frontiers are separate work.'])
    for view in VIEWS:
        groups = defaultdict(list)
        for r in snapshot['rows']:
            item = items[r['item_id']]
            old = baseline[item['id']]
            if (old['cohort'] != item['cohort'] or old['gold'] != item['gold']
                    or old['source_hash'] != digest({'question': item['question'], 'options': item['options']})
                    or r['cohort'] != item['cohort'] or r['gold'] != item['gold']):
                raise ValueError('Historical/current source join mismatch: ' + item['id'])
            selected = r[view]
            if selected['status'] == 'pending':
                raise ValueError('Complete snapshot contains a pending outcome')
            valid = selected['status'] == 'success'
            if valid and selected['prediction'] not in item['options']:
                raise ValueError('Selected result contains an invalid option')
            correct = valid and selected['prediction'] == item['gold']
            if selected.get('correct') != correct:
                raise ValueError('Selected correctness does not match the official key')
            attempts = attempt_groups[r['job']] + attempt_groups[r['job'].replace('test__', 'repair__', 1)]
            standardized = [(a.get('observed') or {}).get('standardized_usd') for a in attempts]
            groups[item['cohort'], r['arm']].append({**old,
                'destination_valid': valid, 'destination_correct': correct,
                'destination_prediction': selected.get('prediction') if valid else None,
                'destination_cost_usd': selected.get('observed', {}).get('standardized_usd'),
                'destination_latency_ms': selected.get('latency_ms'),
                'recovery_inclusive_cost_components': standardized,
                'replacement_applied': bool(r.get('replacement_applied')),
                'selected_status': selected['status']})
        cohorts = {}
        for (cohort, arm), rows in sorted(groups.items()):
            rows.sort(key=lambda r: r['id'])
            value = analyze_arm(rows, manifest['thresholds'][cohort], seed=seed, resamples=resamples)
            value['selected_status_counts'] = dict(Counter(r['selected_status'] for r in rows))
            value['recovery_inclusive_standardized_cost'] = _cost(
                [c for r in rows for c in r['recovery_inclusive_cost_components']], len(rows))
            value['recovery_inclusive_cost_scope'] = 'All original and repair test attempts for this arm, regardless of selected view; excludes development.'
            cohorts.setdefault(cohort, {'jev': _accuracy([r['jev_correct'] for r in rows]), 'arms': {}})['arms'][arm] = value
        result['views'][view] = dict(label='Post hoc repaired diagnostic' if view == 'active' else 'Frozen configuration with uniformly corrected parsing',
                                   cohorts=cohorts)
    return result


def main():
    from . import comparator_refresh as base
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=base.OUTPUT / 'resolved/active-results.json')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--jev-input-price', type=float, default=JEV_INPUT_PRICE)
    parser.add_argument('--jev-price-verified', action='store_true')
    args = parser.parse_args()
    if not _number(args.jev_input_price):
        raise ValueError('Invalid Jev input price')
    snapshot = load_snapshot(args.snapshot)
    if snapshot.get('complete') is not True:
        raise ValueError('Cannot publish final analysis of incomplete collection')
    result = analyze_snapshot(snapshot, base.manifest(), load_jev_baseline(args.jev_input_price),
                              jev_input_price=args.jev_input_price, jev_price_verified=args.jev_price_verified)
    result['analysis_code_sha256'] = filehash(__file__)
    result['sha256'] = digest(result)
    path = args.output / ('analysis-' + result['created_at'].replace(':', '-') + '.json')
    write_new(path, result)
    print(json.dumps(dict(path=str(path), sha256=result['sha256'], views=list(result['views'])), indent=2))


if __name__ == '__main__':
    main()
