"""Resume Clef test jobs interrupted by the Cloudflare daily quota, as fixed in docs/decision-models-v1/quota-amendment.md."""
import json
import time
from collections import Counter

from .common import digest, now, write_new
from .decision_models import DIRECTORY, OUTPUT, TRANSIENT, Runner, canonical, endpoint, inspect, manifest, request, upper, usage
from .common import filehash

QUOTA = 'daily free allocation'
GATEWAY = 'clef-gateway'
AMENDMENT = OUTPUT / 'quota-amendment.md'


class QuotaReached(RuntimeError):
    pass


def latest(runner):
    out = {}
    for r in runner.records:
        if r['job'] not in out or r['attempt'] > out[r['job']]['attempt']:
            out[r['job']] = r
    return out


def eligible(runner, m):
    last = latest(runner); jobs = []
    for i in m['items']:
        if i['split'] != 'test':
            continue
        job = f"clef__{i['id']}"; r = last.get(job)
        if r is None or (r['status'] == 'error' and r.get('http_status') == 429 and QUOTA in (r.get('error') or '')):
            jobs.append((job, i))
    return jobs


def call(runner, job, item):
    req = request('clef', item)
    prior = sorted([r for r in runner.records if r['job'] == job], key=lambda r: r['attempt'])
    if any(r['request_hash'] != digest(req) for r in prior):
        raise RuntimeError('Saved request differs from frozen request')
    url, headers = endpoint('clef', runner.keys)
    # Routes billing through the account's unified-billing AI Gateway. Same endpoint, model and body.
    headers = {**headers, 'cf-aig-gateway-id': GATEWAY}
    first = len(prior) + 1
    for attempt in range(first, first + 3):
        stem = f'{job}__a{attempt}'
        intent, output = runner.d / 'intents' / f'{stem}.json', runner.d / 'attempts' / f'{stem}.json'
        reserve = upper('clef', req)
        if runner.spent + reserve > runner.budget:
            raise RuntimeError('Budget ceiling reached')
        rec = dict(job=job, model='clef', attempt=attempt, manifest_hash=runner.hash, request_hash=digest(req),
                   started_at=now(), budget_charge_usd=reserve, amendment='quota-amendment',
                   amendment_sha256=filehash(AMENDMENT))
        write_new(intent, {**rec, 'request': req})
        tick = time.perf_counter(); retry = False; quota = False
        try:
            r = runner.client.post(url, headers=headers, content=canonical(req).encode('utf-8'))
            rec['http_status'] = r.status_code
            rec['request_id'] = r.headers.get('cf-ray')
            rec['latency_ms'] = (time.perf_counter() - tick) * 1000
            if r.status_code != 200:
                quota = r.status_code == 429 and QUOTA in r.text
                retry = r.status_code in TRANSIENT and not quota
                rec.update(status='error', error_type=f'http_{r.status_code}', error=r.text[:500])
            else:
                raw = r.json()
                rec['raw_response'] = raw
                rec['usage'] = usage('clef', raw)
                if rec['usage']:
                    rec['budget_charge_usd'] = rec['usage']['standardized_usd']
                rec['value'], rec['failure'] = inspect('clef', raw, item)
                rec['status'] = 'success' if rec['value'] is not None else 'invalid'
                rec['terminal'] = True
        except Exception as e:
            retry = True; rec.update(status='error', error_type=type(e).__name__)
        if rec.get('status') == 'error':
            # A quota response stays non-final so the job remains eligible after the allocation resets.
            rec['terminal'] = not quota and (not retry or attempt == first + 2)
        rec['finished_at'] = now()
        write_new(output, rec)
        runner.spent += rec['budget_charge_usd']; runner.records.append(rec)
        if quota:
            raise QuotaReached(job)
        if rec['terminal']:
            return rec
        time.sleep(2 ** (attempt - first + 1) * 2)


def main():
    m = manifest()
    runner = Runner('clef', DIRECTORY / 'clef', m['sha256'], 2.5)
    jobs = eligible(runner, m); done = Counter()
    print(json.dumps({'eligible': len(jobs)}), flush=True)
    try:
        for job, item in jobs:
            done[call(runner, job, item)['status']] += 1
    except QuotaReached as e:
        print(json.dumps({'stopped_on_quota': str(e), 'statuses': done}), flush=True); return
    print(json.dumps({'statuses': done, 'remaining': len(eligible(runner, m)), 'spent_usd': round(runner.spent, 4)}), flush=True)


if __name__ == '__main__':
    main()
