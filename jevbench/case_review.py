"""Characterization of the selected Korean persistent errors with GPT-6.1 Sol.

Uses the original review instruction and schema from the Korean study, changing only the model, so the
characterizations behind Supplementary Table 9 come from the current Sol model. One request per case, no retries
beyond transient failures. Earlier characterizations are kept unchanged in docs/kormed-study-v1.
"""
import json
import time

import httpx

from .common import ROOT, canonical, digest, now, read, write_new
from .luna import credential
from .study import REVIEW_INSTRUCTION, REVIEW_PROPERTIES, schema

DIRECTORY = ROOT / 'runs/case-review-v1'
OUTPUT = ROOT / 'docs/case-review-v1'
MODEL = 'gpt-6.1-sol'
# Korean cases in Supplementary Table 9 with the wrong option shared by the decision models.
CASES = {'doctor-2022-1-18': 'A', 'doctor-2023-1-17': 'C', 'doctor-2024-1-74': 'D', 'doctor-2022-1-75': 'A', 'doctor-2024-3-70': 'C'}


def request(item, selected):
    content = REVIEW_INSTRUCTION + '\n\n' + canonical({'question': item['question'], 'options': item['options'],
                                                       'selected_option': selected, 'official_key': item['gold']})
    return {'model': MODEL, 'reasoning': {'effort': 'low'}, 'store': False, 'service_tier': 'default', 'max_output_tokens': 4096,
            'input': [{'role': 'user', 'content': content}],
            'text': {'format': {'type': 'json_schema', 'name': 'review', 'strict': True, 'schema': schema(REVIEW_PROPERTIES)}}}


def main():
    items = {i['id']: i for i in read(ROOT / 'runs/kormed-study-v1/manifest.json')['items']}
    key = credential(); results = []
    for qid, selected in CASES.items():
        item = items[qid]; req = request(item, selected)
        assert selected != item['gold']
        path = DIRECTORY / f'{qid}.json'
        if path.exists():
            results.append(read(path)); continue
        for attempt in range(1, 4):
            r = httpx.post('https://api.openai.com/v1/responses', json=req, timeout=600, headers={'Authorization': 'Bearer ' + key})
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(4 * attempt); continue
            r.raise_for_status(); break
        raw = r.json()
        text = [c['text'] for o in raw.get('output', []) if o.get('type') == 'message' for c in o.get('content', []) if c.get('type') == 'output_text']
        rec = dict(item_id=qid, selected_option=selected, official_key=item['gold'], request_hash=digest(req), requested_at=now(),
                   resolved_model=raw.get('model'), status=raw.get('status'), usage=raw.get('usage'), raw_response=raw,
                   value=json.loads(text[0]) if raw.get('status') == 'completed' and len(text) == 1 else None)
        write_new(path, rec); results.append(rec)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = [dict(item_id=r['item_id'], selected_option=r['selected_option'], official_key=r['official_key'], model=r['resolved_model'], **(r['value'] or {}))
           for r in results]
    (OUTPUT / 'review.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    for r in out:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == '__main__':
    main()
