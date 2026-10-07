"""Prelocked English MedQA replication and Korean Luna comparator extension."""
from __future__ import annotations
import argparse, concurrent.futures, importlib.metadata, json, platform, random, threading, time
from datetime import datetime, timezone, timedelta
import httpx
import numpy as np
import pyarrow.parquet as pq
from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient
from .common import ROOT, canonical, digest, filehash, now, read, write_new
from .runner import credential as jev_key, validated, run_lock
from .luna import credential as openai_key
from .study import INSTRUCTION as KOREAN_INSTRUCTION
DIRECTORY = ROOT / 'runs/medical-extension-v1'
OUTPUT = ROOT / 'docs/medical-extension-v1'
SEED = 2026091704
MODELS = {'jev':'jev-1.13.0','sol':'gpt-5.6-sol','luna':'gpt-5.6-luna'}
PRICES = {'jev':(.042,0),'gpt-5.6-sol':(5,20),'gpt-5.6-luna':(.25,1.2)}
INSTRUCTION = 'Select the correct answer to `exam.question`. Following the conditions requested in the question, choose the single most appropriate of the four options.'
HF_REV = '17af9355ef89fba60de966eabaeba797c695f86e'
BB_REV = '484a6c066fe8e75c83edea0c88b5169316714fcd'

def api_request(role,item,selected=None):
    state={'exam':{'question':item['question']}}; options=item['options']
    instruction=INSTRUCTION if item['cohort']=='medqa' else KOREAN_INSTRUCTION
    if role=='jev':
        return {'model':MODELS[role],'state':state,'questions':{'answer':{'type':'choice','instructions':instruction,'criteria':options}}}
    return {'model':MODELS[role],'reasoning':{'effort':'none'},'store':False,'service_tier':'default','max_output_tokens':128,
      'input':[{'role':'user','content':instruction+'\n\n'+canonical(state)+'\n\n'+canonical(options)}],
      'text':{'format':{'type':'json_schema','name':'comparator','strict':True,'schema':{'type':'object','properties':{'answer':{'type':'string','enum':list(options)}},'required':['answer'],'additionalProperties':False}}}}

def output_value(raw,role,item):
    if raw.get('status')!='completed': raise ValueError('Response not completed')
    texts=[c['text'] for o in raw.get('output',[]) if o.get('type')=='message' for c in o.get('content',[]) if c.get('type')=='output_text']
    if len(texts)!=1: raise ValueError('Expected one output')
    v=json.loads(texts[0])
    if set(v)!={'answer'} or v['answer'] not in item['options']: raise ValueError('Invalid answer')
    return {'prediction':v['answer'],'correct':v['answer']==item['gold']}

def all_records():
    return [read(p) for p in sorted((DIRECTORY/'attempts').glob('*.json'))]

def prepare():
    if (DIRECTORY/'manifest.json').exists(): return manifest()
    items=[]; sources=[]
    for split in ['dev','test']:
        hf=DIRECTORY/'sources'/f'{split}.json'; bb=DIRECTORY/'sources'/f'bigbio-{split}.parquet'
        a=[json.loads(s) for s in hf.read_text(encoding='utf-8').splitlines()]; b=pq.read_table(bb).to_pylist()
        if len(a)!=len(b) or len(a)!={'dev':1272,'test':1273}[split]: raise ValueError('Unexpected split counts')
        for x,y in zip(a,b):
            options={k:x[f'ending{n}'] for n,k in enumerate('ABCD')}; gold='ABCD'[x['label']]
            if x['sent2'] or x['sent1']!=y['question'] or options!={o['key']:o['value'] for o in y['options']} or gold!=y['answer_idx'] or options[gold]!=y['answer']: raise ValueError('Mirrors disagree')
            if not x['sent1'].strip() or not all(v.strip() for v in options.values()): raise ValueError('Blank input')
            items.append({'id':'medqa-'+x['id'],'cohort':'medqa','split':split,'question':x['sent1'],'options':options,'gold':gold,'exam_step':y['meta_info']})
        for f,url in [(hf,f'https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf/resolve/{HF_REV}/{split}.json'),(bb,f'https://huggingface.co/datasets/bigbio/med_qa/resolve/{BB_REV}/med_qa_en_4options_source/{"validation" if split=="dev" else split}-00000-of-00001.parquet')]:
            sources.append({'local':f.relative_to(ROOT).as_posix(),'url':url,'sha256':filehash(f)})
    km=read(ROOT/'runs/kormed-study-v1/manifest.json')
    items.extend({**i,'cohort':'kormed'} for i in km['items'] if i['split']=='test')
    if len({i['id'] for i in items})!=len(items): raise ValueError('Duplicate IDs')
    preserved={p.relative_to(ROOT).as_posix():filehash(p) for base in ['docs/kormed-study-v1','runs/kormed-study-v1','results'] for p in (ROOT/base).rglob('*') if p.is_file()}
    previous=sum(r['budget_charge_usd'] for r in [read(p) for p in (ROOT/'runs/kormed-study-v1/attempts').glob('*.json')])
    m={'experiment':'medical-extension-v1','created_at':now(),'items':items,'sources':sources,'models':MODELS,'instruction':INSTRUCTION,'korean_instruction':KOREAN_INSTRUCTION,'seed':SEED,
       'budget_usd':20,'total_authorized_usd':100,'prior_conservative_usd':previous,'prices':PRICES,'preserved_files':preserved,
       'scope':'English replication with three models; Korean Luna extension; no additional annotation or retrieval',
       'threshold_policy':'English 20th percentile of all 1272 Jev dev probabilities before test inference; additionally transfer frozen Korean threshold',
       'korean_threshold':read(ROOT/'runs/kormed-study-v1/threshold.json'),
       'repeat_ids':random.Random(SEED).sample([i['id'] for i in items if i['cohort']=='medqa' and i['split']=='dev'],30),
       'deadline_utc':(datetime.now(timezone.utc)+timedelta(hours=3)).isoformat(),
       'primary':'Within each benchmark accuracy and Jev error AUROC; English prospective replication of confidence routing',
       'bootstrap':{'replicates':4000,'unit':'question within benchmark','seed':SEED},
       'limitations':['Public benchmark training exposure unknown','Different languages, exams and option counts confounded','Korean outcomes seen before extension, English outcomes not seen','Luna added after Korean Jev/Sol results; separate collection time'],
       'code_sha256':filehash(__file__)}
    m['sha256']=digest(m); write_new(DIRECTORY/'manifest.json',m);OUTPUT.mkdir(parents=True,exist_ok=True)
    write_new(OUTPUT/'protocol-lock.json',{k:v for k,v in m.items() if k not in ['items','preserved_files']})
    return m

def manifest():
    m=read(DIRECTORY/'manifest.json');h=m.pop('sha256')
    if digest(m)!=h or m['models']!=MODELS or m['instruction']!=INSTRUCTION or m['korean_instruction']!=KOREAN_INSTRUCTION: raise ValueError('Frozen configuration changed')
    m['sha256']=h;return m

def freeze_threshold():
    m=manifest();p=DIRECTORY/'threshold.json'
    if p.exists(): return read(p)
    rr=all_records();good={r['item_id']:r for r in rr if r['role']=='jev' and r['split']=='dev' and r['repeat']==0 and r['status']=='success'}
    dev=[i for i in m['items'] if i['cohort']=='medqa' and i['split']=='dev']
    if len(good)!=len(dev): raise ValueError('Incomplete development probabilities')
    if any(r['split']=='test' and r['item_id'].startswith('medqa-') for r in rr): raise ValueError('Test already started')
    ps=[good[i['id']]['value']['probabilities'][good[i['id']]['value']['prediction']] for i in dev]
    t={'created_at':now(),'manifest_hash':m['sha256'],'n_dev':len(dev),'threshold':float(np.quantile(ps,.2)),'quantile':.2,'method':'numpy linear','rule':'p <= threshold','development_response_hashes':{i['id']:digest(good[i['id']]) for i in dev}}
    write_new(p,t);write_new(OUTPUT/'threshold.json',t);return t

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
        # Older attempts missed the SDK's `status` field and classified a 5xx as
        # terminal. Preserve that record and apply the already planned retry limit.
        if prior and any(r['terminal'] for r in prior):
            recoverable_sdk_5xx = (prior[-1]['status'] == 'error' and
                prior[-1].get('error_type') == 'TypeSafeInternalServerError' and prior[-1]['attempt'] < 3)
            if not recoverable_sdk_5xx: return prior[-1]
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
                status = getattr(getattr(e, 'response', None), 'status_code', None) or getattr(e, 'status_code', None) or getattr(e, 'status', None)
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


def run_phase(runner,phase):
    m=runner.m;items=m['items'];jobs=[]
    if phase=='dev': jobs=[('jev',i,0,None) for i in items if i['cohort']=='medqa' and i['split']=='dev']
    elif phase=='test':
        t=read(DIRECTORY/'threshold.json')
        if t['manifest_hash']!=m['sha256'] or t['n_dev']!=1272: raise ValueError('Threshold not frozen')
        for i in items:
            if i['cohort']=='medqa' and i['split']=='test':
                roles=['jev','sol','luna'];random.Random(SEED+int(digest(i['id'])[:8],16)).shuffle(roles)
                jobs.extend((r,i,0,None) for r in roles)
    elif phase=='korean': jobs=[('luna',i,0,None) for i in items if i['cohort']=='kormed']
    elif phase=='repeat': jobs=[('jev',i,r,None) for i in items if i['id'] in m['repeat_ids'] for r in [1,2]]
    else: raise ValueError('Unknown phase')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(runner.evaluate,*j) for j in jobs]
        for f in concurrent.futures.as_completed(fs): f.result()
    print(json.dumps({'phase':phase,'jobs':len(jobs),'upper_cost_usd':runner.spent}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','dev','test','korean','repeat','threshold','all','status']);a=p.parse_args()
    if a.command=='prepare':
        m=prepare();print(json.dumps({'n':len(m['items']),'sha256':m['sha256']}));return
    if a.command=='threshold': print(json.dumps(freeze_threshold())[:250]);return
    m=manifest()
    if a.command=='status':
        from collections import Counter
        rr=all_records();print(json.dumps({'counts':dict(Counter(r['role']+'/'+r['split']+'/'+r['status'] for r in rr)),'upper_cost_usd':sum(r['budget_charge_usd'] for r in rr)},indent=2));return
    with run_lock(DIRECTORY):
        write_new(DIRECTORY/'invocations'/(now().replace(':','-')+'.json'),{'started_at':now(),'command':a.command,'code_sha256':filehash(__file__),'manifest_hash':m['sha256'],'python':platform.python_version(),'sdk':importlib.metadata.version('typesafe-sdk'),'concurrency':4})
        runner=StudyRunner(m)
        if a.command=='all':
            run_phase(runner,'dev');freeze_threshold();run_phase(runner,'test');run_phase(runner,'korean');run_phase(runner,'repeat')
        else: run_phase(runner,a.command)

if __name__=='__main__': main()
