#!/usr/bin/env python3
"""Matched-backend Perez/Dolly prompt-condition control. No paid API calls."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from probing.utils import common

MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
RUN = 'llama31_dataset_control_v1'
LAYERS = list(range(0, 32, 4))
SEED = 20260916


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def normalize(text):
    return ' '.join(text.lower().split())


def save_new(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'refusing changed manifest: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(run):
    manifest = run / 'experiment.json'
    if manifest.exists():
        print(f'Prepared experiment already exists: {manifest}')
        return
    run.mkdir(parents=True, exist_ok=True)
    def fetch(url):
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()
    repo = 'databricks/databricks-dolly-15k'
    revision = json.loads(fetch(f'https://huggingface.co/api/datasets/{repo}'))['sha']
    raw = fetch(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/databricks-dolly-15k.jsonl')
    (run / 'dolly_source.jsonl').write_bytes(raw)
    old = common.read_jsonl(common.DATA_DIR / 'perez_user_prompts_5k.jsonl')
    old_norm = {normalize(r['user_prompt']) for r in old}
    old_holdout = set(json.loads((common.RESULTS_DIR / 'llama31_5k_subset/prompt_pool_split.json').read_text())['heldout_pool'])
    groups = defaultdict(list)
    seen = set()
    for i, row in enumerate([json.loads(line) for line in raw.splitlines()]):
        text = row['instruction'].strip()
        if row['context'].strip():
            text += '\n\nContext:\n' + row['context'].strip()
        key = normalize(text)
        if len(text) > 2500 or len(text) < 20 or key in old_norm or key in seen:
            continue
        seen.add(key)
        groups[row['category']].append({'prompt_id': f'dolly__{i:05d}', 'dataset': 'dolly',
            'source': row['category'], 'user_prompt': text, 'original_row': i})
    if len(groups) != 8:
        raise ValueError('expected eight Dolly categories')
    rng = random.Random(SEED)
    prompts = []
    for category in sorted(groups):
        candidates = groups[category]
        rng.shuffle(candidates)
        if len(candidates) < 25:
            raise ValueError(f'not enough candidates: {category}')
        for i, row in enumerate(candidates[:25]):
            prompts.append({**row, 'split': 'train' if i < 15 else 'report'})
    sources = defaultdict(list)
    for row in old:
        if row['prompt_id'] in old_holdout:
            sources[row['source']].append(row)
    # Round-robin across original source domains. These 200 questions were not
    # used to fit the legacy 2,000-prompt probe suite.
    for rows in sources.values():
        rng.shuffle(rows)
    chosen = []
    while len(chosen) < 200:
        for source in sorted(sources):
            if sources[source] and len(chosen) < 200:
                row = sources[source].pop()
                chosen.append({'prompt_id': row['prompt_id'], 'dataset': 'perez',
                               'source': source, 'user_prompt': row['user_prompt']})
    rng.shuffle(chosen)
    prompts += [{**r, 'split': 'train' if i < 120 else 'report'} for i, r in enumerate(chosen)]
    pairs = common.load_prompt_pairs()
    # Use exactly the instruction strings that generated the original data.
    for pair in pairs:
        source = common.RESULTS_DIR / 'llama31_5k_subset/generations' / f"{pair['slug']}.jsonl"
        found = {}
        with source.open() as handle:
            for line in handle:
                r = json.loads(line)
                found[r['polarity']] = r['system_prompt']
                if len(found) == 2:
                    break
        for polarity in common.POLARITIES:
            if pair[polarity] != found[polarity]:
                raise ValueError(f"current prompt differs from legacy generation: {pair['slug']} / {polarity}")
    exp = {'model': MODEL, 'model_revision': '0e9e39f249a16976918f6564b8830bc894c89659',
           'seed': SEED, 'prompts': prompts, 'pairs': pairs, 'layers': LAYERS,
           'dolly': {'repository': repo, 'revision': revision,
                     'sha256': hashlib.sha256(raw).hexdigest(), 'license': 'CC-BY-SA-3.0',
                     'url': f'https://huggingface.co/datasets/{repo}'},
           'generation': {'temperature': .6, 'top_p': .9, 'max_new_tokens': 1024, 'batch_size': 24},
           'analysis': {'C': 1.0, 'max_iter': 2000, 'bootstrap': 200,
                        'train_per_dataset': 120, 'report_per_dataset': 80},
           'n_generation_target': 16400, 'exact_prompt_overlap_A_B': 0,
           'note': 'Pilot. Original wording fixed. Fresh matched-backend A/B; legacy probe transfer is secondary. Neutral-context clustering uses report questions only. Labels identify instructions, not judged behavior.'}
    save_new(manifest, exp)
    common.write_jsonl(run / 'user_prompts.jsonl', prompts)
    print(json.dumps({'prepared': str(run), 'counts': dict(Counter((r['dataset'] + '/' + r['split']) for r in prompts)),
                      'generations': exp['n_generation_target'], 'dolly_revision': revision}), flush=True)


def work_items(exp, slug):
    pair = next((p for p in exp['pairs'] if p['slug'] == slug), None)
    items = []
    for prompt in exp['prompts']:
        for polarity in common.POLARITIES if pair else ('neutral',):
            items.append({**prompt, 'slug': slug, 'polarity': polarity,
                          'label': common.POLARITY_LABEL.get(polarity),
                          'system_prompt': pair[polarity] if pair else '',
                          'example_id': f"{slug}__{polarity}__{prompt['dataset']}__{prompt['prompt_id']}"})
    return items


def gpu(run, model_path, smoke=False):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from probing.probe import get_activations as ga
    exp = json.loads((run / 'experiment.json').read_text())
    if smoke:
        exp['prompts'] = [next(p for p in exp['prompts'] if p['dataset'] == d) for d in ('perez', 'dolly')]
        exp['pairs'] = exp['pairs'][:1]
        run = run / 'smoke'
    run.mkdir(parents=True, exist_ok=True)
    gen_dir, act_dir = run / 'generations', run / 'activations'
    gen_dir.mkdir(exist_ok=True); act_dir.mkdir(exist_ok=True)
    save_new(run / 'runtime.json', {'experiment_hash': digest(exp), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'model_path': str(model_path), 'smoke': smoke})
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                device_map='cuda', attn_implementation='sdpa')
    if model.config.max_position_embeddings != 131072 or model.config.hidden_size != 4096:
        raise ValueError('wrong model architecture')
    model.eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    start = time.monotonic()
    for slug in [p['slug'] for p in exp['pairs']] + ['neutral']:
        items = work_items(exp, slug)
        path = gen_dir / f'{slug}.jsonl'
        existing = common.read_jsonl(path) if path.exists() else []
        done = {r['example_id'] for r in existing}
        if len(done) != len(existing) or not done <= {r['example_id'] for r in items}:
            raise ValueError('invalid resume rows')
        # Batches and seeds are fixed before resume filtering; original and new
        # datasets are mixed in every generation stage under identical settings.
        random.Random(SEED).shuffle(items)
        batch_size = exp['generation']['batch_size']
        tok.padding_side = 'left'
        with path.open('a') as handle, torch.inference_mode():
            for offset in range(0, len(items), batch_size):
                batch = items[offset:offset + batch_size]
                known = [r['example_id'] in done for r in batch]
                if all(known):
                    continue
                if any(known):
                    raise ValueError('partial batch checkpoint; inspect before regenerating')
                seed = int(hashlib.sha256(f'{SEED}:{slug}:{offset}'.encode()).hexdigest()[:8], 16)
                torch.manual_seed(seed)
                texts = []
                for row in batch:
                    messages = ([{'role': 'system', 'content': row['system_prompt']}] if row['system_prompt'] else [])
                    messages.append({'role': 'user', 'content': row['user_prompt']})
                    texts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                enc = tok(texts, padding=True, add_special_tokens=False, return_tensors='pt').to('cuda')
                if enc.input_ids.shape[1] + exp['generation']['max_new_tokens'] > 4096:
                    raise ValueError('input exceeds planned extraction length')
                result = model.generate(**enc, do_sample=True, temperature=.6, top_p=.9,
                                        max_new_tokens=1024, pad_token_id=tok.pad_token_id)
                for row, sequence in zip(batch, result[:, enc.input_ids.shape[1]:].tolist()):
                    stop = next((i for i, token in enumerate(sequence) if token in eos), None)
                    response = tok.decode(sequence[:stop] if stop is not None else sequence, skip_special_tokens=True)
                    record = {**row, 'response': response, 'finish_reason': 'stop' if stop is not None else 'length',
                              'model': MODEL, 'batch_seed': seed,
                              'degenerate': 'empty' if not response.strip() else ('truncated' if stop is None else None)}
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush(); os.fsync(handle.fileno())
                print(f'{slug}: generated {offset + len(batch)}/{len(items)}; elapsed {time.monotonic()-start:.0f}s', flush=True)
        rows = common.read_jsonl(path)
        cache = act_dir / f'{slug}.npz'
        if cache.exists():
            marker = json.loads((act_dir / f'{slug}_complete.json').read_text())
            if marker['generation_hash'] != digest(rows):
                raise ValueError('activation provenance mismatch')
            continue
        tok.padding_side = 'right'
        prepared, skipped = ga.prepare_records(rows, tok, SimpleNamespace(max_length=4096))
        arrays = ga.extract(model, tok, prepared, LAYERS, 4096, 8)
        index = [{**{k: p['rec'][k] for k in ('example_id','prompt_id','dataset','source','split','slug','polarity','label','finish_reason','degenerate')},
                  'row': i, 'n_response_tokens': p['resp_end']-p['prompt_len'], 'first5_text': p['first5_text']}
                 for i,p in enumerate(prepared)]
        with cache.with_suffix('.npz.partial').open('wb') as handle:
            np.savez_compressed(handle, **{ga.act_key(pos,l): a for (pos,l),a in arrays.items()})
        common.write_jsonl(act_dir / f'{slug}_index.jsonl', index)
        save_new(act_dir / f'{slug}_skips.json', skipped)
        cache.with_suffix('.npz.partial').rename(cache)
        save_new(act_dir / f'{slug}_complete.json', {'generation_hash': digest(rows), 'n_rows': len(index),
            'n_skips': len(skipped), 'sha256': hashlib.sha256(cache.read_bytes()).hexdigest()})
        print(f'COMPLETE {slug}: {len(index)} activation rows', flush=True)
    save_new(run / 'gpu_complete.json', {'experiment_hash': digest(exp), 'completed_cells': [p['slug'] for p in exp['pairs']] + ['neutral']})


def analyze(run):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score
    from probing.probe.train_probes import fit_probe, score, safe_auc
    from probing.analyze_probes.analyze_probes import apply_probe, load_probes
    from probing.analyze_probes.cluster_syconbench_stability import bootstrap, correlation, partition
    exp = json.loads((run / 'experiment.json').read_text())
    names = [p['slug'] for p in exp['pairs']]
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    fitted = {'perez': {}, 'dolly': {}}
    legacy = {n: load_probes(common.RESULTS_DIR / 'llama31_5k_subset', n) for n in names}
    indices = {n: common.read_jsonl(run / 'activations' / f'{n}_index.jsonl') for n in names + ['neutral']}
    transfers, clusters, prompted_clusters = {}, {}, {}
    # Shared question split applies across all prompt conditions. C is fixed,
    # not selected using report labels. Truncated but nonempty responses stay in
    # the primary analysis; complete-pair sensitivity is recorded separately.
    for layer in LAYERS:
        for position in common.POSITIONS:
            key = f'{position}_L{layer:02d}'
            print('Analysis',key,flush=True)
            test = {}
            for name in names:
                with np.load(run / 'activations' / f'{name}.npz') as z:
                    X = z[key].astype(np.float32)
                ix = indices[name]
                # Keep paired questions only when both polarities survived spans.
                counts = Counter((r['dataset'],r['prompt_id']) for r in ix)
                paired = np.array([counts[r['dataset'],r['prompt_id']] == 2 for r in ix])
                y = np.array([r['label'] for r in ix])
                for arm in ('perez','dolly'):
                    train = np.array([r['dataset']==arm and r['split']=='train' for r in ix]) & paired
                    report = np.array([r['dataset']==arm and r['split']=='report' for r in ix]) & paired
                    if train.sum()<100 or report.sum()<40:
                        raise ValueError(f'insufficient paired rows {name}/{arm}')
                    scaler, clf = fit_probe(X[train], y[train], SEED, 1.0, 2000)
                    fitted[arm][name] = (scaler,clf)
                    if int(clf.n_iter_[0]) >= 2000:
                        raise ValueError(f'probe did not converge: {name}/{arm}/{key}')
                    test[name,arm] = (X[report],y[report], [r for r,k in zip(ix,report) if k])
                    save_dir = run / 'fitted_probes' / arm / name
                    save_dir.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(save_dir / f'{key}.npz', coef=clf.coef_[0], intercept=clf.intercept_[0],mean=scaler.mean_,scale=scaler.scale_)
            for train_arm in ('perez','dolly','legacy'):
                for test_arm in ('perez','dolly'):
                    values, complete_values, score_columns = [], [], []
                    for name in names:
                        row, cleanrow, blocks = [], [], []
                        for source in names:
                            X,y,ix = test[source,test_arm]
                            s = apply_probe(legacy[name][key],X) if train_arm=='legacy' else score(*fitted[train_arm][name],X)
                            bad = {r['prompt_id'] for r in ix if r['finish_reason']!='stop'}
                            clean = np.array([r['prompt_id'] not in bad for r in ix])
                            row.append(safe_auc(y,s));cleanrow.append(safe_auc(y[clean],s[clean]))
                            blocks.append(s)
                        values.append(row);complete_values.append(cleanrow)
                        score_columns.append(np.concatenate(blocks))
                    transfers[f'{train_arm}_to_{test_arm}_{key}']={'train_dataset':train_arm,'test_dataset':test_arm,
                        'position':position,'layer':layer,'cells':names,'auc':values,'complete_pairs_auc':complete_values,
                        'n_report_by_source':{n:len(test[n,test_arm][1]) for n in names}}
                    if position == 'response':
                        rows = [r for source in names for r in test[source,test_arm][2]]
                        S = np.column_stack(score_columns)
                        ids = np.array([r['prompt_id'] for r in rows])
                        lengths = np.array([r['n_response_tokens'] for r in rows])
                        result = bootstrap(S,ids,np.ones(len(ids),dtype=int),lengths,False,200,SEED)
                        # Descriptive sensitivity removes between-condition mean
                        # score differences, not the system prompt itself.
                        centered = S.copy()
                        conditions = np.array([r['slug']+'/'+r['polarity'] for r in rows])
                        for condition in np.unique(conditions):
                            mask = conditions == condition
                            centered[mask] -= centered[mask].mean(axis=0)
                        residual_partition = partition(correlation(centered))
                        prompted_clusters[f'{train_arm}_on_{test_arm}_{key}'] = {
                            'train_dataset':train_arm,'basis':test_arm,'layer':layer,'cells':names,
                            'n_rows':len(rows),'n_questions':len(np.unique(ids)),**result,
                            'condition_mean_removed':residual_partition}
            # Neutral prompt context: no inducing system-prompt mixture can
            # mechanically dominate these correlation matrices.
            if position == 'response':
                with np.load(run / 'activations/neutral.npz') as z: X=z[key].astype(np.float32)
                ix=indices['neutral']
                for test_arm in ('perez','dolly'):
                    mask=np.array([r['dataset']==test_arm and r['split']=='report' for r in ix])
                    ids=np.array([r['prompt_id'] for r,k in zip(ix,mask) if k])
                    lengths=np.array([r['n_response_tokens'] for r,k in zip(ix,mask) if k])
                    for train_arm in ('perez','dolly','legacy'):
                        S=np.column_stack([apply_probe(legacy[n][key],X[mask]) if train_arm=='legacy' else score(*fitted[train_arm][n],X[mask]) for n in names])
                        result=bootstrap(S,ids,np.ones(len(ids),dtype=int),lengths,False,200,SEED)
                        clusters[f'{train_arm}_on_{test_arm}_{key}']={'train_dataset':train_arm,'basis':test_arm,'layer':layer,
                            'n_questions':len(ids),'cells':names,**result}
            (out / 'transfer.json').write_text(json.dumps(transfers,indent=2)+'\n')
            (out / 'neutral_clusters.json').write_text(json.dumps(clusters,indent=2)+'\n')
            (out / 'prompted_clusters.json').write_text(json.dumps(prompted_clusters,indent=2)+'\n')
    lines=['# Dataset-control pilot results','',
        'Fresh matched-backend Perez versus Dolly; 120 training and 80 report questions per dataset, 20 fixed system-prompt pairs. AUROC labels identify generating instructions, not independently judged sycophancy.', '',
        '| Layer / response average | Perez→Perez | Perez→Dolly | Dolly→Perez | Dolly→Dolly |',
        '|---|---:|---:|---:|---:|']
    for layer in LAYERS:
        row=[]
        for a,b in [('perez','perez'),('perez','dolly'),('dolly','perez'),('dolly','dolly')]:
            m=np.array(transfers[f'{a}_to_{b}_response_L{layer:02d}']['auc'],dtype=float)
            row.append(f'{np.nanmedian(m[~np.eye(len(names),dtype=bool)]):.3f}')
        lines.append(f'| L{layer} | '+' | '.join(row)+' |')
    lines+=['','Entries above are medians of off-diagonal cell AUROCs, not pooled AUROC and not concept counts. Inspect full matrices and control polarities before interpretation.','',
            '## Neutral-context clustering','', '| Training probes / layer | Perez neutral best k | Dolly neutral best k | Membership ARI |','|---|---:|---:|---:|']
    for layer in LAYERS:
        for a in ('perez','dolly','legacy'):
            r=clusters[f'{a}_on_perez_response_L{layer:02d}'];s=clusters[f'{a}_on_dolly_response_L{layer:02d}']
            lines.append(f"| {a} L{layer} | {r['best_k']} | {s['best_k']} | {adjusted_rand_score(r['labels'],s['labels']):.3f} |")
    lines+=['','Question-bootstrap stability and membership arrays are in neutral_clusters.json. Matched comparisons across the mixture of inducing prompt conditions, plus condition-mean-removal sensitivity, are in prompted_clusters.json. Both bases use report questions only. Searching k=2–9 always returns clusters; this does not establish that discrete concepts exist.','',
            'System-prompt wording is held fixed, not experimentally removed. A surviving pattern supports dataset robustness, not proof that prompts cause collapse. Paraphrase testing is a distinct follow-up. Original-versus-fresh legacy comparisons additionally change the generation backend and training sample size.']
    (out / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    save_new(run / 'analysis_complete.json',{'experiment_hash':digest(exp),'n_transfer_matrices':len(transfers),'n_neutral_cluster_views':len(clusters),'n_prompted_cluster_views':len(prompted_clusters)})


def watch(run, host, port, identity):
    import fcntl
    lock = (run / 'watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ssh=['ssh','-i',identity,'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-p',port,host]
    scp=['scp','-i',identity,'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-P',port]
    remote=f'/workspace/syconbench_eval/prompt_probes/results/{RUN}'
    exp=json.loads((run/'experiment.json').read_text())
    names=[p['slug'] for p in exp['pairs']]+['neutral']
    def command(cmd):
        for attempt in range(5):
            try:
                return subprocess.run(ssh+[cmd],check=True,text=True,stdout=subprocess.PIPE).stdout.strip()
            except subprocess.CalledProcessError as exc:
                if exc.returncode != 255 or attempt == 4:
                    raise
                print('Transient SSH failure; retrying in 30 seconds.', flush=True)
                time.sleep(30)
    pending=set(names)
    while pending:
        status=command('supervisorctl status dataset_control_v1 || test $? -eq 3')
        print(status,flush=True)
        for name in names:
            if name not in pending:continue
            if command(f'test -f {remote}/activations/{name}_complete.json && echo yes || echo no')!='yes':continue
            for relative in [f'generations/{name}.jsonl',f'activations/{name}.npz',f'activations/{name}_index.jsonl',f'activations/{name}_skips.json',f'activations/{name}_complete.json']:
                local=run/relative;local.parent.mkdir(parents=True,exist_ok=True)
                if local.exists():continue
                temporary=local.with_name(local.name+'.download')
                subprocess.run(scp+[f'{host}:{remote}/{relative}',str(temporary)],check=True)
                temporary.rename(local)
            marker=json.loads((run/f'activations/{name}_complete.json').read_text())
            path=run/f'activations/{name}.npz'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=marker['sha256']:raise ValueError('download checksum mismatch')
            if digest(common.read_jsonl(run/f'generations/{name}.jsonl')) != marker['generation_hash']:
                raise ValueError('generation hash mismatch')
            import numpy as np
            index = common.read_jsonl(run/f'activations/{name}_index.jsonl')
            if len(index) != marker['n_rows']:
                raise ValueError('index length mismatch')
            with np.load(path) as arrays:
                if len(arrays.files) != 24:
                    raise ValueError('missing activation configurations')
                for key in arrays.files:
                    a = arrays[key]
                    if a.shape != (len(index),4096) or not np.isfinite(a).all():
                        raise ValueError(f'invalid activation array: {key}')
            pending.remove(name)
            print('Downloaded and verified',name,flush=True)
        (run/'workflow_status.json').write_text(json.dumps({'stage':'generating_extracting' if pending else 'analyzing','remaining_cells':sorted(pending)},indent=2))
        if pending and not any(state in status for state in ('RUNNING','STARTING')):
            raise RuntimeError(f'GPU job stopped with unfinished cells: {status}')
        if pending:time.sleep(45)
    analyze(run)
    (run/'workflow_status.json').write_text(json.dumps({'stage':'complete','report':str(run/'analysis/RESULTS.md')},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','gpu','analyze','watch'])
    parser.add_argument('--run-name',default=RUN)
    parser.add_argument('--model-path',default='/workspace/syconbench_model')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--host',default='root@ssh4.vast.ai')
    parser.add_argument('--port',default='15883')
    parser.add_argument('--identity',default='/Users/oscar/.ssh/vast_vla')
    args=parser.parse_args();run=common.resolve_run_dir(args.run_name)
    if args.stage=='prepare':prepare(run)
    elif args.stage=='gpu':gpu(run,args.model_path,args.smoke)
    elif args.stage=='analyze':analyze(run)
    else:
        try:
            watch(run,args.host,args.port,args.identity)
        except Exception as exc:
            (run/'workflow_status.json').write_text(json.dumps({'stage':'error','error':str(exc)},indent=2))
            raise


if __name__=='__main__':main()
