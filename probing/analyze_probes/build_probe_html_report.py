#!/usr/bin/env python3
"""Build an offline HTML atlas from saved probe analysis; no fitting or API calls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

POSITIONS = ['response', 'first5', 'last_prompt']
LAYERS = list(range(0, 32, 4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='llama31_5k_subset')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run = root / 'prompt_probes/results' / args.run_name
    out = run / 'analysis/interactive_report/dist'
    out.mkdir(parents=True, exist_ok=True)
    sources = {}

    def read(path):
        content = path.read_bytes()
        sources[str(path.relative_to(run))] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    stability = read(run / 'analysis/syconbench/clustering_stability/results.json')
    names = stability['cells']
    metrics = {n: read(run / 'probes' / n / 'metrics.json') for n in names}
    labels = {n: metrics[n]['rows'][0].get('cell', n) for n in names}
    data = dict(names=names, labels=labels, layers=LAYERS, positions=POSITIONS,
                maps={}, targets={}, id={}, headlines={}, sources=sources)

    def add_map(key, **values):
        assert key not in data['maps']
        data['maps'][key] = values

    for path in sorted((run / 'analysis').glob('score_correlation_*_eval_*.json')):
        old = read(path)
        if 'cluster_sweep_signed' not in old:
            continue
        config, base = path.stem.removeprefix('score_correlation_').split('_eval_')
        position, layer = config.rsplit('_L', 1)
        if position not in POSITIONS:
            continue
        sweep = old['cluster_sweep_signed']
        best = max(sweep, key=lambda k: sweep[k]['silhouette'])
        add_map(path.stem, kind='correlation', base=base, position=position, layer=int(layer),
                title=f'{base} · {position} · L{int(layer)}', names=old['cells'],
                values=old['correlation'], clusters=list(sweep[best]['clusters'].values()),
                k=int(best), silhouette=sweep[best]['silhouette'], n=old['n_samples'],
                source=str(path.relative_to(run)))

    for tag, result in stability['configs'].items():
        setting, config = tag.split('_response_')
        layer = int(config.removeprefix('L'))
        raw = result['raw']
        base = f'syconbench:{setting}'
        add_map(tag, kind='correlation', base=base, position='response', layer=layer,
                title=f'SYCON {setting} · response · L{layer}', names=names,
                values=raw['correlation'], clusters=raw['clusters'], k=raw['best_k'],
                silhouette=raw['silhouette'], n=result['n_turns'], stability=result,
                source='analysis/syconbench/clustering_stability/results.json')
        if layer in (12, 16, 20):
            for mode, field in [('selected', 'coclustering_selected_k'), ('fixed3', 'coclustering_fixed_k3')]:
                clusters = raw['clusters'] if mode == 'selected' else [
                    [n for n, label in zip(names, raw['fixed_k3_labels']) if label == i]
                    for i in sorted(set(raw['fixed_k3_labels']))]
                add_map(f'{tag}_bootstrap_{mode}', kind='consensus', base=base,
                        position='response', layer=layer, title=f'SYCON {setting} · L{layer} · bootstrap {mode}',
                        names=names, values=raw[field], clusters=clusters, mode=mode,
                        k=raw['best_k'] if mode == 'selected' else 3, n=result['n_turns'],
                        source='analysis/syconbench/clustering_stability/results.json')

    for base in ['sypr', 'moral', 'elephant', 'syconbench', 'cross_cell']:
        summary = read(run / f'eval_{base}/summary.json')
        rows = [r for r in summary['rows'] if r['slug'] in names]
        for field in sorted({r['label_field'] for r in rows}):
            key = f'{base}:{field}'
            target = dict(base=base, field=field, note=summary.get('note', ''), values={}, counts={})
            for pos in POSITIONS:
                target['values'][pos] = {}
                for split in ('selection', 'eval'):
                    lookup = {(r['slug'], r['layer']): r for r in rows if r['label_field'] == field
                              and r['position'] == pos and r['split'] == split and r['dataset'] == 'all'}
                    target['values'][pos][split] = [[lookup.get((n, l), {}).get('auc') for l in LAYERS] for n in names]
                    if lookup:
                        sample = next(iter(lookup.values()))
                        target['counts'][split] = {k: sample[k] for k in ('n', 'n_pos', 'n_neg')}
            data['targets'][key] = target
            if base != 'cross_cell':
                for pos in POSITIONS:
                    add_map(f'auc_{base}_{field}_{pos}', kind='auc', base=base, position=pos,
                            title=f'{base} · {field} · {pos} AUROC', names=names, columns=LAYERS,
                            values=target['values'][pos]['eval'], target=key,
                            source=f'eval_{base}/summary.json')
        if base == 'cross_cell':
            for pos in POSITIONS:
                for layer in LAYERS:
                    lookup = {(r['slug'], r['dataset']): r['auc'] for r in rows
                              if r['position'] == pos and r['layer'] == layer and r['split'] == 'eval' and r['dataset'] in names}
                    add_map(f'transfer_{pos}_L{layer:02d}', kind='transfer', base=base, position=pos,
                            layer=layer, title=f'Between system prompts · {pos} · L{layer}', names=names,
                            values=[[lookup.get((a, b)) for b in names] for a in names],
                            source='eval_cross_cell/summary.json')
        else:
            data['headlines'][base] = read(run / f'analysis/matrix_{base}.json')['fields']

    for pos in POSITIONS:
        cube = [[next((r['holdout_auc'] for r in metrics[n]['rows'] if r['position'] == pos and r['layer'] == l), None)
                 for l in LAYERS] for n in names]
        data['id'][pos] = cube
        add_map(f'within_prompt_{pos}', kind='id', base='within_prompt', position=pos,
                title=f'Own system-prompt contrast · {pos}', names=names, columns=LAYERS, values=cube,
                source='probes/*/metrics.json')

    # Compact finite JSON. All data is aggregate; no conversation text or credentials.
    def rounded(obj):
        if isinstance(obj, float):
            return round(obj, 6)
        if isinstance(obj, dict):
            return {k: rounded(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [rounded(v) for v in obj]
        return obj
    assets = Path(__file__).with_name('probe_report_assets')
    for filename in ('report.css', 'report.js'):
        shutil.copyfile(assets / filename, out / filename)
    payload = rounded(data)
    # Selection argmax must use original precision, not rounded display values.
    for key in ('targets', 'id', 'headlines'):
        payload[key] = data[key]
    (out / 'data.js').write_text('window.PROBE_REPORT = ' + json.dumps(payload, allow_nan=False, separators=(',', ':')) + ';\n')
    template = (assets / 'page.html').read_text()
    pages = {'index': 'overview', 'heatmaps': 'gallery', **{k: k for k in data['maps']}}
    for filename, page in pages.items():
        (out / f'{filename}.html').write_text(template.replace('__PAGE__', page))
    (out / 'provenance.json').write_text(json.dumps(sources, indent=2) + '\n')
    (out.parent / '.openai').mkdir(exist_ok=True)
    (out.parent / '.openai/hosting.json').write_text(json.dumps({'static': {'directory': 'dist'}}, indent=2) + '\n')
    print(f'Built {len(pages)} HTML pages in {out}', flush=True)


if __name__ == '__main__':
    main()
