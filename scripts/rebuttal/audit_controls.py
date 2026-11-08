"""Independently score released control outputs and trace the latent reference."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from audit_release import ROOT, hit


def main():
    source = ROOT / 'data/eval_rollups/15_reviewer_controls/s1_baseline_full40_seed1/evaluation_records.jsonl'
    records = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    out = ROOT / 'results/rebuttal/audit'
    out.mkdir(parents=True, exist_ok=True)
    arms = defaultdict(list)
    mismatches = []
    for r in records:
        arms[r['method']].append(r)
        for c,v in r['outcomes'].items():
            actual = hit(r['target'], v['output'])
            if actual != v.get('match'):
                mismatches.append([r['example_id'],r['method'],c])
    reference = {r['example_id']:r for r in arms['latent_codecattack']}
    summary = {}
    for method, rows in arms.items():
        ds = [r['snr_db']-reference[r['example_id']]['snr_db'] for r in rows]
        summary[method] = dict(n=len(rows), unique_ids=len({r['example_id'] for r in rows}),
            same_cohort={r['example_id'] for r in rows}==set(reference),
            snr_within_1db=sum(abs(d)<=1 for d in ds), max_abs_snr_difference=max(map(abs,ds)),
            channels={c:dict(hits=sum(hit(r['target'],r['outcomes'][c]['output']) for r in rows), n=len(rows))
                      for c in rows[0]['outcomes']})
    provenance = []
    base = ROOT / 'data/eval_rollups/06_main_multiseed/01_finance_voice_agent/qwen2_audio'
    for bundle in ['eps_1.0','eps_1.0_seed1','eps_1.0_seed7','eps_1.0_seed42']:
        summaries = {Path(json.loads(p.read_text())['wav_path']).name:(p,json.loads(p.read_text()))
                     for p in (base/bundle/'audio').glob('*_summary.json')}
        pairs = []
        for r in reference.values():
            found = summaries.get(r['carrier'])
            if not found:
                continue
            p,d = found
            cp = p.with_name(p.name.replace('_summary.json','_codecs.json'))
            codecs = json.loads(cp.read_text()) if cp.exists() else {}
            output = codecs.get('aac_robustness',{}).get('64',{}).get('output')
            pairs.append(dict(example_id=r['example_id'], target_equal=r['target']==d['target_text'],
                clean_equal=r['outcomes']['clean']['output']==d['final_output'],
                aac_equal=r['outcomes']['aac_64']['output']==output,
                aac_observed=isinstance(output,str), aac_hit=hit(d['target_text'],output or '')))
        provenance.append(dict(bundle=bundle, matched_carriers=len(pairs),
            matching_targets=sum(p['target_equal'] for p in pairs),
            identical_clean_outputs=sum(p['clean_equal'] for p in pairs),
            identical_aac_outputs=sum(p['aac_equal'] for p in pairs),
            aac_observed=sum(p['aac_observed'] for p in pairs), aac_hits=sum(p['aac_hit'] for p in pairs)))
    report = dict(source=str(source.relative_to(ROOT)), arms=summary, stored_match_disagreements=mismatches,
                  latent_reference_paths=sorted({r['adv_path'].split('/audio/')[0] for r in reference.values()}),
                  main_bundle_comparison_on_same_40_carriers=provenance,
                  limitation='Released SNR fields are checked arithmetically, not remeasured: adversarial WAVs are absent.')
    (out/'controls_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
