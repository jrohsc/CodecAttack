"""Check the released experiment resolves its actual data and prevents split leakage."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from reviewer_controls.run_attack import load_config, partition_pairs


def test_carrier_splits_resolve_and_do_not_leak_recordings():
    cfg = load_config(ROOT / 'scripts/reviewer_controls/config.json')
    for scenario, counts in [('s1', (10, 40)), ('s3a', (10, 30))]:
        dev = partition_pairs(cfg, scenario, 'development', False)
        evaluation = partition_pairs(cfg, scenario, 'evaluation', False)
        assert (len(dev),len(evaluation)) == counts
        assert not ({p['carrier'] for _,p in dev} & {p['carrier'] for _,p in evaluation})
        for _,p in dev + evaluation:
            assert (ROOT / cfg['scenarios'][scenario]['carrier_root'] / p['carrier']).is_file()


def test_primary_metric_rejects_empty_target_and_retains_punctuation():
    sys.path.insert(0, str(ROOT / 'scripts/rebuttal'))
    from audit_release import hit
    assert not hit('', 'arbitrary output')
    assert hit('Hello, world.', 'prefix HELLO,\nWORLD. suffix')
    assert not hit('Hello, world.', 'hello world')


def test_summary_without_clean_does_not_invent_survival():
    from reviewer_controls.common import aggregate_rows
    row=dict(method='m',snr_db=5,post_above_4khz_fraction=0,
             outcomes={'opus64':{'match':True}})
    channel=aggregate_rows([row])['m']['channels']['opus64']
    assert channel['asr']==1
    assert channel['conditional_survival'] is None
    assert channel['compression_induced_new_successes'] is None
    assert channel['clean_paired_denominator']==0


def test_summary_partial_clean_uses_only_paired_rows():
    from reviewer_controls.common import aggregate_rows
    base=dict(method='m',snr_db=5,post_above_4khz_fraction=0)
    rows=[dict(base,outcomes={'opus64':{'match':True}}),
          dict(base,outcomes={'clean':{'match':False},'opus64':{'match':True}})]
    channel=aggregate_rows(rows)['m']['channels']['opus64']
    assert channel['asr_denominator']==2
    assert channel['clean_paired_denominator']==1
    assert channel['compression_induced_new_successes']==1
