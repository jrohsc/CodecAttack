from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rebuttal.run_paper_ensemble import MEMBERS, settings, validate_history


def test_paper_settings_keep_members_and_eligible_checkpoint():
    s = settings()
    assert s['epsilon'] == 1 and s['steps'] == 600 and s['perceptual_weight'] == .1
    assert s['members'] == ['qwen25_omni', 'audio_flamingo', 'phi4']
    assert max(s['warmup'], int(s['steps'] * s['channel_warmup_ratio'])) == 180


def test_warmup_minimum_cannot_select_checkpoint_and_missing_member_fails():
    s = settings(True)
    h = dict(loss=[.001, 3., 2., 4.], per_model_loss={m: [1.] * (4 if i == 0 else 3) for i,m in enumerate(MEMBERS)})
    validate_history(h, s, dict(step=3, loss=2.))
    with pytest.raises(RuntimeError, match='not eligible'):
        validate_history(h, s, dict(step=1, loss=.001))
    with pytest.raises(RuntimeError, match='minimum'):
        validate_history(h, s, dict(step=2, loss=3.))
    h['per_model_loss']['phi4'] = []
    with pytest.raises(RuntimeError, match='phi4'):
        validate_history(h, s, dict(step=3, loss=2.))
