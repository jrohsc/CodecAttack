import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from ensemble_attack import persistent_worker_command


def test_explicit_worker_uses_exact_interpreter_without_conda(monkeypatch):
    monkeypatch.setenv('CODECATTACK_PYTHON_AUDIO_FLAMINGO',sys.executable)
    assert persistent_worker_command('audio_flamingo','/missing','/worker.py') == [sys.executable,'-u','/worker.py','--model','audio_flamingo']


def test_missing_explicit_interpreter_fails_instead_of_using_wrong_env(monkeypatch):
    monkeypatch.setenv('CODECATTACK_PYTHON_AUDIO_FLAMINGO','/missing/python')
    with pytest.raises(ValueError):persistent_worker_command('audio_flamingo','/base','/worker.py')
