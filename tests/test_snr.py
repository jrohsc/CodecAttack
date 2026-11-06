import numpy as np
import soundfile as sf
import pytest

from audio_agent_bench.snr import snr_db


def _write_wav(path, data, sr):
    sf.write(str(path), data.astype(np.float32), sr, subtype="PCM_16")


def test_identical_audio_has_infinite_snr(tmp_path):
    sr = 24000
    sig = np.sin(np.linspace(0, 10, sr)).astype(np.float32)
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _write_wav(a, sig, sr)
    _write_wav(b, sig, sr)
    assert snr_db(a, b) > 60.0


def test_noise_equal_to_signal_gives_zero_db(tmp_path):
    sr = 24000
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(sr).astype(np.float32) * 0.1
    noise = rng.standard_normal(sr).astype(np.float32) * 0.1
    adv = sig + noise
    car = tmp_path / "car.wav"
    ad = tmp_path / "adv.wav"
    _write_wav(car, sig, sr)
    _write_wav(ad, adv, sr)
    assert abs(snr_db(ad, car)) < 3.0


def test_resamples_when_sample_rates_differ(tmp_path):
    sig_24k = np.sin(np.linspace(0, 10, 24000)).astype(np.float32)
    sig_48k = np.sin(np.linspace(0, 10, 48000)).astype(np.float32)
    car = tmp_path / "car_48k.wav"
    ad = tmp_path / "adv_24k.wav"
    _write_wav(car, sig_48k, 48000)
    _write_wav(ad, sig_24k, 24000)
    assert snr_db(ad, car) > 20.0


def test_stereo_carrier_is_mixed_to_mono(tmp_path):
    sr = 24000
    mono = np.sin(np.linspace(0, 10, sr)).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1)
    car = tmp_path / "car.wav"
    ad = tmp_path / "adv.wav"
    _write_wav(car, stereo, sr)
    _write_wav(ad, mono, sr)
    assert snr_db(ad, car) > 60.0


def test_length_mismatch_truncates_to_min(tmp_path):
    sr = 24000
    sig = np.sin(np.linspace(0, 10, sr)).astype(np.float32)
    car = tmp_path / "car.wav"
    ad = tmp_path / "adv.wav"
    _write_wav(car, sig, sr)
    _write_wav(ad, sig[: sr // 2], sr)
    assert snr_db(ad, car) > 60.0
