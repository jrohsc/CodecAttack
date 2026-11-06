from audio_agent_bench.probe import select_probe_pairs


def _pair(carrier, target="t"):
    return {"carrier": carrier, "target": target}


def test_picks_first_per_unique_carrier():
    pairs = [_pair("a.mp3"), _pair("a.mp3", "t2"), _pair("b.mp3"),
             _pair("b.mp3", "t2"), _pair("c.mp3")]
    out = select_probe_pairs(pairs, k=5)
    assert [p["carrier"] for p in out] == ["a.mp3", "b.mp3", "c.mp3"]
    assert out[0]["target"] == "t"
    assert out[1]["target"] == "t"


def test_caps_at_k():
    pairs = [_pair(f"c{i}.mp3") for i in range(10)]
    out = select_probe_pairs(pairs, k=5)
    assert len(out) == 5
    assert [p["carrier"] for p in out] == [f"c{i}.mp3" for i in range(5)]


def test_preserves_input_order():
    pairs = [_pair("b.mp3"), _pair("a.mp3"), _pair("b.mp3", "t2"), _pair("a.mp3", "t2")]
    out = select_probe_pairs(pairs, k=5)
    assert [p["carrier"] for p in out] == ["b.mp3", "a.mp3"]


def test_fewer_unique_than_k_returns_all_unique():
    pairs = [_pair("a.mp3"), _pair("b.mp3"), _pair("a.mp3", "t2")]
    out = select_probe_pairs(pairs, k=5)
    assert [p["carrier"] for p in out] == ["a.mp3", "b.mp3"]


def test_empty_input_returns_empty_list():
    assert select_probe_pairs([], k=5) == []
