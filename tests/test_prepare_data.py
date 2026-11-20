import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from conftest import FakeEmbedder, FakeTokenizer

SCRIPT = Path(__file__).resolve().parent.parent / "experiments" / "prepare_data.py"


class Clock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now


class SlowEmbedder(FakeEmbedder):
    """Every embedding call takes 5 s; chunking (the tokenizer) takes far longer."""

    def __init__(self, clock):
        self.clock = clock

    def embed_passages(self, texts):
        self.clock.now += 5
        return super().embed_passages(texts)

    def embed_queries(self, texts):
        self.clock.now += 5
        return super().embed_queries(texts)

    def tokenizer(self):
        tok = FakeTokenizer()
        count = tok.count

        def slow_count(text):
            self.clock.now += 1000
            return count(text)

        tok.count = slow_count
        return tok


def test_embedding_seconds_covers_only_the_embedding_calls(raw_dir, tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    clock = Clock()
    embedder = SlowEmbedder(clock)
    monkeypatch.setattr(script, "time", SimpleNamespace(perf_counter=clock.perf_counter))
    monkeypatch.setattr(script, "ensure_dataset", lambda data_dir: raw_dir)
    monkeypatch.setattr(script, "detect_device", lambda: "cpu")
    monkeypatch.setattr(script.SentenceTransformerEmbedder, "load", lambda device: embedder)
    monkeypatch.setattr(sys, "argv", ["prepare_data.py", "--data-dir", str(tmp_path / "out")])

    script.main()

    assert "embedding_seconds=10.0" in capsys.readouterr().out
