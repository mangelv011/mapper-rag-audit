import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from mapper_rag_audit.download import ensure_dataset


def make_zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def fetcher(payload: bytes, calls: list):
    def fetch(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_bytes(payload)

    return fetch


def test_download_verifies_extracts_and_returns_the_dataset_folder(tmp_path):
    payload = make_zip({"scifact/corpus.jsonl": b"{}\n"})
    sha = hashlib.sha256(payload).hexdigest()
    raw = ensure_dataset(tmp_path, "http://x/s.zip", sha, fetcher(payload, []))
    assert raw == tmp_path / "raw" / "scifact"
    assert (raw / "corpus.jsonl").read_bytes() == b"{}\n"


def test_second_run_does_not_download_again(tmp_path):
    payload = make_zip({"scifact/corpus.jsonl": b"{}\n"})
    sha = hashlib.sha256(payload).hexdigest()
    calls: list = []
    ensure_dataset(tmp_path, "http://x/s.zip", sha, fetcher(payload, calls))
    ensure_dataset(tmp_path, "http://x/s.zip", sha, fetcher(payload, calls))
    assert len(calls) == 1


def test_hash_mismatch_is_refused_and_nothing_is_extracted(tmp_path):
    payload = make_zip({"scifact/corpus.jsonl": b"{}\n"})
    with pytest.raises(ValueError, match="SHA-256"):
        ensure_dataset(tmp_path, "http://x/s.zip", "0" * 64, fetcher(payload, []))
    assert not (tmp_path / "raw" / "scifact").exists()
    assert not (tmp_path / "raw" / "scifact.zip").exists()
    assert not (tmp_path / "raw" / "scifact.part").exists()


def test_cached_zip_with_wrong_hash_is_replaced_by_a_fresh_download(tmp_path):
    payload = make_zip({"scifact/corpus.jsonl": b"{}\n"})
    sha = hashlib.sha256(payload).hexdigest()
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "scifact.zip").write_bytes(b"corrupt")
    calls: list = []
    ensure_dataset(tmp_path, "http://x/s.zip", sha, fetcher(payload, calls))
    assert len(calls) == 1


@pytest.mark.parametrize("evil", ["../evil.txt", "scifact/../../evil.txt", "/abs/evil.txt"])
def test_members_escaping_the_target_folder_are_refused(tmp_path, evil):
    payload = make_zip({"scifact/corpus.jsonl": b"{}\n", evil: b"pwned"})
    sha = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="outside"):
        ensure_dataset(tmp_path, "http://x/s.zip", sha, fetcher(payload, []))
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path.parent / "evil.txt").exists()
    assert not (tmp_path / "raw" / "scifact" / "corpus.jsonl").exists()


def test_failed_fetch_leaves_no_partial_file(tmp_path):
    def failing_fetch(url: str, dest: Path) -> None:
        dest.write_bytes(b"half")
        raise TimeoutError("stalled")

    with pytest.raises(TimeoutError):
        ensure_dataset(tmp_path, "http://x/s.zip", "0" * 64, failing_fetch)
    assert not (tmp_path / "raw" / "scifact.part").exists()
    assert not (tmp_path / "raw" / "scifact.zip").exists()


def test_default_fetch_streams_the_response_to_the_file_with_a_timeout(tmp_path, monkeypatch):
    from mapper_rag_audit import download

    seen = {}

    def fake_urlopen(url, timeout=None):
        seen.update(url=url, timeout=timeout)
        return io.BytesIO(b"payload")

    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "out.part"
    download._fetch_url("http://x/s.zip", dest)
    assert dest.read_bytes() == b"payload"
    assert seen["url"] == "http://x/s.zip"
    assert seen["timeout"] is not None and seen["timeout"] > 0
