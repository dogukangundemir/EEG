import json
import urllib.error

import pytest

from src import download  # noqa: E402


def _entry(path):
    return {"path": path, "type": "blob", "size": 123}


FAKE_TREE = [
    _entry("dataset_description.json"),
    _entry("participants.tsv"),
    _entry("participants.json"),
    _entry("README.md"),
    _entry(".bidsignore"),
    _entry("CHANGES"),
    _entry("sub-001/eeg/sub-001_task-jacobsen_eeg.bdf"),
    _entry("sub-001/eeg/sub-001_task-jacobsen_events.tsv"),
    _entry("sub-002/eeg/sub-002_task-jacobsen_eeg.bdf"),
    _entry("code/download.py"),
    _entry("derivatives/notes.txt"),
]


class FakeResponse:
    """Minimal stand-in for the object returned by urllib.request.urlopen."""

    def __init__(self, body=b"", headers=None, on_first_read=None):
        self.body = body
        self.pos = 0
        self.headers = headers if headers is not None else {}
        self.on_first_read = on_first_read

    def read(self, size=-1):
        if self.pos == 0 and self.on_first_read is not None:
            self.on_first_read()
        if self.pos >= len(self.body):
            return b""
        if size is None or size < 0:
            size = len(self.body)
        chunk = self.body[self.pos : self.pos + size]
        self.pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(monkeypatch, handler):
    """Replace urllib.request.urlopen with handler(url, req) -> response/exception.

    Returns the list of (url, method, timeout) tuples of the calls made.
    """
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append((req.full_url, getattr(req, "method", "GET"), timeout))
        result = handler(req.full_url, req)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen)
    return calls


def _http_error(url, code, msg="Error"):
    return urllib.error.HTTPError(url, code, msg, None, None)


def test_wanted_files_top_level_includes_readme_and_bidsignore():
    paths = {e["path"] for e in download._wanted_files(FAKE_TREE, ["001"])}
    assert "README.md" in paths, "README.md must be selected as a top-level file"
    assert ".bidsignore" in paths, ".bidsignore must be selected as a top-level file"
    assert "dataset_description.json" in paths
    assert "participants.tsv" in paths
    assert "CHANGES" in paths


def test_wanted_files_pads_subject_ids():
    # "1" must find sub-001 files (zero-padding), not be used verbatim.
    paths = {e["path"] for e in download._wanted_files(FAKE_TREE, ["1"])}
    assert "sub-001/eeg/sub-001_task-jacobsen_eeg.bdf" in paths
    assert "sub-001/eeg/sub-001_task-jacobsen_events.tsv" in paths


def test_wanted_files_excludes_other_subjects():
    paths = {e["path"] for e in download._wanted_files(FAKE_TREE, ["001"])}
    assert not any(p.startswith("sub-002/") for p in paths)


def test_wanted_files_accepts_subject_id_variants():
    for variant in (1, "1", "001", "sub-001"):
        paths = {e["path"] for e in download._wanted_files(FAKE_TREE, [variant])}
        assert "sub-001/eeg/sub-001_task-jacobsen_eeg.bdf" in paths, (
            f"subject variant {variant!r} must resolve to sub-001"
        )
        assert not any(p.startswith("sub-002/") for p in paths)


def test_wanted_files_ignores_non_bids_dirs():
    paths = {e["path"] for e in download._wanted_files(FAKE_TREE, ["001"])}
    assert "code/download.py" not in paths
    assert "derivatives/notes.txt" not in paths


def test_normalize_subject_variants():
    assert download._normalize_subject(1) == "001"
    assert download._normalize_subject("1") == "001"
    assert download._normalize_subject("001") == "001"
    assert download._normalize_subject("sub-001") == "001"


def test_download_file_writes_part_and_removes_it_on_success(monkeypatch, tmp_path):
    body = b"x" * 10
    dest = tmp_path / "d" / "file.tsv"
    part_path = dest.with_suffix(dest.suffix + ".part")
    seen = {}

    def on_first_read():
        seen["during"] = sorted(p.name for p in dest.parent.iterdir())

    resp = FakeResponse(body, {"Content-Length": str(len(body))}, on_first_read)
    calls = _patch_urlopen(monkeypatch, lambda url, req: resp)

    assert download._download_file("https://example.com/file.tsv", dest) is True
    assert dest.read_bytes() == body
    # While downloading, only the .part file exists; afterwards it is gone.
    assert seen["during"] == ["file.tsv.part"]
    assert not part_path.exists()
    assert list(dest.parent.iterdir()) == [dest]
    # The request used the module timeout constant.
    assert calls and all(timeout == download.TIMEOUT_S for _, _, timeout in calls)


def test_download_file_content_length_mismatch_raises_and_cleans_up(
    monkeypatch, tmp_path
):
    body = b"short"  # 5 bytes, but Content-Length announces 10
    dest = tmp_path / "d" / "file.tsv"
    part_path = dest.with_suffix(dest.suffix + ".part")
    resp = FakeResponse(body, {"Content-Length": "10"})
    _patch_urlopen(monkeypatch, lambda url, req: resp)

    with pytest.raises(RuntimeError, match="Incomplete download"):
        download._download_file("https://example.com/file.tsv", dest)
    assert not dest.exists()
    assert not part_path.exists()


def test_download_dataset_skips_existing_file_with_matching_size(
    monkeypatch, tmp_path, capsys
):
    tree = [_entry("participants.tsv")]
    monkeypatch.setattr(download, "_fetch_tree", lambda: tree)
    dest = tmp_path / "participants.tsv"
    dest.write_bytes(b"12345")
    monkeypatch.setattr(download, "_remote_size", lambda url: 5)
    _patch_urlopen(
        monkeypatch,
        lambda url, req: AssertionError("urlopen must not run when skipping"),
    )

    downloaded = download.download_dataset(tmp_path, subjects=["001"])

    assert downloaded == []
    assert dest.read_bytes() == b"12345"  # untouched
    out = capsys.readouterr().out
    assert "[skip] participants.tsv" in out
    assert "Done: 0 downloaded, 1 skipped" in out


def test_download_dataset_404_on_s3_falls_back_to_github_raw(monkeypatch, tmp_path):
    tree = [_entry("README.md")]
    monkeypatch.setattr(download, "_fetch_tree", lambda: tree)
    monkeypatch.setattr(download, "_remote_size", lambda url: None)

    def handler(url, req):
        if url.startswith(download.S3_BASE):
            raise _http_error(url, 404, "Not Found")
        return FakeResponse(b"readme", {"Content-Length": "6"})

    calls = _patch_urlopen(monkeypatch, handler)

    download.download_dataset(tmp_path, subjects=["001"])

    assert (tmp_path / "README.md").read_bytes() == b"readme"
    assert download.MIRROR_RAW_BASE + "README.md" in [c[0] for c in calls]


def test_fetch_tree_raises_on_truncated(monkeypatch):
    payload = json.dumps({"truncated": True, "tree": []}).encode("utf-8")
    _patch_urlopen(
        monkeypatch,
        lambda url, req: FakeResponse(payload, {"Content-Type": "application/json"}),
    )
    with pytest.raises(RuntimeError, match="truncated=true.*incomplete"):
        download._fetch_tree()


def test_fetch_tree_wraps_rate_limit_in_runtime_error(monkeypatch):
    def handler(url, req):
        raise _http_error(url, 403, "rate limit exceeded")

    _patch_urlopen(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="60 requests per hour"):
        download._fetch_tree()


def test_download_dataset_continues_after_failure_and_raises_at_end(
    monkeypatch, tmp_path, capsys
):
    tree = [_entry("README.md"), _entry("participants.tsv")]
    monkeypatch.setattr(download, "_fetch_tree", lambda: tree)
    monkeypatch.setattr(download, "_remote_size", lambda url: None)

    def handler(url, req):
        if url.endswith("README.md"):
            raise _http_error(url, 404, "Not Found")  # fails on both mirrors
        return FakeResponse(b"abc", {"Content-Length": "3"})

    _patch_urlopen(monkeypatch, handler)

    with pytest.raises(RuntimeError) as exc_info:
        download.download_dataset(tmp_path, subjects=["001"])

    # The surviving file was fully downloaded before the error was raised.
    assert (tmp_path / "participants.tsv").read_bytes() == b"abc"
    assert "README.md" in str(exc_info.value)
    out = capsys.readouterr().out
    # The summary line was still printed, and README.md was not "skipped".
    assert "Done: 1 downloaded, 0 skipped" in out
