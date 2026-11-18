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
