import mne
import numpy as np
import pytest

from src import io  # noqa: E402

BDF_PATH = io.BIDS_ROOT / "sub-001" / "eeg" / "sub-001_task-jacobsen_eeg.bdf"


# Data-dependent tests are skipped via the `raw` fixture below (not a module
# mark), so the synthetic get_events test runs even without the dataset.
@pytest.fixture(scope="module")
def raw():
    if not BDF_PATH.exists():
        pytest.skip(f"sub-001 BDF not downloaded yet ({BDF_PATH})")
    return io.load_raw("001", preload=True)


@pytest.fixture(scope="module")
def events(raw):
    return io.get_events(raw)


def test_64_eeg_channels(raw):
    types = raw.get_channel_types()
    n_eeg = sum(t == "eeg" for t in types)
    assert n_eeg == 64


def test_exg_are_eog(raw):
    types = dict(zip(raw.ch_names, raw.get_channel_types()))
    for i in range(1, 9):
        name = f"EXG{i}"
        assert name in types, f"{name} missing from raw.ch_names"
        assert types[name] == io.EXTERNAL_CH_TYPES[name], (
            f"{name} should be '{io.EXTERNAL_CH_TYPES[name]}', got {types[name]}"
        )


def test_sfreq_512(raw):
    assert raw.info["sfreq"] == 512.0


def test_status_channel_removed(raw):
    assert "Status" not in raw.ch_names
    assert "Status" not in [n for n, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]


def test_po7_po8_present(raw):
    for ch in ("PO7", "PO8"):
        assert ch in raw.ch_names
    # Montage positions must be set (not "n/a") for PO7/PO8 so that
    # topographic plots work later.
    loc = {ch: c["loc"] for ch, c in zip(raw.ch_names, raw.info["chs"])}
    for ch in ("PO7", "PO8"):
        assert np.all(np.isfinite(loc[ch][:3])), f"{ch} has no electrode position"


def test_montage_is_biosemi64(raw):
    """Positions in the loaded raw must match the standard biosemi64 template.

    electrodes.tsv is in CTF coordinates with quoted names, so mne-bids
    cannot apply it (see src/io.py docstring) and we set the standard
    template by hand.  This pins the positions to that template.
    """
    eeg_names = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]
    info = mne.create_info(eeg_names, sfreq=512.0, ch_types="eeg")
    reference = mne.io.RawArray(
        np.zeros((len(eeg_names), 2)), info, verbose=False
    )
    reference.set_montage("biosemi64")

    loc = {ch: c["loc"] for ch, c in zip(raw.ch_names, raw.info["chs"])}
    ref_loc = {ch: c["loc"] for ch, c in zip(reference.ch_names, reference.info["chs"])}
    for ch in ("PO7", "PO8"):
        np.testing.assert_allclose(loc[ch][:3], ref_loc[ch][:3], rtol=0, atol=1e-6)


def test_events_both_conditions_between_70_and_90(events):
    events_out, event_id = events
    assert "reflection" in event_id and "random" in event_id
    assert set(event_id) == set(io.EVENT_ID)
    for name in io.EVENT_ID:
        n = int((events_out[:, 2] == io.EVENT_ID[name]).sum())
        assert 70 <= n <= 90, f"{name} has {n} events, expected 70-90"


def test_events_counts_printout(events, capsys):
    # The task asks for the actual counts to be visible in the pytest output:
    # run with `-s` to see this print.
    events_out, event_id = events
    counts = {name: int((events_out[:, 2] == code).sum()) for name, code in event_id.items()}
    print(f"event counts for sub-001: {counts}")

    # Independent check: the counts must equal the per-code counts read
    # directly from the subject's events.tsv sidecar (value column).
    events_tsv = io.BIDS_ROOT / "sub-001" / "eeg" / "sub-001_task-jacobsen_events.tsv"
    with open(events_tsv, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        value_col = header.index("value")
        tsv_values = [int(line.rstrip("\n").split("\t")[value_col]) for line in f]
    tsv_counts = {
        name: sum(v == code for v in tsv_values)
        for name, code in io.EVENT_ID.items()
    }
    assert counts == tsv_counts


def test_load_raw_accepts_sub_prefix(raw):
    """load_raw accepts the BIDS prefix form "sub-001" like src.download."""
    raw2 = io.load_raw("sub-001", preload=False)
    assert raw2.ch_names == raw.ch_names
    assert raw2.info["sfreq"] == raw.info["sfreq"]


def _synthetic_raw(descriptions, sfreq=512.0):
    """Build a 1-channel Raw with one annotation per description, 1 s apart."""
    info = mne.create_info(["EEG1"], sfreq=sfreq, ch_types="eeg")
    data = np.zeros((1, int(sfreq * (len(descriptions) + 1))))
    raw = mne.io.RawArray(data, info, verbose=False)
    onsets = list(range(1, len(descriptions) + 1))
    raw.set_annotations(
        mne.Annotations(onsets, np.zeros(len(onsets)), descriptions)
    )
    return raw


def test_get_events_synthetic_no_condition_merge():
    """get_events must keep 'reflection' and 'random' separate.

    'random' appears first, so mne.assigns it code 1 and 'reflection' code 2.
    The old code remapped source codes to EVENT_ID values one pass after
    another *in place*: reflection (code 2 -> 1) then random (code 1 -> 3)
    caught the freshly written reflection rows too, collapsing both
    conditions into 'random'.  A stray code-255 block-start marker is
    included and must be dropped.
    """
    raw = _synthetic_raw(["random"] * 5 + ["reflection"] * 4 + ["255"])
    events_out, event_id = io.get_events(raw)

    assert set(event_id) == set(io.EVENT_ID)
    assert set(events_out[:, 2]) == {io.EVENT_ID["reflection"], io.EVENT_ID["random"]}
    counts = {name: int((events_out[:, 2] == code).sum()) for name, code in event_id.items()}
    assert counts == {"reflection": 4, "random": 5}


def test_get_events_synthetic_no_255_marker():
    """Same as the merge test above but WITHOUT the stray '255' annotation.

    The previous test passed even on the old buggy code: the extra '255'
    annotation shifts MNE's codes (it sorts first), so 'random' got code 2
    and 'reflection' code 3, and the in-place remap no longer collided.
    Without the marker the bank order is random=1, reflection=2 — the order
    that triggers the in-place cascade on the old code — so this test fails
    on the old implementation and guards the fix.
    """
    raw = _synthetic_raw(["random"] * 5 + ["reflection"] * 4)
    events_out, event_id = io.get_events(raw)

    assert set(event_id) == set(io.EVENT_ID)
    assert set(events_out[:, 2]) == {io.EVENT_ID["reflection"], io.EVENT_ID["random"]}
    counts = {name: int((events_out[:, 2] == code).sum()) for name, code in event_id.items()}
    assert counts == {"reflection": 4, "random": 5}
