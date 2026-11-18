import sys
from pathlib import Path

import numpy as np
import pytest

# Make src/ importable when pytest is run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import io  # noqa: E402

BDF_PATH = io.BIDS_ROOT / "sub-001" / "eeg" / "sub-001_task-jacobsen_eeg.bdf"

pytestmark = pytest.mark.skipif(
    not BDF_PATH.exists(),
    reason=f"sub-001 BDF not downloaded yet ({BDF_PATH})",
)


@pytest.fixture(scope="module")
def raw():
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


def test_events_both_conditions_between_70_and_90(events):
    events_out, event_id = events
    assert "reflection" in event_id and "random" in event_id
    set(event_id) == set(io.EVENT_ID)
    for name in io.EVENT_ID:
        n = int((events_out[:, 2] == io.EVENT_ID[name]).sum())
        assert 70 <= n <= 90, f"{name} has {n} events, expected 70-90"


def test_events_counts_printout(events, capsys):
    # The task asks for the actual counts to be visible in the pytest output:
    # run with `-s` to see this print.
    events_out, event_id = events
    counts = {name: int((events_out[:, 2] == code).sum()) for name, code in event_id.items()}
    print(f"event counts for sub-001: {counts}")
    assert sum(counts.values()) == len(events_out)
    assert events_out[:, 2].tolist().count(io.EVENT_ID["reflection"]) > 0
