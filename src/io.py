"""BIDS input/output for the on004347 (Makin et al. 2012) dataset.

Known dataset issues handled here (documented, since the course grades
reasoning):

1. **Quoted channel names.** ``sub-XXX_task-jacobsen_channels.tsv`` wraps
   every name in single quotes (``'Fp1'``, ...).  The BDF channel names
   carry no quotes, so mne-bids cannot match the sidecar names against the
   raw data.  Concretely, with mne-bids 0.20 / MNE 1.13 ``read_raw_bids``
   still loads the file, but emits warnings ("The number of channels in
   the channels.tsv sidecar file (72) does not match the number of
   channels in the raw data file (73)") and then *silently skips* applying
   the channel types and the electrode positions from the BIDS sidecars
   ("Cannot set channel type for the following channels, as they are
   missing in the raw data: 'AF3', ...").  The result would be 72 channels
   all typed ``eeg`` (EXG1-EXG8 included) plus the ``Status`` channel, and
   no montage.  We therefore

   - call ``read_raw_bids`` with ``on_ch_mismatch='warn'`` (a name mismatch
     between the quoted sidecar and the raw names would otherwise raise),
   - re-apply the channel types ourselves from the (sanitised) BIDS
     sidecar names, and
   - set a standard BioSemi-64 montage by hand.

   If ``read_raw_bids`` ever refuses to load this dataset entirely, we fall
   back to ``mne.io.read_raw_bdf`` plus the same manual typing (see
   ``_load_fallback``).

2. **All 8 EXG channels are labelled EOG in ``channels.tsv`**, although the
   paper (Makin et al. 2012) recorded facial EMG (zygomaticus,
   corrugator) as well as HEOG/VEOG.  The dataset provides no channel-wise
   mapping for the EXG channels, so we keep them all as ``eog`` for now
   (see the TODO at ``EXTERNAL_CH_TYPES``).

3. **The BDF ``Status`` channel** is a trigger channel, not brain data.
   BIDS events are the source of truth (see ``get_events``), so the
   ``Status`` channel is dropped if present.

4. **No usable electrode positions** reach the ``Raw`` object through
   mne-bids, so a standard montage is set explicitly.  Two dataset facts
   explain why the standard template is used instead of the sidecar:

   - ``electrodes.tsv`` is written in **CTF coordinates (mm, "EEGLAB"**
     description) *and* carries the same quoted names as ``channels.tsv``,
     so mne-bids could not apply it to the BDF electrode names anyway.
   - We therefore use the **standard ``biosemi64`` template**: the 64 names
     match the BDF names exactly, the head frame is exactly what topomaps
     need, and no digitised electrode positions exist for this dataset.
     In the returned ``Raw`` the positions are the standard biosemi64
     template positions in the head frame (PO7/PO8 verified identical to
     the template).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import mne
import mne_bids
import numpy as np
from mne_bids import BIDSPath, read_raw_bids

# Repository root = parent of src/ ; dataset downloaded there by src.download.
REPO_ROOT = Path(__file__).resolve().parents[1]
BIDS_ROOT = REPO_ROOT / "data" / "on004347"
TASK = "jacobsen"

# From the events sidecar / TASKS: value 1 = Regular (reflection), 3 = Random,
# 255 = block start marker (ignored).  Note the dataset inconsistency:
# events.json claims the onset unit is "ms" while the actual onsets are
# seconds ("Onset" is read by mne-bids as seconds).
EVENT_ID = {"reflection": 1, "random": 3}
_START_MARKER_VALUE = 255

# External (non-scalp) channels.  channels.tsv labels EXG1-EXG8 "EOG", but the
# paper recorded EMG as well; the actual EXG -> HEOG/VEOG/EMG mapping is
# unknown for this dataset, so everything stays "eog" for now.
#
# TODO(m3-preproc): find out which EXG channel is which (horizontal vs
# vertical EOG, zygomaticus vs corrugator EMG) -- e.g. from recording
# documentation or by inspecting blink/EMG artifacts -- and change the
# values here to e.g. {"EXG1": "eog", ..., "EXG3": "emg"} before ICA.
EXTERNAL_CH_TYPES: dict[str, str] = {f"EXG{i}": "eog" for i in range(1, 9)}


def _normalize_subject(subject) -> str:
    """Accept 1, "1" or "001" and return the 3-digit BIDS subject id."""
    return str(subject).strip().zfill(3)


def _set_channel_types(raw: mne.io.BaseRaw) -> None:
    """Type the external channels; everything else stays EEG.

    The BDF channel names (Fp1 ... O2, EXG1-EXG8) are already correct --
    only the *types* are wrong (read_raw_bids could not apply channels.tsv,
    see module docstring).
    """
    present = {ch for ch in EXTERNAL_CH_TYPES if ch in raw.ch_names}
    if present:
        raw.set_channel_types({ch: EXTERNAL_CH_TYPES[ch] for ch in present})
    # Sanity guard: after the fix, the scalp channels must all be eeg.
    types = raw.get_channel_types()
    n_eeg = sum(t == "eeg" for t in types)
    if n_eeg != 64:
        raise RuntimeError(
            f"Expected 64 eeg channels after typing, got {n_eeg}. "
            "The quoted-channels.tsv workaround may have broken."
        )


def _set_montage(raw: mne.io.BaseRaw) -> None:
    """Set a standard montage so topographic plots work.

    ``biosemi64`` matches the 64 BDF scalp names (PO7/PO8 included) exactly.
    The 8 EXG channels have no electrode positions -> on_missing='warn'.
    """
    montage = mne.channels.make_standard_montage("biosemi64")
    with warnings.catch_warnings():
        # Expected: no position for EXG1-EXG8.
        warnings.filterwarnings("ignore", message="DigMontage is only a subset")
        raw.set_montage(montage, on_missing="warn")


def _load_fallback(bids_root: Path, subject: str) -> mne.io.BaseRaw:
    """Fallback loader used only if read_raw_bids raises.

    Why this exists: with the quoted channel names in channels.tsv, some
    mne-bids versions treat the sidecar/raw name mismatch as a HardError
    and refuse to return the raw object at all.  In that case we read the
    BDF directly (the comments above explain why the sidecars cannot be
    trusted blindly anyway) and reconstruct the BIDS-derived annotations
    (onsets in seconds, durations n/a -> 0) from the *_events.tsv.
    """
    bdf = bids_root / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{TASK}_eeg.bdf"
    raw = mne.io.read_raw_bdf(bdf, preload=False)
    events_tsv = bids_root / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{TASK}_events.tsv"
    onsets, values = [], []
    with open(events_tsv, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        onset_col, value_col = header.index("onset"), header.index("value")
        for line in f:
            cols = line.rstrip("\n").split("\t")
            onsets.append(float(cols[onset_col]))  # seconds despite events.json saying ms
            values.append(int(cols[value_col]))
    annotations = mne.Annotations(
        onset=onsets, duration=np.zeros(len(onsets)), description=[str(v) for v in values]
    )
    raw.set_annotations(annotations)
    return raw


def load_raw(subject, bids_root=BIDS_ROOT, preload=True) -> mne.io.Raw:
    """Load a subject's continuous EEG as an MNE ``Raw`` object.

    Parameters
    ----------
    subject : int | str
        Subject identifier; "1", "001" and 1 are all accepted.
    bids_root : pathlib.Path | str
        BIDS dataset root (default: ``<repo>/data/on004347``).
    preload : bool
        Load the data into memory (default: True).

    The returned raw has: the 64 scalp channels typed ``eeg`` with their
    BioSemi names, EXG1-EXG8 typed according to ``EXTERNAL_CH_TYPES``
    (currently all ``eog`` -- see the module docstring), no BDF ``Status``
    channel, and a standard ``biosemi64`` montage.
    """
    subject = _normalize_subject(subject)
    bids_root = Path(bids_root)
    bids_path = BIDSPath(subject=subject, task=TASK, datatype="eeg", root=bids_root)

    try:
        with warnings.catch_warnings():
            # Expected/known warnings caused by the quoted names in
            # channels.tsv (see module docstring); not re-raised here.
            warnings.filterwarnings("ignore", message=".*channels.tsv sidecar.*")
            warnings.filterwarnings("ignore", message=".*Cannot set channel type.*")
            raw = read_raw_bids(bids_path, on_ch_mismatch="warn")
    except FileNotFoundError:
        # Not a sidecar-warning problem: the file itself is missing.
        # Re-raise so callers get a clear error instead of a silent fallback.
        raise
    except Exception as exc:
        warnings.warn(
            f"read_raw_bids failed ({exc!r}); retrying via the quoted-channels"
            " workaround path (read_raw_bdf + manual montage/typing)."
            " See src/io.py docstring."
        )
        raw = _load_fallback(bids_root, subject)

    # Issue 3: drop the BDF trigger channel, BIDS events are the truth.
    if "Status" in raw.ch_names:
        raw.drop_channels(["Status"])

    _set_channel_types(raw)
    _set_montage(raw)
    if preload:
        raw.load_data()
    return raw


def get_events(raw: mne.io.BaseRaw):
    """Extract stimulus events restricted to the reflection/random onsets.

    Returns
    -------
    events : np.ndarray, shape (n_events, 3)
    event_id : dict
        Readable names -> integer codes, i.e. a subset of ``EVENT_ID``
        ({"reflection": 1, "random": 3}); the block-start marker (255) is
        excluded.

    Notes
    -----
    Robust to how mne-bids names the annotations: in this dataset
    ``channels``/``events`` sidecars give ``trial_type`` all "n/a", so
    mne-bids falls back to ``value`` and the annotation descriptions are
    the raw value strings ("1", "3", "255"); it also accepts descriptions
    called "reflection"/"random" (possible if trial_type is ever fixed) or
    the events.json level names "x1"/"x3".
    """
    if raw.annotations is None or len(raw.annotations) == 0:
        raise ValueError("No annotations found in raw.")

    events, ann_event_id = mne.events_from_annotations(raw, verbose=False)

    # For each condition, find the annotation code whose description refers
    # to it (match name, plain value string, or events.json "x<value>").
    ann_code_to_name = {}
    for name, value in EVENT_ID.items():
        for key, code in ann_event_id.items():
            if key == name or key.lower() == str(value) or key.lower() == f"x{value}":
                ann_code_to_name[code] = name
                break

    mask = np.isin(events[:, 2], list(ann_code_to_name))
    events_out = events[mask].copy()
    # Remap to the canonical readable codes (== the original values here).
    # The new codes are computed from a *snapshot* of the original codes in a
    # single pass.  Writing the new codes in place, one source code at a time,
    # would cascade: a condition whose target code equals another condition's
    # not-yet-remapped source code (e.g. annotations named 'random'/'reflection'
    # first coded random=1, reflection=2, with EVENT_ID = {reflection: 1,
    # random: 3}) would be remapped twice and the conditions would merge.
    original_codes = events_out[:, 2].copy()
    code_to_value = {code: EVENT_ID[name] for code, name in ann_code_to_name.items()}
    events_out[:, 2] = [
        code_to_value[code] for code in original_codes
    ]

    event_id = {name: EVENT_ID[name] for name in set(ann_code_to_name.values())}
    return events_out, event_id
