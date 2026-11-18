"""Plotting helpers for the on004347 (Makin et al. 2012) dataset.

Currently two functions used by ``notebooks/01_explore.ipynb``:

- :func:`plot_raw_with_events` — continuous EEG with the stimulus events
  marked (M2 deliverable: "plot of continuous EEG with events marked").
- :func:`plot_psd` — power spectral density, to document line noise and
  drift.

Callers choose the matplotlib backend themselves; the notebook runs headless
under nbconvert, where matplotlib uses ``Agg`` automatically.
"""

from __future__ import annotations

import matplotlib


def plot_raw_with_events(
    raw,
    events,
    event_id: dict[str, int],
    start=None,
    duration: float = 20.0,
    n_channels: int = 30,
    fname=None,
):
    """Plot continuous EEG with stimulus events marked.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous data (``src.io.load_raw`` output).
    events : np.ndarray, shape (n_events, 3)
        Event matrix (``src.io.get_events`` output).
    event_id : dict
        Name -> integer code, e.g. ``{"reflection": 1, "random": 3}``.
    start : float | None
        Time (s) of the first sample shown.  If ``None`` (default), the
        window is placed on the first event minus 2 s so that a few
        stimulus events are visible.
    duration : float
        Seconds shown (default 20).
    n_channels : int
        How many channels to show (default 30, plenty for 64 scalp EEG).
    fname : str | pathlib.Path | None
        If given, save the figure here as a PNG (dpi=150).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if start is None:
        # Show the stretch around the first few events, so the plot actually
        # contains event markers (raw.first_samp is 0 for these BDF files).
        first_sample = int(events[0, 0])
        start = max(first_sample / raw.info["sfreq"] - 2.0, 0.0)

    fig = raw.plot(
        events=events,
        start=start,
        duration=duration,
        n_channels=n_channels,
        show=False,
        scalings=dict(eeg=50e-6, eog=150e-6),
    )
    if fname is not None:
        fig.savefig(str(fname), dpi=150, bbox_inches="tight")
    return fig


def plot_psd(raw, fmax: float = 70, fname=None):
    """Plot the power spectral density (PSD) of the EEG channels.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous data (``src.io.load_raw`` output).
    fmax : float
        Maximum frequency shown, in Hz (default 70).  70 Hz is chosen so
        that the 50 Hz power-line component (``PowerLineFrequency = 50`` in
        the dataset's ``eeg.json``) is clearly visible together with the
        surrounding 1/f slope.  A visible line-noise peak documents that we
        know our data — and, if it ever mattered, where a notch filter
        would go.
    fname : str | pathlib.Path | None
        If given, save the figure here as a PNG (dpi=150).

    Returns
    -------
    matplotlib.figure.Figure
    """
    psd = raw.compute_psd(fmax=fmax, picks="eeg")
    fig = psd.plot(show=False)
    _save(fig, fname)
    return fig


def _save(fig, fname) -> None:
    """Save *fig* to *fname* (dpi=150) when *fname* is given."""
    if fname is not None:
        fig.savefig(str(fname), dpi=150, bbox_inches="tight")
