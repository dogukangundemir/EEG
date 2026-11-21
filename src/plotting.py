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


def plot_raw_with_events(
    raw,
    events,
    event_id: dict[str, int],
    start=None,
    duration: float = 20.0,
    picks=None,
    highpass: float = 0.1,
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
        window is chosen so that it contains as many events of *both*
        conditions as possible (each window starts 2 s before a stimulus).
    duration : float
        Seconds shown (default 20).
    picks : list of str | str | None
        Channels to show.  Default ``None`` shows the first ``n_channels``
        channels.  For slide figures pass the posterior ROI channels
        (P*/PO*/O*/Iz) so PO7 and PO8 are visible on top without frontal
        blink artifacts dominating the scale.
    highpass : float
        Display-only high-pass filter passed to ``raw.plot`` (default
        0.1 Hz).  **This filters only the drawing, never the data**: the
        BioSemi recording has no hardware high-pass, so its slow DC drift
        (several hundred µV) pushes traces through neighbouring rows and
        hides the brain signal.  A 0.1 Hz display high-pass removes that
        drift for the eye while the ``Raw`` object stays untouched
        (filtering decisions are made in M3); 0.1 Hz also leaves the SPN
        (300–1000 ms) and all other components of interest intact.
    n_channels : int
        How many channels to show when ``picks`` is not given (default 30).
    fname : str | pathlib.Path | None
        If given, save the figure here as a PNG (dpi=151).

    Returns
    -------
    matplotlib.figure.Figure
    """
    sfreq = raw.info["sfreq"]
    if start is None:
        # Choose the 20 s window that contains the most events of *both*
        # conditions (each candidate window starts 2 s before a stimulus),
        # so the markers clearly show reflection and random interleaved.
        onsets = events[:, 0] / sfreq
        codes = events[:, 2]
        best_key, best_start = None, 0.0
        for onset in onsets:
            s = max(onset - 2.0, 0.0)
            m = (onsets >= s) & (onsets <= s + duration)
            n = {c: int((codes[m] == c).sum()) for c in event_id.values()}
            key = (min(n.values()), sum(n.values()))
            if best_key is None or key > best_key:
                best_key, best_start = key, s
        start = best_start

    # One clearly different, colour-blind-safe colour per condition, fixed so
    # the markers and the legend agree: reflection = blue, random = red.
    name_color = {"reflection": "#1f77b4", "random": "#d62728"}
    fallback = ["#2ca02c", "#9467bd", "#ff7f0e"]
    for i, name in enumerate(event_id):
        name_color.setdefault(name, fallback[i % len(fallback)])
    # MNE expects event_color keyed by the integer event *code* (1, 3, ...).
    # MNE 1.13 resolves name keys through event_id, so a name-keyed dict
    # would also work; we still key by the integer code because it is
    # explicit and robust (independent of event_id changes).
    colors = {event_id[name]: name_color[name] for name in event_id}

    # The events array is the only marker source we want here, but MNE's raw
    # ploter *also* draws the raw's own annotations ("1", "3", ...) in its
    # default colours, with translucent bands: on a slide these read as
    # wrongly coloured markers.  MNE 1.13 has no explicit "hide annotations"
    # option, but annotation_regex filters which annotation labels are drawn;
    # a never-matching regex "$^" hides them all.  This is the cheapest
    # correct approach: no copy of the data (unlike raw.copy() +
    # set_annotations(None)) and no mutation of the caller's raw.annotations.
    fig = raw.plot(
        events=events,
        event_id=event_id,
        event_color=colors,
        annotation_regex="$^",
        start=start,
        duration=duration,
        picks=picks,
        n_channels=n_channels,
        highpass=highpass,
        show=False,
        scalings=dict(eeg=50e-6, eog=150e-6),
        verbose=False,
    )

    # MNE draws all event markers as one LineCollection with linewidth 0.5 at
    # zorder 10003, *below* the EEG traces (zorder 10005).  Where traces cross
    # a marker they paint over it, so at slide size some markers look grey
    # instead of their condition colour.  Draw them thicker, opaque and above
    # the traces so every marker shows its colour.
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba

    marker_rgba = {to_rgba(c) for c in colors.values()}
    all_collections = [c for ax in fig.axes for c in ax.collections]
    for coll in all_collections:
        if isinstance(coll, LineCollection) and len(coll.get_colors()) and all(
            tuple(c) in marker_rgba for c in map(tuple, coll.get_colors())
        ):
            coll.set_linewidth(2.5)
            coll.set_alpha(1.0)
            coll.set_zorder(10010)

    # MNE's raw ploter colors the event lines but may not draw a legend;
    # add a small one (one coloured line per condition) so the markers read
    # reflection / random on a static slide figure.
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=name_color[name], lw=2, label=f"{name} ({event_id[name]})")
        for name in event_id
    ]
    # Place the legend below the figure's time axis so it covers neither the
    # traces nor the channel labels at the top; the savefig call uses
    # bbox_inches="tight", which keeps this outside-of-figure legend in the
    # saved PNG.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
               ncol=len(handles), frameon=True, framealpha=1.0, fontsize="small")

    # A slightly larger canvas than the MNE defaults keeps the legend, the
    # event labels and the scale bar readable on a slide.
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * 1.08, h * 1.08)

    if fname is not None:
        fig.savefig(str(fname), dpi=151, bbox_inches="tight")
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
        If given, save the figure here as a PNG (dpi=151).

    Returns
    -------
    matplotlib.figure.Figure
    """
    psd = raw.compute_psd(fmax=fmax, picks="eeg")
    fig = psd.plot(show=False)
    # Same slide-readability treatment as in plot_raw_with_events.
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * 1.08, h * 1.08)
    _save(fig, fname)
    return fig


def _save(fig, fname) -> None:
    """Save *fig* to *fname* (dpi=151) when *fname* is given."""
    if fname is not None:
        fig.savefig(str(fname), dpi=151, bbox_inches="tight")
