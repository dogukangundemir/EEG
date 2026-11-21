"""Tests for src.plotting.plot_raw_with_events (M2 slide figure)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import mne

from src.plotting import plot_raw_with_events


@pytest.fixture()
def raw():
    """Small synthetic raw: frontal + posterior EEG, 60 s at 100 Hz."""
    sfreq = 100.0
    ch_names = ["Fp1", "P1", "Pz", "POz", "PO7", "PO8", "O1", "Iz", "O2"]
    info = mne.create_info(ch_names, sfreq, "eeg")
    rng = np.random.default_rng(0)
    data = rng.normal(0, 5e-6, size=(len(ch_names), int(60 * sfreq)))
    data += (rng.random((len(ch_names), 1)) - 0.5) * 50e-6  # slow drift
    return mne.io.RawArray(data, info)


EVENTS = np.array(
    [[3000, 0, 1], [3700, 0, 3], [4700, 0, 1], [5500, 0, 3]]
)
EVENT_ID = {"reflection": 1, "random": 3}


def test_saves_figure_with_picks(raw, tmp_path):
    fname = tmp_path / "fig.png"
    posterior = ["P1", "Pz", "POz", "PO7", "PO8", "O1", "Iz", "O2"]
    fig = plot_raw_with_events(
        raw, EVENTS, EVENT_ID, picks=posterior, duration=20, fname=fname
    )
    assert fig is not None
    assert fname.exists() and fname.stat().st_size > 0
    # PO7/PO8 must be shown (they are in the figure's channel selection).
    shown = [t.get_text() for ax in fig.axes for t in ax.get_yticklabels()]
    assert "PO7" in shown and "PO8" in shown
    assert "Fp1" not in shown
    plt.close(fig)


def test_events_labelled_and_legend(raw):
    fig = plot_raw_with_events(raw, EVENTS, EVENT_ID, duration=20)
    # One legend entry per condition, with the condition names.
    texts = [t.get_text() for leg in fig.legends for t in leg.get_texts()]
    assert any("reflection" in t for t in texts)
    assert any("random" in t for t in texts)
    plt.close(fig)


def test_display_highpass_does_not_touch_data(raw):
    before = raw.get_data().copy()
    fig = plot_raw_with_events(raw, EVENTS, EVENT_ID, duration=20, highpass=0.1)
    np.testing.assert_array_equal(raw.get_data(), before)
    plt.close(fig)


REFLECTION_COLOR = "#1f77b4"  # colour-blind-safe blue
RANDOM_COLOR = "#d62728"      # colour-blind-safe red


def test_event_color_keyed_by_codes_and_matches_legend(raw, monkeypatch):
    """event_color passed to raw.plot must be keyed by the integer event
    codes and agree with the legend colours.

    MNE 1.13 resolves name keys through event_id, so either keying works;
    we key by the integer codes because that is explicit and robust.
    """
    captured = {}

    def fake_plot(self, **kwargs):
        captured.update(kwargs)
        return plt.figure()

    monkeypatch.setattr(type(raw), "plot", fake_plot)
    fig = plot_raw_with_events(raw, EVENTS, EVENT_ID, duration=20)
    plt.close(fig)

    ec = captured["event_color"]
    # Keyed by the integer codes, not the condition names.
    assert set(ec) == {1, 3}
    assert ec[EVENT_ID["reflection"]] == REFLECTION_COLOR
    assert ec[EVENT_ID["random"]] == RANDOM_COLOR

    # Legend: one entry per condition, in the same colour as its markers.
    entries = {
        t.get_text(): line.get_color()
        for leg in fig.legends
        for line, t in zip(leg.get_lines(), leg.get_texts())
    }
    assert entries[f"reflection ({EVENT_ID['reflection']})"] == REFLECTION_COLOR
    assert entries[f"random ({EVENT_ID['random']})"] == RANDOM_COLOR


def test_legend_does_not_cover_traces(raw, monkeypatch):
    """The legend must sit below the time axis, clear of traces and labels."""
    def fake_plot(self, **kwargs):
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.8])  # stand-in for the traces
        ax.axvline(0.5, color=REFLECTION_COLOR, lw=2)
        ax.axvline(0.6, color=RANDOM_COLOR, lw=2)
        return fig

    monkeypatch.setattr(type(raw), "plot", fake_plot)
    fig = plot_raw_with_events(raw, EVENTS, EVENT_ID, duration=20)
    fig.canvas.draw()
    leg_bbox = fig.legends[0].get_window_extent()
    traces_bbox = fig.axes[0].get_window_extent()
    plt.close(fig)
    assert not leg_bbox.overlaps(traces_bbox)
    assert leg_bbox.y1 <= traces_bbox.y0  # entirely below the trace axes


def test_annotations_hidden_and_untouched(raw):
    """The events array is the only marker source: the raw's own annotations
    must not appear in the figure (no "1"/"3"/"255" labels, no bands), and
    the caller's raw.annotations must be unchanged afterwards.
    """
    raw.set_annotations(
        mne.Annotations(
            onset=[5.0, 12.0, 17.0],
            duration=[0.5, 0.5, 0.5],
            description=["1", "3", "255"],
        )
    )
    before = raw.annotations.copy()

    fig = plot_raw_with_events(raw, EVENTS, EVENT_ID, start=0.0, duration=20)
    fig.canvas.draw()
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    for label in ("1", "3", "255"):
        assert label not in texts

    assert raw.annotations == before
    plt.close(fig)


def test_no_annotation_bands_with_real_plot(raw):
    """With the real MNE plot, no translucent annotation bands remain:
    the FillBetweenPolyCollection that MNE uses to draw the bands must be
    gone entirely.
    """
    raw.set_annotations(
        mne.Annotations(onset=[5.0], duration=[10.0], description=["3"])
    )
    posterior = ["P1", "Pz", "POz", "PO7", "PO8", "O1", "Iz", "O2"]
    fig = plot_raw_with_events(raw, EVENTS, EVENT_ID, picks=posterior, duration=20)
    fig.canvas.draw()
    fills = [
        c
        for ax in fig.axes
        for c in ax.collections
        if type(c).__name__ == "FillBetweenPolyCollection"
    ]
    assert len(fills) == 0
    plt.close(fig)


def test_event_markers_visible_at_slide_size(tmp_path):
    """Markers are drawn above the traces and keep their colour at 50 % size.

    Regression test: MNE draws the event lines 0.5 px wide *below* the traces,
    so on a downscaled slide some markers looked grey instead of their colour.
    """
    import matplotlib.image as mpimg
    from matplotlib.collections import LineCollection

    sfreq = 256.0
    rng = np.random.default_rng(0)
    info = mne.create_info([f"EEG{i:02d}" for i in range(20)], sfreq, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((20, int(30 * sfreq))) * 20e-6, info, verbose=False)
    onsets = [3.0, 7.0, 11.0, 15.0]
    codes = [3, 1, 3, 1]
    events = np.array([[int(t * sfreq), 0, c] for t, c in zip(onsets, codes)])
    event_id = {"reflection": 1, "random": 3}
    expected = {1: (31, 119, 180), 3: (214, 39, 40)}

    fname = tmp_path / "raw.png"
    fig = plot_raw_with_events(raw, events, event_id, start=0.0, duration=20, fname=fname)

    ax = fig.mne.ax_main
    trace_z = max(line.get_zorder() for line in ax.lines)
    marker_colls = [c for c in ax.collections if isinstance(c, LineCollection) and c.get_zorder() == 10010]
    assert marker_colls, "event LineCollection not restyled"
    assert all(c.get_zorder() > trace_z for c in marker_colls)
    assert all(min(c.get_linewidths()) >= 2 for c in marker_colls)

    # Downscale the saved PNG to 50 % (slide size) and check each marker colour.
    img = mpimg.imread(fname)[:, :, :3]
    small = img[::2, ::2]  # nearest-neighbour 50 % downscale
    # bbox_inches="tight" crops the canvas, so locate markers by colour:
    for code, rgb in expected.items():
        target = np.array(rgb) / 255.0
        dist = np.sqrt(((small - target) ** 2).sum(axis=2)) * 255
        cols = np.where((dist < 80).sum(axis=0) > small.shape[0] * 0.3)[0]
        # two events per condition -> two separate column groups expected
        groups = np.split(cols, np.where(np.diff(cols) > 3)[0] + 1) if len(cols) else []
        assert len(groups) >= 2, f"event code {code}: expected 2 visible markers, found {len(groups)}"
    plt.close(fig)
