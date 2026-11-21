# EEG 2026 - Symmetry and the Sustained Posterior Negativity

This repository hosts the course project for **EEG 2026 (WS 2026/27)** ([course page](https://www.s-ccs.de/course_EEG/)). The goal of the project is to reproduce a published EEG finding from an open dataset using MNE-Python, and then to extend the analysis. Specifically, we replicate the finding that reflectional symmetry elicits a stronger sustained posterior negativity (SPN) than random visual patterns, using the open combined EEG/EMG dataset published in BIDS format on NEMAR/OpenNeuro. The project is carried out by Team BrainBees and progresses through five milestones, from dataset preparation to the full analysis of all subjects and an outlook on further analyses.

## Team

Team **BrainBees**:

| Member |
|---|
| Hazal Kantar |
| Dogukan Gündemir |

## Dataset

We use the open EEG/EMG dataset published on NEMAR (also available on OpenNeuro as `ds004347`, in BIDS format):

> Makin, A. D. J., Wilton, M. M., Pecchinenda, A., & Bertamini, M. (2026). Symmetry perception and affective responses: a combined EEG/EMG study (Version v1.0.0) [Data set]. NEMAR. https://doi.org/10.82901/nemar.on004347

The study contained in the dataset is described in the original publication:

> Makin, A. D. J., Wilton, M. M., Pecchinenda, A., & Bertamini, M. (2012). Symmetry perception and affective responses: A combined EEG/EMG study. *Neuropsychologia*, 50(14), 3250-3261. https://doi.org/10.1016/j.neuropsychologia.2012.10.003

Raw data are stored in a BIDS structure under `data/`; the directory is git-ignored and raw data are never committed to the repository.

## Research question and hypothesis

**Research question:** Does reflectional symmetry elicit a stronger sustained posterior negativity (SPN) than random visual patterns?

**Hypothesis:** Reflectional symmetry produces a stronger (more negative) sustained posterior negativity than random patterns. The primary contrast of interest is **reflection vs. random**.

**Finding to replicate:** A stronger (more negative) posterior amplitude for symmetry compared to random patterns, with the SPN window at approximately 300-1000 ms after stimulus onset. The N1 (approximately 170-200 ms) is examined as a secondary component.

## Experimental paradigm

Twenty-four participants classified reflectional or random abstract patterns while EEG (64-channel BioSemi system) and facial EMG were recorded. Each trial followed the sequence:

1. Fixation (1.5 s)
2. Pattern presentation (3 s)
3. Classification response

## Planned analysis

We compute event-related potentials (ERPs) with MNE-Python and compare conditions at posterior electrodes.

| Contrast | Electrodes | Component | Time window | Method |
|---|---|---|---|---|
| Reflection vs. random (primary) | PO7 / PO8 | SPN | approx. 300-1000 ms | ERP amplitude comparison |
| Reflection vs. random (secondary) | PO7 / PO8 | N1 | approx. 170-200 ms | ERP amplitude comparison |

The SPN is the primary component of interest; the N1 is a secondary focus.

## Milestones

Slides for each milestone are due on the Monday before the presentation date at 10:00.

| Milestone | Date | Content | Status | Material |
|---|---|---|---|---|
| M1 | 28.10.2026 | Paper and dataset | Completed | [Slide](docs/milestones/M1_BrainBees.pdf) |
| M2 | 25.11.2026 | Processing pipeline: authors' pipeline vs. our pipeline, data download and loading, continuous data of one subject | In progress | Code, notebook and figures on this branch |
| M3 | 16.12.2026 | First subject analysed | Planned | - |
| M4 | 27.01.2027 | All subjects | Planned | - |
| M5 | 10.02.2027 | Further analyses and outlook | Planned | - |

## Milestone 2: data acquisition and first look

### Download

One subject requires about 116 MB of disk space. Reruns skip files that are already complete. From the repository root:

```bash
python -m src.download --subjects 001   # one subject
python -m src.download --all            # all 24 subjects
```

The data are fetched from the public OpenNeuro S3 mirror of `ds004347` into `data/on004347/` (BIDS format). This avoids the DataLad/git-annex dependency.

### Loading

`src.io.load_raw(1)` returns the continuous recording: 64 EEG channels with the standard `biosemi64` montage, 8 external EXG channels typed as EOG, and the Status channel removed. `src.io.get_events(raw)` returns the events and the `event_id` mapping `{reflection: 1, random: 3}`; the block-start code 255 is excluded.

### Exploration

The first exploration of the continuous data is carried out in `notebooks/01_explore.ipynb`. Two figures document the outcome:

- `docs/figures/sub-001_raw_events.png`: 20 s of posterior channels, including PO7 and PO8, with the reflection and random events marked.
- `docs/figures/sub-001_psd.png`: power spectrum of the recording.

### Facts about sub-001

| Property | Value |
|---|---|
| Sampling rate | 512 Hz |
| Channels | 64 EEG + 8 EXG |
| Reference | CMS/DRL (re-referencing needed) |
| Trials | 80 reflection, 79 random |
| Power line frequency | 50 Hz |
| Signal characteristics | Frontal channels dominated by blinks; PO3 is a candidate bad channel |

### Known dataset issues

- Channel names in `channels.tsv` are quoted.
- `electrodes.tsv` uses CTF coordinates, so the standard montage is applied.
- `events.json` states onsets in ms although they are in seconds.
- The assignment of the EXG channels to EOG/EMG is not documented.

### Tests

```bash
python -m pytest -q
```

The test suite comprises 24 tests.

## Repository structure

Content of this branch:

```text
EEG/
├── README.md
├── requirements.txt
├── conftest.py
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── download.py
│   ├── io.py
│   └── plotting.py
├── tests/
│   ├── test_download.py
│   ├── test_io.py
│   └── test_plotting.py
├── notebooks/
│   └── 01_explore.ipynb
├── docs/
│   ├── figures/
│   │   ├── sub-001_raw_events.png
│   │   └── sub-001_psd.png
│   └── milestones/
│       └── M1_BrainBees.pdf
└── data/          # local only, not committed
```

Later milestones will extend the repository with additional source modules in `src/` (reusable functions), analysis notebooks in `notebooks/`, and figures. Raw data live in a BIDS structure under `data/`, which is git-ignored; raw data are never committed.

Repository conventions (course rules):

- Reusable functions are implemented in `src/*.py`.
- Analyses are carried out in notebooks under `notebooks/`.
- Data follow the BIDS structure; raw data are never committed (`data/` is git-ignored).

## Getting started

Requirements: Python 3.11.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate            # Windows

# Install dependencies (includes mne and mne-bids)
pip install -r requirements.txt
```

## References

- Makin, A. D. J., Wilton, M. M., Pecchinenda, A., & Bertamini, M. (2012). Symmetry perception and affective responses: A combined EEG/EMG study. *Neuropsychologia*, 50(14), 3250-3261. https://doi.org/10.1016/j.neuropsychologia.2012.10.003
- Makin, A. D. J., Wilton, M. M., Pecchinenda, A., & Bertamini, M. (2026). Symmetry perception and affective responses: a combined EEG/EMG study (Version v1.0.0) [Data set]. NEMAR. https://doi.org/10.82901/nemar.on004347 (also on OpenNeuro as `ds004347`).
