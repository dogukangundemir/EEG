"""Download the OpenNeuro ds004347 (NEMAR on004347) BIDS dataset.

We download directly from the OpenNeuro S3 mirror
(https://s3.amazonaws.com/openneuro.org/ds004347/<path>) instead of using
DataLad: git-annex is not available on Windows, so DataLad cannot be installed
in every team member's environment. The files on the S3 mirror are identical
to those in the NEMAR on004347 repository — the git-annex remote.log shows
that the annex remote is the OpenNeuro S3 bucket, i.e. the repository simply
points at the very same bytes.

The file list comes from the GitHub tree API of the NEMAR mirror
(https://api.github.com/repos/nemarDatasets/on004347/git/trees/main?recursive=1),
which is filtered to the top-level BIDS files plus sub-XXX/eeg/* for the
requested subjects. Files that already exist locally with the correct size
are skipped, so a repeated run downloads nothing.

Usage:
    python -m src.download --subjects 001 [002 ...] [-o | --overwrite] [--root data/on004347]
    python -m src.download --all [--root data/on004347]
"""

import argparse
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DATASET_ID = "ds004347"
S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}/"
GITHUB_TREE_URL = (
    "https://api.github.com/repos/nemarDatasets/on004347/git/trees/main?recursive=1"
)
TOP_LEVEL_SUFFIXES = (
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "README",
    "CHANGES",
)


def _remote_size(s3_url):
    """Return the Content-Length of s3_url via a HEAD request, or None."""
    req = urllib.request.Request(s3_url, method="HEAD", headers={"User-Agent": "python-urllib"})
    try:
        with urllib.request.urlopen(req) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except OSError:
        return None


def _fetch_tree():
    """Return the list of blob dicts from the GitHub tree API (main branch)."""
    req = urllib.request.Request(GITHUB_TREE_URL, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req) as resp:
        import json

        data = json.loads(resp.read().decode("utf-8"))
    return [entry for entry in data.get("tree", []) if entry.get("type") == "blob"]


def _wanted_files(tree, subjects):
    """Filter tree entries to top-level BIDS files + sub-XXX/eeg/* for subjects."""
    wanted = []
    for entry in tree:
        path = entry["path"]
        if "/" not in path:
            if path in TOP_LEVEL_SUFFIXES:
                wanted.append(entry)
        elif path.startswith(tuple(f"sub-{s}/eeg/" for s in subjects)):
            wanted.append(entry)
    return wanted


def _download_file(s3_url, dest):
    """Download s3_url to dest via a .part file, printing progress. Returns True on download."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(s3_url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req) as resp, open(part_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(
                    f"\r  {dest.name}: {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB"
                    f" ({100 * downloaded / total:.1f}%)",
                    end="",
                    flush=True,
                )
    if total and downloaded != total:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Incomplete download of {s3_url}: {downloaded} of {total} bytes"
        )
    part_path.replace(dest)
    if total:
        print()
    return True


def download_dataset(bids_root, subjects=None, overwrite=False):
    """Download top-level BIDS files and the eeg/ files of the given subjects
    from the OpenNeuro S3 mirror into bids_root (a dataset path such as
    data/on004347). Existing files with the right size are skipped unless
    overwrite=True. Returns the list of relative paths that were downloaded.
    """
    tree = _fetch_tree()
    if not subjects:  # None/empty means all subjects present in the tree
        subjects = sorted(
            {
                entry["path"].split("/")[0].removeprefix("sub-")
                for entry in tree
                if entry["path"].startswith("sub-") and "/eeg/" in entry["path"]
            }
        )
    files = _wanted_files(tree, subjects)
    if not files:
        print(f"No matching files found for subjects {subjects}")
        return []

    root = Path(bids_root)
    print(f"Target: {root.resolve()}")
    print(f"{len(files)} file(s) selected for subjects {', '.join(subjects)}\n")

    downloaded = []
    for entry in files:
        path = entry["path"]
        dest = root / path
        # Do not trust the GitHub tree size: large files are stored as git-annex
        # pointers there, so use the authoritative S3 HEAD Content-Length.
        s3_url = S3_BASE + urllib.parse.quote(path)
        size = _remote_size(s3_url)
        if (
            not overwrite
            and dest.is_file()
            and (size is None or dest.stat().st_size == size)
        ):
            print(f"[skip] {path} (already exists, {dest.stat().st_size} bytes)")
            continue
        print(f"[get ] {path} ({(size or 0) / 1e6:.1f} MB)")
        _download_file(s3_url, dest)
        downloaded.append(path)

    print(
        f"\nDone: {len(downloaded)} downloaded, "
        f"{len(files) - len(downloaded)} skipped."
    )
    return downloaded


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download OpenNeuro ds004347 from the S3 mirror."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--subjects",
        nargs="+",
        metavar="NNN",
        help="Space-separated 3-digit subject ids, e.g. --subjects 001 002",
    )
    group.add_argument(
        "--all", action="store_true", help="Download every subject in the dataset"
    )
    parser.add_argument(
        "--root", default="data/on004347", help="Target BIDS root (default: data/on004347)"
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Re-download files even if they exist with the right size",
    )
    args = parser.parse_args(argv)
    subjects = None if args.all else args.subjects
    download_dataset(args.root, subjects=subjects, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
