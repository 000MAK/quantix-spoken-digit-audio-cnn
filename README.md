# Spoken Digit Recognition — 1D CNN

A raw-waveform audio CNN with an explicit speaker-held-out evaluation split.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-offline%20unit%20tests-2ea44f?style=flat)
![Data](https://img.shields.io/badge/data-fetched%20at%20runtime-6f42c1?style=flat)

## Why this project

This repository is part of Mehmet Ali Kaya's applied statistics, actuarial-risk research and machine-learning portfolio. It emphasizes explicit assumptions, repeatable scripts, deterministic seeds, data provenance and honest evaluation rather than inflated performance claims.

**Focus:** Deep learning · speech/audio.

## Data and provenance

The Free Spoken Digit Dataset (FSDD) is cloned from its maintainers at run time; recordings are not committed here.

- Source: [https://github.com/Jakobovski/free-spoken-digit-dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
- Attribution / terms: FSDD is maintained upstream and versioned using tags/Zenodo. Check the upstream repository and version metadata for current dataset licensing and citation instructions. Dataset files are not redistributed by this project.
- Data are not mirrored in this repository. Running the example requires network access and may download data or model artifacts into ignored local directories.

## Setup

Requires Python 3.11+ and an internet connection for the first full run. CPU-only execution is supported, though neural-network experiments may take a while.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py
```

## Outputs

The script writes metrics and reports under `outputs/`; model projects also save the learned state or estimator there. Local data, outputs and model artifacts are ignored by Git by default. Most arguments can be inspected with:

```bash
python main.py --help
```

## Quality and scope

The CI workflow runs Python compilation and offline unit tests. Unit tests deliberately do not download datasets or train models. Full end-to-end runs should be performed separately in an internet-enabled environment. Performance results are not included as fixed claims because they depend on the data snapshot, dependencies, hardware and run settings.

**Code license:** MIT; see [`LICENSE`](LICENSE). The code license does **not** change third-party dataset, image, audio or model terms. Review the source page before reuse, redistribution or commercial use. No customer or confidential data are used.
