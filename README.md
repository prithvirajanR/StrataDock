# StrataDock v 1.6.01

StrataDock is a Linux-first molecular docking workstation for receptor-ligand screening. It wraps receptor preparation, ligand preparation, binding-site selection, AutoDock Vina docking, optional GNINA CPU/CUDA docking, interaction extraction, ADMET-style ligand profiling, 3D viewing, reporting, validation, and session export into one reproducible workflow.

This repository is the clean modular rebuild. The older one-file StrataDock prototype has been replaced by a package-oriented layout with tests and a one-command Ubuntu/WSL setup path.

## What StrataDock Does

StrataDock is designed for users who usually have:

- one or more protein receptor `.pdb` files
- one ligand file, such as `.sdf`, `.smi`, `.txt`, or a `.zip` containing ligand files
- sometimes a known reference ligand, but often no reference ligand at all

The app can:

- prepare protein receptors for docking
- prepare ligands from SDF or SMILES-like inputs
- define docking boxes from a reference ligand, manual coordinates, or fpocket-predicted binding pockets
- dock one ligand, a ligand library, or one ligand/library across multiple receptors
- run AutoDock Vina by default
- run GNINA in CPU mode, with NVIDIA CUDA support when available
- keep AMD GPU machines on supported CPU paths, because official GNINA GPU acceleration is CUDA/NVIDIA-based
- score, rank, filter, and visualize docking results
- generate CSV, JSON, HTML, PDF, PyMOL, 3Dmol, and full-session ZIP artifacts
- run bundled validation cases to check whether the installation behaves sanely

## Current Version

`v 1.6.01`

Python package version: `1.6.1`

## Supported Platforms

Primary target:

- Ubuntu Linux
- WSL2 Ubuntu on Windows

Known-good style of use:

- clone or copy the repo into a Linux-native path such as `~/StrataDock`
- run `bash setup_ubuntu.sh`
- start with `bash run_stratadock.sh`

Windows is not the primary runtime target. The app may be edited from Windows, but docking and setup are expected to run in Ubuntu/WSL.

## Quick Start

```bash
git clone https://github.com/prithvirajanR/StrataDock.git
cd StrataDock
bash setup_ubuntu.sh
bash run_stratadock.sh
```

Then open:

```text
http://localhost:8502
```

If `localhost` has browser weirdness on Windows/WSL, try:

```text
http://127.0.0.1:8502
```

## One-Command Setup

The installer is intentionally boring and direct:

```bash
bash setup_ubuntu.sh
```

It performs these steps:

1. Installs system packages through `apt-get` when available.
2. Creates or repairs a Linux `.venv`.
3. Installs or builds fpocket.
4. Installs the Python package in editable mode.
5. Downloads bundled validation data.
6. Downloads AutoDock Vina.
7. Downloads GNINA.
8. Runs a Vina validation case.
9. Runs a GNINA CPU smoke test.
10. Writes `run_stratadock.sh`.

The setup script is meant to handle most of the annoying parts for a normal Ubuntu/WSL user. If `apt` is busy because Ubuntu unattended upgrades are running, the script waits for the package manager lock instead of telling you to delete lock files.

## Launching The App

After setup:

```bash
bash run_stratadock.sh
```

The launcher activates `.venv`, puts `.venv/bin` on `PATH`, and starts Streamlit on port `8502`.

To use another port:

```bash
STRATADOCK_PORT=8503 bash setup_ubuntu.sh
bash run_stratadock.sh
```

or edit the generated launcher.

## App Workflow

The Streamlit app is organized around the same flow as the original StrataDock UI:

1. Upload
2. Settings
3. Run
4. Results
5. Help

### Upload

Upload one or more receptor PDB files and one ligand input.

Supported receptor input:

- `.pdb`

Supported ligand inputs:

- `.sdf`
- `.smi`
- `.smiles`
- `.txt`
- `.zip` containing supported ligand files

Multiple SDF files should be placed in a `.zip` if the browser uploader or workflow needs a single ligand archive.

### Settings

Choose:

- docking engine
- binding-site mode
- receptor preparation options
- ligand preparation options
- docking parameters
- run parallelism

Docking engines:

- AutoDock Vina
- GNINA

Binding-site modes:

- Reference ligand box
- fpocket suggested pockets
- Manual box coordinates

### Run

The app creates a clean run folder, prepares inputs, resolves docking boxes, runs docking jobs, extracts interactions, builds visualizations, and writes reports.

The live terminal shows progress events while the run is executing.

### Results

Results include:

- docking score table
- ADMET/pharmacokinetic-style ligand profile
- binding-site/pocket view
- 3D viewer
- downloads

### Help

The Help page includes workflow notes and environment diagnostics.

## Docking Engines

### AutoDock Vina

Vina is the default engine. It is CPU-safe, robust, and works on normal Linux/WSL machines.

Important settings:

- `exhaustiveness`
- `num_modes`
- `energy_range`
- `seed`
- `scoring`

For quick smoke testing, `exhaustiveness=1-2` and `num_modes=1` are fast but low-confidence.

For results you actually care about, start with:

```text
exhaustiveness = 8
num_modes = 9
seed = 1
```

Then increase exhaustiveness for final checks when runtime allows.

### GNINA

GNINA is included in the Linux installer and can run in CPU mode.

GNINA outputs:

- primary affinity score
- CNNscore
- CNNaffinity

GNINA GPU acceleration expects NVIDIA CUDA hardware. AMD GPU machines are detected/explained, but official GNINA GPU acceleration is not treated as supported on AMD. On AMD systems, use AutoDock Vina or GNINA CPU mode.

## Binding-Site Definition

Binding-site definition is one of the most important parts of docking.

### Reference Ligand Mode

Use this when you have a co-crystal ligand or known ligand coordinates.

The app builds the docking box around the reference ligand.

This is usually the most reliable mode for validation/redocking.

### fpocket Mode

Use this when the user only has a protein and ligand library, with no known reference ligand.

fpocket predicts geometric pockets from the receptor alone. StrataDock can dock against the top predicted pockets, for example top 3 or top 5.

Important caution:

- fpocket pocket rank is not a guarantee of biological relevance
- docking every predicted pocket can produce nonsense poses
- positive Vina scores are possible in bad pockets
- top 3 is a practical default
- top 5 is safer for not missing a site but slower and noisier

### Manual Box Mode

Use this when you already know box center and size coordinates.

Manual boxes are useful for reproducing published docking settings or expert-selected binding sites.

## Positive Docking Scores

A positive Vina docking score is not a software crash. It means Vina completed and returned a pose, but the pose is energetically unfavorable.

StrataDock flags these as:

```text
unfavorable_score
```

Treat those rows as low-confidence or clashing poses. They should not be interpreted as useful hits.

This matters most when:

- docking into fpocket-predicted pockets
- using low exhaustiveness
- using only one output pose
- docking a ligand against unrelated receptors

For meaningful interpretation, rerun suspicious cases with stronger settings:

```text
exhaustiveness = 8 or higher
num_modes = 9
```

## Receptor Preparation

The receptor preparation pipeline can:

- remove waters
- remove non-protein hetero atoms
- keep metals
- handle alternate locations
- write cleaned receptor PDB
- write receptor PDBQT
- write receptor preparation reports

Outputs include:

- cleaned PDB
- receptor PDBQT
- receptor report JSON
- receptor report TXT

## Ligand Preparation

The ligand preparation pipeline can:

- load ligands from SDF or SMILES-style files
- handle multi-ligand input
- embed 3D coordinates when needed
- prepare PDBQT through Meeko tooling
- compute ligand descriptors
- optionally strip salts
- optionally neutralize
- optionally set protonation pH when supported by available tools

Ligand preparation outputs include:

- prepared SDF
- ligand PDBQT
- descriptor values
- ADMET-style flags

## ADMET / Pharmacokinetic-Style Output

StrataDock computes lightweight ligand profiling values using RDKit-style descriptors.

Reported fields include:

- molecular weight
- LogP
- TPSA
- hydrogen bond donors
- hydrogen bond acceptors
- QED
- Lipinski failures
- rotatable bonds
- heavy atom count
- aromatic rings
- formal charge
- fraction Csp3
- molar refractivity
- Rule of Five pass/fail
- Veber pass/fail
- BBB penetration estimate
- hERG risk heuristic
- hepatotoxicity risk heuristic
- mutagenicity risk heuristic
- PAINS alert count
- Brenk alert count
- structural alert count

These are screening heuristics, not clinical predictions.

## Output Files

Each run folder may include:

```text
results.csv
results.json
best_by_ligand.csv
best_by_pocket.csv
run_summary.txt
run_manifest.json
session_manifest.json
report.html
report.pdf
boxes.json
run_events.jsonl
```

Per-pose artifacts include:

```text
*_pose.pdbqt
*_complex.pdb
*_interactions.json
*_interactions.csv
*_visualize.pml
*_viewer.html
```

The app also supports full session ZIP export.

## Result Columns

Important score fields:

- `docking_score`: generic primary score used by the current UI
- `docking_scores`: all parsed scores/modes
- `score_type`: `vina_score` or `gnina_affinity`
- `score_label`: human-readable score label
- `vina_score`: legacy compatibility column
- `vina_scores`: legacy compatibility column

For GNINA runs, legacy `vina_score` remains populated for backward compatibility but the correct interpretation is available through `score_type`, `score_label`, and `docking_engine`.

## Validation

StrataDock includes small co-crystal redocking validation cases.

Run one case:

```bash
source .venv/bin/activate
python scripts/validation/run_validation_case.py hiv_protease_1hsg
```

Run all validation cases:

```bash
source .venv/bin/activate
python scripts/validation/run_validation_all.py
```

Available validation cases:

- `hiv_protease_1hsg`
- `egfr_1m17`
- `abl_kinase_1iep`
- `trypsin_3ptb`
- `hsv_tk_1kim`

The validation output includes RMSD-style checks where reference native ligands are available.

## Command-Line Usage

Run a single docking job:

```bash
source .venv/bin/activate
python scripts/user/run_single.py \
  --receptor data/validation/hiv_protease_1hsg/receptor.pdb \
  --ligand data/validation/hiv_protease_1hsg/ligand_input.sdf \
  --box-from-ligand data/validation/hiv_protease_1hsg/native_ligand.pdb \
  --out-dir runs/example_single \
  --exhaustiveness 8 \
  --num-modes 9
```

Run a ligand batch:

```bash
source .venv/bin/activate
python scripts/user/run_batch.py \
  --receptor data/validation/hiv_protease_1hsg/receptor.pdb \
  --ligands data/validation/hiv_protease_1hsg/ligand_input.sdf \
  --box-from-ligand data/validation/hiv_protease_1hsg/native_ligand.pdb \
  --out-dir runs/example_batch \
  --engine vina \
  --exhaustiveness 8 \
  --num-modes 9
```

Run GNINA CPU:

```bash
source .venv/bin/activate
python scripts/user/run_batch.py \
  --receptor data/validation/hiv_protease_1hsg/receptor.pdb \
  --ligands data/validation/hiv_protease_1hsg/ligand_input.sdf \
  --box-from-ligand data/validation/hiv_protease_1hsg/native_ligand.pdb \
  --out-dir runs/example_gnina_cpu \
  --engine gnina \
  --gnina-cpu-only \
  --exhaustiveness 4 \
  --num-modes 1
```

Suggest fpocket pockets:

```bash
source .venv/bin/activate
python scripts/user/suggest_pockets.py \
  --receptor data/validation/hiv_protease_1hsg/receptor.pdb \
  --out-dir runs/pocket_preview \
  --top-n 5
```

Inspect receptor prep:

```bash
source .venv/bin/activate
python scripts/user/inspect_receptor.py \
  --receptor data/validation/hiv_protease_1hsg/receptor.pdb \
  --out-dir runs/receptor_inspection
```

## Project Layout

```text
.
|-- streamlit_app.py
|-- setup_ubuntu.sh
|-- run_stratadock.sh
|-- pyproject.toml
|-- src/stratadock/
|   |-- core/
|   |-- tools/
|   `-- ui/
|-- scripts/
|   |-- install/
|   |-- user/
|   `-- validation/
|-- tests/
`-- data/validation/
```

Important modules:

- `src/stratadock/core/batch.py`: batch and ensemble orchestration
- `src/stratadock/core/docking.py`: Vina/GNINA dispatch
- `src/stratadock/core/gnina.py`: GNINA command and parser helpers
- `src/stratadock/core/receptors.py`: receptor preparation
- `src/stratadock/core/ligands.py`: ligand loading/preparation
- `src/stratadock/core/pockets.py`: fpocket pocket suggestion
- `src/stratadock/core/reports.py`: HTML/PDF reports
- `src/stratadock/ui/tables.py`: result filtering and warning flags

## Development

Create environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip pytest
python -m pip install -e ".[ui]"
```

Run tests:

```bash
python -m pytest -q
```

Run a targeted test file:

```bash
python -m pytest tests/test_batch.py -q
```

Compile-check edited Python files:

```bash
python -m py_compile streamlit_app.py src/stratadock/core/batch.py
```

## What Is Intentionally Not Included

This repository does not commit:

- `.venv`
- generated run folders
- downloaded Vina binaries
- downloaded GNINA binaries
- built fpocket source
- Playwright screenshots/cache
- local logs
- generated reports

The installer creates/downloads those locally.

## Notes For Real Use

Docking scores are approximate screening signals. They are not experimental binding affinities.

For better confidence:

- define the binding site carefully
- prefer reference ligand boxes when available
- use fpocket top pockets only as a no-reference fallback
- rerun promising hits with higher exhaustiveness
- inspect poses visually
- compare interactions against known biology when possible
- treat positive docking scores as bad poses
- validate the installation before trusting new target screens

## Current Limitations

- GNINA GPU acceleration is only treated as supported on NVIDIA CUDA systems.
- AMD GPU systems use CPU-safe paths.
- fpocket predictions are geometric suggestions, not biological truth.
- ADMET outputs are lightweight heuristics.
- Protein structure prediction / fold-to-dock is intentionally not part of this version.
- Lead optimization is intentionally not part of this version.

## Useful Troubleshooting

Check environment:

```bash
source .venv/bin/activate
python scripts/install/check_environment.py
```

If port `8502` is busy:

```bash
lsof -i :8502
```

or choose a different port.

If Ubuntu package manager is locked, wait. Do not delete dpkg lock files.

If fpocket is missing from apt, setup builds it locally from source.

If GNINA fails on GPU, rerun in CPU mode first.

If docking scores look insane, rerun the same case with:

```text
exhaustiveness = 8 or 16
num_modes = 9
```

Then compare whether the top score is stable.
