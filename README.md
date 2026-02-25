# ⚗️ StrataDock

**StrataDock** is an end-to-end, AI-powered molecular docking and drug discovery suite. Built as a modern, parallelized replacement for legacy pipelines, it runs entirely in your browser with a cinematic, dark-mode interface.

It integrates world-class open-source cheminformatics engines — **GNINA**, **AutoDock Vina**, **RDKit**, **Meeko**, and **fpocket** — to handle everything from raw chemical text ingestion and 3D geometry generation, to AI-guided binding prediction, pharmacokinetic screening, multi-pocket parallel docking, and interactive 3D pose visualization.

---

## 🌟 Full Feature Set

### 🔬 Docking Engines

| Engine            | Type                | Best For                                                                                                                      |
| ----------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **GNINA**         | AI / CNN + Physics  | Gold standard. Uses a deep convolutional neural network trained on PDB crystal structures. Accounts for receptor flexibility. |
| **AutoDock Vina** | Physics / Empirical | Industry-standard. Fast and reliable scoring using free energy approximation.                                                 |

- **CNN Scoring Modes (GNINA only)**
  - `rescore` — Physics-guided pose search, then AI grades each pose.
  - `refinement` — AI actively guides atomic movement during the pose search itself. Slower but more accurate.

---

### 🧬 Ligand Preparation Pipeline (RDKit + Meeko + OpenBabel)

StrataDock ingests ligands from multiple sources and fully automates their preparation:

1. **Input Formats:** `.SDF` (3D structures), `.SMI` (SMILES strings), multiligand `.SMI` files.
2. **3D Embedding:** If a 1D SMILES string is provided, RDKit automatically generates a 3D model using the **ETKDGv3** algorithm (the current experimental torsion-angle knowledge-based method).
3. **Force Field Optimization:** Atoms are physically relaxed using **MMFF94**, **MMFF94s**, or **UFF** to minimize internal strain energy before docking.
4. **Protonation at pH (OpenBabel):** Set a target physiological pH (default: 7.4). OpenBabel adjusts proton count on amine/acid groups to match the biological environment.
5. **PDBQT Conversion (Meeko):** The optimized 3D structure is converted into the `.pdbqt` format understood by docking engines, with options to merge non-polar hydrogens and add hydration water molecules.

**Configurable Ligand Tuning:**

- Embedding method (ETKDGv3 / ETKDGv2 / ETKDG)
- Max embedding attempts (10 – 5000)
- Force field selection & optimization steps
- Toggle explicit hydrogen addition
- Toggle chirality preservation

---

### 🧪 Receptor Preparation

- Accepts `.PDB` protein structure files.
- Automatically strips water molecules and HETATM records.
- Runs `prepare_receptor4.py` (AutoDockTools/MGLTools) to add polar hydrogens and assign Gasteiger partial charges.
- Outputs production-ready `.pdbqt` receptor files.

---

### 📍 Binding Site Detection & Grid Box Definition

Three methods available:

| Method                    | How It Works                                                                                                                                                                                                                                                            |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Auto-Detect (fpocket)** | Clicks "Suggest Pockets". StrataDock runs `fpocket` to mathematically analyze the protein's surface topology and identify the **top 5 most druggable binding cavities**, ordered by druggability score. Each cavity gets its own bounding box for concurrent screening. |
| **Reference Ligand**      | Upload a known co-crystallized ligand. StrataDock draws the box around its exact 3D coordinates, padded by a specified number of Angstroms.                                                                                                                             |
| **Blind Docking**         | No reference provided. StrataDock maps the full protein extents for a global search.                                                                                                                                                                                    |

- **Interactive Box Previewer:** Before launching a run, visualize the 3D protein and the exact placement of the search grid using an embedded `py3Dmol` renderer.

---

### ⚙️ Parallel Batch Execution

- **Multi-Ligand × Multi-Receptor × Multi-Pocket matrix:** Every permutation of (Receptor, Ligand, Pocket) is queued and executed concurrently.
- **CPU Multi-threading:** Uses Python `joblib` to auto-detect logical CPU cores and parallelize independent jobs.
- **Hardware Detection at Launch:** Automatically detects NVIDIA GPU availability and configures GNINA accordingly. Falls back to CPU mode seamlessly.
- **Real-Time Cinematic Terminal:** Every docking job streams its raw output in real-time to a dark, scrolling terminal window with color-coded lines (errors in red, scoring info in blue). The terminal glows with a pulsing cyan border while a calculation is actively running.
- **Cancel Button:** Halt the entire pipeline mid-run cleanly at any time.

---

### 📊 Results & Analytics

After docking, results are organized into a **3-tab analytics hub:**

#### Tab 1: 📊 Docking Scores

- **Heatmap Table:** Each row is a (Receptor, Ligand, Pocket) combination. The `Vina Forcefield (kcal/mol)` column is **color-coded** from deep green (most negative = strongest binding) to bright red (weakest binding), letting you instantly identify top candidates.
- **CNN Pose Probability Bar:** An embedded visual progress bar (0–1) shows the AI's confidence that each pose represents a true biological binding mode.
- **Sortable & Filterable:** Sort by Vina score, CNN score, or ligand name. Filter by a Vina score threshold slider.

#### Tab 2: 💊 Pharmacokinetics (ADMET)

Computed entirely with RDKit, no external API calls:

- **Molecular Weight** (MW < 500 Da ideal — Lipinski Rule #1)
- **Lipophilicity (LogP)** (0–5 ideal — Lipinski Rule #2)
- **Topological Polar Surface Area (TPSA)** (< 90 Å² for CNS / blood-brain barrier penetration)
- **H-Bond Donors (HBD)** & **H-Bond Acceptors (HBA)**
- **Lipinski Failures Count** (0 = fully drug-like)
- **QED Score** (Quantitative Estimate of Drug-likeness, 0.0–1.0)

#### Tab 3: 🧊 3D Pose Viewer

- Select any hit from the dropdowns to instantly render the ligand-receptor co-complex in 3D.
- Toggle between _cartoon ribbon_ (backbone view) and _molecular surface_ (surface skin) representations.
- Rotate, zoom, and inspect the exact atomic interactions driving the binding affinity.

---

### 💾 Export & Reproducibility

- **📦 Full Experiment (.zip):** One-click archive containing:
  - `stratadock_results.csv` — All docking scores and ADMET data
  - `stratadock_report.pdf` — Formatted PDF ranked hit report
  - `inputs/` — All uploaded receptor `.pdb` files
  - `outputs/` — All docked pose `.sdf` files
  - `visualize.pml` — Auto-generated PyMOL script to load the entire project in one click on the desktop app
- **📄 PDF Report Only:** Download just the formatted ranked hit report.
- **Project Sessions (.stk):** Serialize the entire browser session state to a file. Load it anytime to instantly restore uploaded files, pocket selections, and settings without re-running anything.

---

### 📖 Help & FAQ (In-App)

A dedicated **Help** page built into the app explains every concept in plain English:

- What Vina scores and CNN probabilities mean
- When to use GNINA vs AutoDock Vina
- How to interpret ADMET pharmacokinetics
- What fpocket does and when to use multiple pockets
- How receptor flexibility (induced fit) affects results

---

## 🛠️ Installation

### Prerequisites (Windows)

Install Windows Subsystem for Linux 2 (WSL2):

```powershell
wsl --install -d Ubuntu-22.04
```

Restart your computer and set up a username/password when Ubuntu launches.

### One-Command Setup

Navigate to the StrataDock folder in your Linux/WSL terminal and run:

```bash
bash setup.sh
```

The setup script automatically:

- Installs **Miniconda** if missing
- Creates an isolated `stratadock` Conda environment
- Installs all Python libraries (Streamlit, RDKit, Meeko, BioPython, fpdf2, py3Dmol, etc.)
- Downloads the correct **GNINA** binary for your hardware (GPU vs CPU)
- Installs **AutoDock Vina**, **OpenBabel**, **fpocket**, and **MGLTools**

### Launch

```bash
./run.sh
```

Open your browser and navigate to: **`http://localhost:8501`**

---

## 🖥️ System Requirements

| Component | Minimum             | Recommended           |
| --------- | ------------------- | --------------------- |
| OS        | Ubuntu 20.04 / WSL2 | Ubuntu 22.04 / WSL2   |
| RAM       | 8 GB                | 16 GB+                |
| CPU       | 4 cores             | 8+ cores              |
| GPU       | None (CPU mode)     | NVIDIA GPU (CUDA 12+) |
| Storage   | 5 GB                | 10 GB+                |

---

## 📂 Project Structure

```
StrataDock/
├── app.py           — Core Streamlit application (all UI, logic, and docking pipeline)
├── setup.sh         — Automated one-click environment installer
├── run.sh           — Application launcher
├── ligands/         — Place your .sdf or .smi ligand files here
│   └── test_drugs.smi
├── receptors/       — Place your target .pdb protein files here
└── README.md        — This file
```

---

## 🔬 Scoring Metrics Explained

| Metric                   | Unit      | Better Direction | Meaning                                                                             |
| ------------------------ | --------- | ---------------- | ----------------------------------------------------------------------------------- |
| **Vina Forcefield**      | kcal/mol  | More negative    | Estimated free energy of binding. The thermodynamic strength of the interaction.    |
| **CNN Pose Probability** | 0.0 – 1.0 | Higher           | AI confidence that this pose matches a real PDB-crystallized binding geometry.      |
| **CNN Affinity (pKd)**   | pKd       | Higher           | AI-predicted potency. `pKd = -log10(Kd)`. A pKd of 9 means nanomolar affinity.      |
| **QED**                  | 0.0 – 1.0 | Higher           | Overall drug-likeness composite score. QED > 0.7 is generally considered excellent. |

---

## 🧠 Built With

- **Streamlit** — Browser UI framework
- **GNINA** — Deep learning docking engine (CNN scoring)
- **AutoDock Vina** — Physics-based docking engine
- **RDKit** — Cheminformatics, 3D embedding, ADMET calculations
- **Meeko** — PDBQT ligand preparation
- **OpenBabel** — Protonation at physiological pH
- **fpocket** — Automated binding pocket detection
- **py3Dmol** — In-browser 3D molecular viewer
- **pandas / numpy** — Data processing
- **fpdf2** — PDF report generation
- **joblib** — CPU parallelism

---

_StrataDock — Computational Drug Discovery, Democratized._
