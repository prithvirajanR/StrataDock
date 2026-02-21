# ⚗️ StrataDock

**StrataDock** is an end-to-end, AI-powered drug discovery and molecular docking suite. Designed as a modern, parallelized replacement for legacy pipelines (like POAP), StrataDock operates entirely within a clean, interactive browser application.

It integrates industry-leading open-source cheminformatics engines (GNINA, AutoDock Vina, RDKit, and Meeko) to handle raw chemical text ingestion, 3D structure generation, AI binding prediction, toxicity screening, and interactive 3D visualization.

---

## 🌟 Key Features

### 1. The Core Engines

- **GNINA**: The primary AI docking engine. GNINA uses a Convolutional Neural Network (CNN) trained on thousands of known protein-ligand crystal structures. It physically evaluates 3D atomic interactions and scores them like a human expert, drastically outperforming standard empirical scoring.
- **AutoDock Vina**: The industry-standard fallback engine for rapid, reliable empirical energy scoring.
- **RDKit & Meeko**: The backbone tools used to prepare ligands. StrataDock natively translates 1D chemical text strings (SMILES) into optimized 3D physical structures (`.pdbqt`) ready for docking.

### 2. Intelligent Upload & Configuration Pipeline

- **High-Throughput Cross-Docking**: Upload dozens of `.PDB` protein structures and `.SDF`/`.SMI` ligand files simultaneously to test vast libraries across multiple receptor targets.
- **Automated 3D Embedding (ETKDGv3)**: If you upload a `SMILES` string, StrataDock automatically builds a 3D model, runs a Force Field simulation (MMFF94 or UFF) to optimize geometry, and readies it for docking.
- **Physiological pH Adjustments**: Set a target pH (e.g., pH 7.4 for blood) in the Settings. StrataDock utilizes OpenBabel to add or remove protons from the ligand _before_ docking begins, ensuring accurate biological simulation.

### 3. The "Auto-Box" Binding Site Detector

Say goodbye to manually measuring X/Y/Z coordinate grids perfectly:

- **Reference Ligand Target**: Upload a known drug that binds to the pocket, and StrataDock mathematically draws a 3D bounding box precisely encompassing the binding site, padded by a specified number of Angstroms.
- **Whole Protein Blind Docking**: If no reference ligand is provided, StrataDock maps the extreme boundaries of the entire protein to run a global "blind" docking search.
- **Interactive 3D Grid Previewer**: Before launching a run, click "Preview Box" to spin a 3D model of your protein and visually verify the exact placement of the binding grid.

### 4. Massive Parallel Execution

- **CPU Multi-threading**: The application leverages `joblib` to auto-detect your processor cores, running docking jobs simultaneously across parallel threads.
- **Hardware Agnostic**: Run exactly the same codebase on Windows (via WSL2), native Linux, NVIDIA GPUs, or pure CPU environments. The system intelligently detects hardware at launch and configures GNINA accordingly.

### 5. High-Fidelity Results & Analytics Hub

When the batch docking finishes, the results are organized into four dynamic tabs:

- **📊 Docking Scores**: A data table showing Receptor/Ligand pairs, `Vina Empirical Score` (binding strength in kcal/mol), and the `GNINA CNN Affinity` (the neural network's potency prediction).
- **💊 Pharmacokinetics (ADMET)**: A dedicated RDKit screening module. Calculates "Drug-Likeness" including Molecular Weight, Lipophilicity (LogP), Blood-Brain Barrier Penetration (TPSA), Rule of 5 Violations, and the master `QED` score.
- **📈 Binding Analytics**: Interactive Plotly scatter plots allowing you to visually identify highly potent outliers where the Neural Network disagrees with empirical scoring.
- **🧊 3D Viewer**: Embedded `py3Dmol`. Select any ligand-receptor hit to instantly view the physical binding pose in 3D. Toggle between "Cartoon Ribbon" structures and solid "Molecular Surface" skins.

### 6. Archiving & Reproducibility

- **"Full Experiment" (.zip)**: One massive download button that generates an archive containing a clean CSV of all scores/ADMET data, a formatted summary PDF report, all input PDB structures, every output docked SDF pose, and a custom `visualize.pml` PyMOL script to instantly load the entire project natively in PyMOL Desktop.
- **Project Sessions (.stk)**: Export a "StrataDock Project" save state. A month later, upload that single `.stk` file to immediately restore your entire browser session, uploaded proteins, and tuning parameters without having to re-calculate a single score!

---

## 🛠️ Installation Guide (Zero-Code Setup)

StrataDock comes with a bulletproof, one-click setup script designed specifically for non-programmers. It safely sandboxes all complex math libraries, AI binaries, and Python dependencies so it will never crash your local environment.

### Prerequisites (Windows Users)

If you are on Windows, you must have Windows Subsystem for Linux (WSL2) installed.

1. Open PowerShell as Administrator.
2. Run: `wsl --install -d Ubuntu-22.04`
3. Restart your computer if prompted. Create a username and password when Ubuntu launches.

### Installation Steps

Whether you are on native Linux, macOS, or inside WSL2 (Ubuntu), the installation is identical:

1. Download or extract the `StrataDock` folder to your computer.
2. Open your terminal and navigate inside the folder:
   ```bash
   cd path/to/StrataDock
   ```
3. Run the master setup script:
   ```bash
   bash setup.sh
   ```
   _This may take several minutes. The script will dynamically scan your hardware (checking for NVIDIA graphics) and automatically install the precise drivers, Conda mathematical engines, and UI requirements without any user input._

### Running the Application

Whenever you want to use StrataDock, simply navigate to the folder and run the launcher:

```bash
./run.sh
```

The terminal will confirm that StrataDock has launched. Open your web browser (Chrome, Edge, Safari, etc.) and go to:
**`http://localhost:8501`**

---

## 📂 Folder Structure

- **`/ligands`**: Drop all your `.sdf` or `.smi` files here before uploading them.
- **`/receptors`**: Drop all your target `.pdb` protein files here.
- **`app.py`**: The core Streamlit web application.
- **`setup.sh` & `run.sh`**: The environment manager and application launcher.
