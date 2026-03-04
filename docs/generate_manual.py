#!/usr/bin/env python3
"""
StrataDock Technical Manual Generator
Generates a comprehensive PDF manual using fpdf2.
Run: python generate_manual.py
Output: stratadock_manual.pdf
"""

from fpdf import FPDF, XPos, YPos
import os

# ?? Colour palette ????????????????????????????????????????????????????????????
C_BG       = (13,  17,  23)    # near-black background (for cover)
C_ACCENT   = (139, 92, 246)    # purple accent
C_HEADING  = (30,  58, 138)    # dark blue headings
C_SUBHEAD  = (67, 56, 202)     # indigo subheadings
C_TEXT     = (30,  30,  30)    # dark text
C_MUTED    = (100, 100, 100)
C_ROW_ALT  = (245, 245, 255)   # light purple tint for table rows
C_ROW_HDR  = (200, 191, 255)   # header row
C_WARN     = (255, 237, 213)   # warning bg
C_WARN_TXT = (154,  52,  18)

OUT_FILE = os.path.join(os.path.dirname(__file__), "stratadock_manual.pdf")


class StrataDockPDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(20, 20, 20)
        self._toc = []  # (title, page, level)
        self._current_section = ""

    # ??? Header / Footer ??????????????????????????????????????????????????????
    def header(self):
        if self.page_no() <= 2:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*C_MUTED)
        self.cell(0, 8, f"StrataDock Alpha v0.7 -- Technical Manual", align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(200, 200, 220)
        self.set_line_width(0.3)
        self.line(20, 14, 190, 14)

    def footer(self):
        if self.page_no() <= 2:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*C_MUTED)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # ??? Helpers ??????????????????????????????????????????????????????????????
    def h1(self, title):
        self._toc.append((title, self.page_no(), 1))
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*C_HEADING)
        self.ln(4)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*C_ACCENT)
        self.set_line_width(0.8)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(3)
        self.set_text_color(*C_TEXT)

    def h2(self, title):
        self._toc.append((title, self.page_no(), 2))
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*C_SUBHEAD)
        self.ln(3)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)
        self.set_text_color(*C_TEXT)

    def h3(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*C_TEXT)
        self.ln(2)
        self.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def body(self, text, indent=0):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*C_TEXT)
        self.set_x(20 + indent)
        self.multi_cell(170 - indent, 5.5, text)
        self.ln(1)

    def bullet(self, text, indent=5):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*C_TEXT)
        self.set_x(20 + indent)
        self.cell(4, 5.5, "-")
        self.set_x(20 + indent + 4)
        self.multi_cell(165 - indent, 5.5, text)

    def note(self, text):
        self.set_fill_color(*C_WARN)
        self.set_text_color(*C_WARN_TXT)
        self.set_font("Helvetica", "I", 9.5)
        self.set_x(20)
        self.multi_cell(170, 5.5, f"[!]  {text}", fill=True)
        self.ln(1)
        self.set_text_color(*C_TEXT)

    def code(self, text):
        self.set_fill_color(235, 235, 245)
        self.set_text_color(80, 30, 140)
        self.set_font("Courier", "", 9)
        self.set_x(20)
        self.multi_cell(170, 5, text, fill=True)
        self.ln(1)
        self.set_text_color(*C_TEXT)
        self.set_font("Helvetica", "", 10)

    def param_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [170 // len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*C_ROW_HDR)
        self.set_text_color(30, 10, 80)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True)
        self.ln()
        alt = False
        for row in rows:
            self.set_fill_color(*(C_ROW_ALT if alt else (255, 255, 255)))
            self.set_text_color(*C_TEXT)
            self.set_font("Helvetica", "", 8.5)
            max_lines = 1
            for i, cell in enumerate(row):
                lines = self.get_string_width(str(cell)) / (col_widths[i] - 2)
                max_lines = max(max_lines, int(lines) + 1)
            row_h = 5.5 * max_lines
            x0, y0 = self.get_x(), self.get_y()
            for i, cell in enumerate(row):
                self.set_xy(x0 + sum(col_widths[:i]), y0)
                self.multi_cell(col_widths[i], 5.5, str(cell), border=1, fill=True)
            self.set_xy(x0, y0 + row_h)
            alt = not alt
        self.ln(3)


# ?? Build document ?????????????????????????????????????????????????????????????
def build():
    pdf = StrataDockPDF()

    # ?? Cover page ?????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.set_fill_color(*C_BG)
    pdf.rect(0, 0, 210, 297, style="F")
    pdf.set_xy(20, 60)
    pdf.set_font("Helvetica", "B", 42)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(170, 18, "StrataDock", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_xy(20, 82)
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(*C_ACCENT)
    pdf.cell(170, 10, "Alpha v0.7  |  Technical Manual & User Documentation", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    # decorative line
    pdf.set_draw_color(*C_ACCENT)
    pdf.set_line_width(1.2)
    pdf.line(40, 100, 170, 100)
    pdf.set_xy(20, 108)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(180, 180, 200)
    lines = [
        "AI-Accelerated Molecular Docking Pipeline",
        "Full documentation: Installation - Parameters - Features - Best Practices",
        "",
        "Developed by: Prithvi Rajan",
        "Department of Bioinformatics, Freie Universitat Berlin",
        "(c) 2026  |  StrataDock Alpha",
    ]
    for line in lines:
        pdf.set_x(20)
        pdf.cell(170, 7, line, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ?? Table of Contents placeholder page ?????????????????????????????????????
    pdf.add_page()
    pdf.set_text_color(*C_HEADING)
    pdf.set_font("Helvetica", "B", 16)
    pdf.ln(5)
    pdf.cell(0, 10, "Table of Contents", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*C_ACCENT)
    pdf.set_line_width(0.8)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(4)
    # Will be filled after building
    toc_page = pdf.page_no()

    # ??????????????????????????????????????????????????????????????????????????
    # 1. Introduction
    # ??????????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.h1("1. Introduction & Overview")
    pdf.body(
        "StrataDock is an AI-accelerated, end-to-end molecular docking pipeline built as a "
        "Streamlit web application. It integrates two industry-standard docking engines -- "
        "AutoDock Vina and GNINA -- with a suite of novel AI-powered tools including ESMFold "
        "structure prediction, HydroDock water analysis, an ADMET pharmacokinetics dashboard, "
        "a Lead Optimization R-Group Workbench, and automatic binding site detection."
    )
    pdf.body(
        "This document covers every feature, every configurable parameter, all possible "
        "parameter combinations, and best practices for getting the most out of the pipeline -- "
        "from first installation to interpreting final results."
    )

    pdf.h2("1.1 Pipeline Architecture")
    steps = [
        ("Upload", "Receptor PDB(s), Ligand files (SDF/SMI/TXT), OR fold from sequence via ESMFold"),
        ("Settings", "Engine selection, Protein Prep, Ligand Prep, Docking parameters"),
        ("Run", "Parallelized docking with live console, automatic protein preparation"),
        ("Results", "Scores, ADMET, Binding Site, 3D Viewer + HydroDock, Lead Opt"),
    ]
    pdf.param_table(["Page", "Purpose"], steps, [40, 130])

    pdf.h2("1.2 Supported Formats")
    pdf.param_table(
        ["Type", "Formats", "Notes"],
        [
            ("Receptor", ".pdb", "Standard Protein Data Bank format"),
            ("Ligand (3D)", ".sdf", "Structure-Data File with 3D coordinates"),
            ("Ligand (1D)", ".smi, .txt, .smiles", "SMILES strings, one per line"),
            ("Ligand (bundle)", ".zip", "ZIP of any above formats"),
            ("Session", ".stratadock", "StrataDock project export (pickle+base64)"),
        ],
        [30, 60, 80]
    )

    # ??????????????????????????????????????????????????????????????????????????
    # 2. Installation
    # ??????????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.h1("2. Installation")

    pdf.h2("2.1 Requirements")
    pdf.param_table(
        ["Requirement", "Version", "Notes"],
        [
            ("OS", "Ubuntu 20.04+ / WSL2", "Windows users: WSL2 with Ubuntu recommended"),
            ("Python", "3.10+", "Via Miniconda/Anaconda"),
            ("RAM", "8 GB minimum", "16 GB+ recommended for large proteins"),
            ("Disk", "5 GB+", "For dependencies and docking binaries"),
            ("GPU (optional)", "NVIDIA CUDA 12+", "Required for GNINA GPU mode"),
        ],
        [35, 40, 95]
    )

    pdf.h2("2.2 Setup Steps")
    pdf.h3("Step 1 -- Clone the repository")
    pdf.code("git clone https://github.com/your-org/stratadock.git\ncd stratadock")

    pdf.h3("Step 2 -- Run the setup script")
    pdf.code("bash setup.sh")
    pdf.body(
        "The setup script automatically: creates a conda environment (stratadock), installs "
        "all Python dependencies, downloads the GNINA binary (CPU or GPU build based on "
        "hardware detection), and verifies the installation."
    )

    pdf.h3("Step 3 -- Launch the application")
    pdf.code("streamlit run app.py\n# Or to specify port:\nstreamlit run app.py --server.port 8501")
    pdf.body("Open your browser at http://localhost:8501")

    pdf.h2("2.3 GPU Setup (Optional)")
    pdf.body(
        "For GNINA GPU acceleration, CUDA 12+ and cuDNN 9 must be installed. Run "
        "nvidia-smi to verify GPU visibility. StrataDock auto-detects NVIDIA GPUs on launch "
        "and displays a GPU indicator in the top-right corner. No manual configuration needed."
    )
    pdf.note(
        "CPU-only mode works fully with AutoDock Vina. GNINA CNN scoring in CPU mode is "
        "extremely slow -- set CNN Mode to 'none' to use pure Vina scoring when no GPU available."
    )

    # ??????????????????????????????????????????????????????????????????????????
    # 3. Upload Page
    # ??????????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.h1("3. Upload Page")

    pdf.h2("3.1 Receptor Upload")
    pdf.body(
        "Upload one or more receptor .pdb files. StrataDock supports batch (N-receptor) "
        "docking -- each receptor will be docked against all ligands in a full NxM cross-dock matrix."
    )
    pdf.bullet("Multiple receptors can be uploaded simultaneously for cross-docking studies")
    pdf.bullet("Files must be in PDB format (.pdb extension)")
    pdf.bullet("Crystal structures from RCSB PDB are directly compatible")
    pdf.bullet("Protein preparation (hydrogens, loops, protonation) is handled automatically in Settings")

    pdf.h2("3.2 Ligand Upload")
    pdf.body("Upload ligand files in any supported format. Multiple files can be uploaded simultaneously.")
    pdf.param_table(
        ["Format", "Description", "Preprocessing"],
        [
            (".sdf", "3D structure-data file", "Direct 3D coordinates used if valid; RDKit re-embeds if invalid"),
            (".smi / .smiles", "SMILES string(s)", "2D->3D embedding via RDKit ETKDG/ETKDGv3"),
            (".txt", "SMILES, one per line", "Same as .smi"),
            (".zip", "Archive of the above", "Automatically extracted and processed"),
        ],
        [30, 60, 80]
    )

    pdf.h2("3.3 Reference Ligand (Optional)")
    pdf.body(
        "Upload a reference ligand (.sdf) that is already known to bind in the target pocket "
        "(e.g. a co-crystallized ligand from the PDB). Used to automatically define the docking box "
        "via Compute Autobox."
    )

    pdf.h2("3.4 Docking Box Definition")
    pdf.body("The docking box defines the 3D search space. Two methods are available:")
    pdf.h3("Option A -- Compute Autobox from Reference Ligand")
    pdf.body(
        "Upload a reference ligand and click Compute Autobox. StrataDock fits a grid box "
        "tightly around the ligand coordinates plus a configurable padding (default 4 A)."
    )
    pdf.h3("Option B -- Suggest Pockets (Auto)")
    pdf.body(
        "Runs StrataDock's custom geometric cavity-detection algorithm. The algorithm: "
        "(1) constructs an alpha-shape of the receptor surface; (2) maps Voronoi vertices "
        "of the alpha-complex to identify interior voids; (3) filters by hydrophobicity, "
        "exposure, and volume; (4) clusters nearby vertices into pocket centroids; "
        "(5) ranks by druggability score (volume x hydrophobicity). "
        "The top 5 pockets are displayed and auto-selected as docking boxes."
    )

    pdf.h2("3.5 Fold-to-Dock -- AI Structure Prediction")
    pdf.body(
        "This feature allows docking directly from a protein sequence -- no PDB file required. "
        "Powered by ESMFold (Meta AI), a state-of-the-art language model that predicts protein "
        "3D structures from amino acid sequences with near-AlphaFold2 accuracy, completely free."
    )
    pdf.h3("How to use")
    for step in [
        "Click the ? Fold-to-Dock card in the right column of the Upload page",
        "The dialog opens -- paste any protein sequence (FASTA with >header or plain amino acids)",
        "Enter an optional structure name (default: predicted_structure.pdb)",
        "Click ? Predict -- ESMFold API is called at api.esmatlas.com",
        "The pLDDT confidence bar appears showing per-residue confidence (0-100 scale)",
        "Click ? Load as Receptor to inject the predicted PDB into the docking pipeline",
    ]:
        pdf.bullet(step)

    pdf.h3("pLDDT Confidence Score")
    pdf.param_table(
        ["pLDDT Range", "Confidence", "Interpretation"],
        [
            (">= 70", "High (green)", "Reliable structure, good for docking"),
            ("50-69", "Medium (amber)", "Reasonable but treat results cautiously"),
            ("< 50", "Low (red)", "Likely disordered region, unreliable for rigid docking"),
        ],
        [35, 40, 95]
    )
    pdf.note(
        "ESMFold works best with sequences under 400 residues. Longer sequences are "
        "supported but may time out (120 second limit). Well-folded proteins (enzymes, "
        "kinases, receptors) typically score >=80 pLDDT."
    )
    pdf.h3("Example sequences for testing")
    pdf.code(
        "# Villin HP36 (36 AA) -- fast, high pLDDT ~88\nLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLS\n\n"
        "# Ubiquitin (76 AA) -- classic test protein, pLDDT ~90\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG\n\n"
        "# Disordered peptide (low pLDDT ~30) -- for testing low-confidence visualization\nGGGGGSGGGGSGGGGSGGGGS"
    )
    pdf.h3("API Technical Details")
    pdf.param_table(
        ["Parameter", "Value"],
        [
            ("Endpoint", "https://api.esmatlas.com/foldSequence/v1/pdb/"),
            ("Method", "POST with plain sequence body"),
            ("Content-Type", "application/x-www-form-urlencoded"),
            ("Response", "PDB format text (B-factor column = pLDDT per atom)"),
            ("Timeout", "120 seconds"),
            ("Cost", "Free -- Meta AI public API"),
        ],
        [55, 115]
    )

    pdf.h2("3.6 Demo Mode")
    pdf.body(
        "Click the ? Demo button (top-right of the receptor section) to load a pre-configured "
        "demo dataset: COVID-19 Main Protease (PDB: 6LU7) as receptor, and Aspirin, Ibuprofen, "
        "and Paracetamol as test ligands. Ideal for quick pipeline validation."
    )

    # ??????????????????????????????????????????????????????????????????????????
    # 4. Settings Page
    # ??????????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.h1("4. Settings Page")
    pdf.body(
        "The Settings page configures the docking engine, protein preparation, "
        "ligand preparation, and parallelization. All settings persist across runs "
        "and can be saved/loaded as project files."
    )

    pdf.h2("4.1 Docking Engine Selection")
    pdf.param_table(
        ["Engine", "Algorithm", "GPU Needed", "Best For"],
        [
            ("AutoDock Vina", "Empirical force field + gradient descent", "No (CPU only)", "Fast screening, all hardware"),
            ("GNINA", "Vina physics + CNN deep learning scoring", "Recommended (NVIDIA)", "Accuracy-critical campaigns"),
        ],
        [35, 60, 35, 40]
    )

    pdf.h2("4.2 GNINA Parameters")
    pdf.param_table(
        ["Parameter", "Default", "Range/Options", "Description"],
        [
            ("Exhaustiveness", "8", "1-128", "Search depth. Higher = more thorough but slower. 8 is standard; 32+ for high-accuracy."),
            ("Number of Poses", "9", "1-20", "Max binding poses to generate per ligand."),
            ("CNN Model", "default", "default, crossdock, general_default0*", "Neural network model weights. 'default' = crossdock_default0."),
            ("CNN Weight", "default", "default, crossdock_default2, ...", "Model checkpoint variant. Leave 'default' unless benchmarking."),
            ("Min RMSD", "1.0", "0.1-5.0", "Minimum RMSD between poses for clustering (Angstroms)."),
            ("Seed", "0", "Any integer", "Random seed for reproducibility. 0 = random."),
            ("GPU Device", "0", "0, 1, 2, ...", "CUDA device index. 0 = primary GPU."),
            ("Flex Distance", "0 (off)", "0-5.0 A", "Enables flexible docking for side-chains within this distance of pocket center."),
            ("CNN Mode", "rescore", "none, rescore, refinement", "How the CNN integrates with pose search (see section 4.2.1)."),
        ],
        [35, 20, 45, 70]
    )

    pdf.h3("4.2.1 CNN Modes -- Detailed")
    pdf.param_table(
        ["Mode", "Speed", "Accuracy", "Mechanism"],
        [
            ("none", "Fastest", "Standard Vina accuracy", "Bypasses CNN entirely. Identical to AutoDock Vina. Best for CPU-only workflows."),
            ("rescore", "Fast (default)", "Good (+CNN scoring)", "Vina generates all poses, CNN re-scores them as post-processing. Small overhead, significant accuracy gain."),
            ("refinement", "Very slow", "Highest", "CNN actively overrides Vina during energy minimization, guiding each pose step. 5-10x slower than rescore but most biologically accurate."),
        ],
        [25, 20, 25, 100]
    )

    pdf.h3("4.2.2 Flexible Docking (flexdist)")
    pdf.body(
        "When flexdist > 0, any amino acid side-chain within that distance of the docking box "
        "center is 'unfrozen' and allowed to rotate/translate during simulation (induced-fit). "
        "This dramatically improves biological realism but has important scoring implications."
    )
    pdf.note(
        "With flexdist enabled, Vina force-field scores often become POSITIVE (e.g. +1.8 kcal/mol) "
        "due to the deformation penalty from bending the protein. ALWAYS use CNN Pose Probability "
        "as the primary metric when flexdist > 0, NOT the Vina score."
    )
    pdf.param_table(
        ["flexdist", "Effect", "Recommended Use"],
        [
            ("0 (default)", "Fully rigid receptor", "Virtual screening, fast workflows"),
            ("3.0-4.0 A", "Pocket side-chains mobile", "Validation against co-crystal, accuracy studies"),
            ("5.0 A", "Large flexible region", "Cryptic pocket exploration, large conformational changes"),
        ],
        [30, 55, 85]
    )

    pdf.add_page()
    pdf.h2("4.3 AutoDock Vina Parameters")
    pdf.param_table(
        ["Parameter", "Default", "Range/Options", "Description"],
        [
            ("Exhaustiveness", "8", "1-128", "Search depth. 8 for screening; 32 for publication-quality."),
            ("Number of Poses", "9", "1-20", "Max output poses per ligand."),
            ("Energy Range", "3.0", "0.5-10.0 kcal/mol", "Max energy difference from best pose to include in output."),
            ("Scoring Function", "vina", "vina, vinardo, ad4", "vina = standard; vinardo = optimized for drug-like molecules; ad4 = AutoDock4 FF (requires maps)."),
            ("Seed", "0", "Any integer", "0 = random. Set an integer for reproducible results."),
        ],
        [35, 20, 45, 70]
    )

    pdf.h2("4.4 Protein Preparation Parameters")
    pdf.body(
        "The protein preparation pipeline runs automatically before every docking job, "
        "executing the selected steps in order. All steps are individually toggleable."
    )
    pdf.param_table(
        ["Parameter", "Default", "Description"],
        [
            ("Remove Water", "On", "Deletes all HOH (water) HETATM records from the PDB. Recommended for most docking studies."),
            ("Remove Heteroatoms", "On", "Removes all non-protein, non-nucleic HETATM records (ligands in crystal, ions, buffers)."),
            ("Keep Metals", "On", "Exception to Remove Heteroatoms: retains metal ions (Zn, Mg, Ca, Fe etc.) as they are often structurally critical."),
            ("Add Hydrogens", "On", "Adds explicit hydrogen atoms using PDBFixer at physiological pH 7.4. Essential for correct H-bond detection."),
            ("Fix Atoms", "On", "Uses PDBFixer to add any missing heavy atoms to incomplete residues based on standard residue templates."),
            ("Fix Loops", "Off", "Models any missing loop regions using OpenMM's modeller. Slow but useful for structures with significant gaps."),
            ("propka Protonation", "Off", "Assigns protonation states using propka at pH 7.4. More accurate than default state assignment. Adds computational overhead."),
            ("Energy Minimization", "Off", "Runs OpenMM with Amber14 force field to relax the prepped structure to a local energy minimum."),
            ("Minimization Steps", "500", "Number of OpenMM gradient descent steps. 500 for light relaxation; 2000+ for full minimization."),
        ],
        [40, 18, 112]
    )

    pdf.h2("4.5 Ligand Preparation Parameters")
    pdf.body(
        "Applied to SMILES inputs converted to 3D via RDKit. SDF files with valid 3D coordinates "
        "skip embedding but still go through force field relaxation if configured."
    )
    pdf.param_table(
        ["Parameter", "Default", "Options/Range", "Description"],
        [
            ("Embed Method", "ETKDGv3", "ETKDG, ETKDGv3, random", "3D coordinate generator. ETKDGv3 = best for drug-like molecules."),
            ("Max Embed Attempts", "10", "1-50", "Retries if initial 3D embedding fails. Increase for complex/strained molecules."),
            ("Force Field", "MMFF94s", "MMFF94, MMFF94s, UFF", "MMFF94s = gold standard for drug-like; UFF = fallback for unusual chemistry."),
            ("FF Steps", "500", "0-5000", "Force field minimization steps after embedding."),
            ("Add Hydrogens", "On", "On/Off", "Add explicit H atoms before minimization."),
            ("Keep Chirality", "On", "On/Off", "Preserve specified stereocenters during 3D generation."),
            ("Merge Non-polar H", "Off", "On/Off", "Merge non-polar hydrogens into their heavy atoms for faster docking (reduces atom count)."),
            ("Hydrate Ligand", "Off", "On/Off", "Add explicit water molecules around the ligand (experimental)."),
        ],
        [35, 25, 35, 75]
    )

    pdf.h2("4.6 Parallelization Parameters")
    pdf.param_table(
        ["Parameter", "Default", "Options", "Description"],
        [
            ("n_jobs", "1", "1 - CPU_count", "Number of parallel docking jobs. -1 = use all cores. Recommended: n_cores / 2 for GNINA."),
            ("Backend", "threading", "threading, multiprocessing, loky", "Joblib backend. threading = shared memory, safe for Streamlit; multiprocessing = separate processes."),
        ],
        [30, 20, 40, 80]
    )

    # ??????????????????????????????????????????????????????????????????????????
    # 5. Run Page
    # ??????????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.h1("5. Run Page")

    pdf.h2("5.1 Execution Flow")
    pdf.body("When Run Docking is clicked, StrataDock executes the following pipeline:")
    steps = [
        ("1", "Validation", "Check receptor(s), ligand(s), and docking box are configured"),
        ("2", "Protein Prep", "Apply selected protein preparation steps (per-receptor)"),
        ("3", "Ligand Prep", "Convert SMILES to 3D SDF; validate and minimise all ligands"),
        ("4", "Docking", "Run NxM docking jobs (each receptor x each ligand) in parallel"),
        ("5", "PDBQT conversion", "Convert outputs to PDBQT, extract binding poses"),
        ("6", "Complex PDB", "Merge receptor + best pose into downloadable complex PDB"),
        ("7", "ADMET analysis", "Compute pharmacokinetic descriptors for all ligands"),
        ("8", "Binding site", "Compute residue contacts for each best pose"),
        ("9", "Results", "Render all output tabs and enable ZIP download"),
    ]
    pdf.param_table(["Step", "Phase", "Action"], steps, [12, 35, 123])

    pdf.h2("5.2 Live Console")
    pdf.body(
        "The cinematic terminal console streams real-time output from the docking engine. "
        "Lines are colour-coded: white = standard output, yellow = warnings, red = errors. "
        "The console auto-scrolls and captures all stdout/stderr from GNINA/Vina."
    )

    pdf.h2("5.3 Cross-docking Matrix")
    pdf.body(
        "When multiple receptors AND multiple ligands are uploaded, StrataDock runs a full "
        "NxM cross-docking matrix: every ligand is docked against every receptor. Results "
        "are organized by receptor in the output ZIP archive."
    )
    pdf.note(
        "For large matrices (e.g. 5 receptors x 100 ligands = 500 jobs), enable parallelization "
        "and set n_jobs to -1 to use all available CPU cores."
    )

    # ??????????????????????????????????????????????????????????????????????????
    # 6. Results Page
    # ??????????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.h1("6. Results Page")
    pdf.body("The Results page renders across 5 tabs after a docking job completes.")

    pdf.h2("6.1 Docking Scores Tab")
    pdf.body("Displays a ranked table of all ligand-receptor pairs by binding affinity.")
    pdf.param_table(
        ["Column", "Engine", "Description"],
        [
            ("Ligand", "Both", "Ligand name from input file"),
            ("Receptor", "Both", "Receptor PDB filename"),
            ("Vina Affinity (kcal/mol)", "Both", "Estimated binding free energy. More negative = stronger binding."),
            ("CNN Pose Probability", "GNINA only", "0-1 probability of true binding pose. >0.5 = highly confident."),
            ("CNN Affinity (pKd)", "GNINA only", "Empirical predicted binding affinity. Higher = better."),
            ("Pose Rank", "Both", "Rank of this pose within the receptor-ligand pair."),
        ],
        [45, 30, 95]
    )
    pdf.h3("Score Interpretation Guide")
    pdf.param_table(
        ["Vina Score (kcal/mol)", "Interpretation"],
        [
            ("< -9.0", "Strong binder -- excellent lead candidate"),
            ("-7.0 to -9.0", "Moderate binder -- promising hit"),
            ("-5.0 to -7.0", "Weak binder -- consider for scaffold optimization"),
            ("> -5.0", "Poor binder -- likely non-binding"),
            ("Positive", "Only meaningful with flexdist>0; ignore Vina score, use CNN Pose Prob"),
        ],
        [60, 110]
    )

    pdf.h2("6.2 ADMET -- Pharmacokinetics Tab")
    pdf.body(
        "Computes drug-likeness and pharmacokinetic properties for all docked ligands "
        "using RDKit molecular descriptors aligned to SwissADME methodology."
    )
    pdf.param_table(
        ["Property", "Threshold", "Description"],
        [
            ("Molecular Weight (MW)", "<= 500 Da", "Lipinski Rule 1. Heavier molecules have poor oral absorption."),
            ("LogP (lipophilicity)", "<= 5", "Lipinski Rule 2. >5 = poor solubility, membrane permeability issues."),
            ("H-Bond Donors (HBD)", "<= 5", "Lipinski Rule 3. OH and NH groups; too many = poor permeability."),
            ("H-Bond Acceptors (HBA)", "<= 10", "Lipinski Rule 4. N and O atoms; too many = poor absorption."),
            ("TPSA", "<= 140 A2", "Topological Polar Surface Area. >140 = poor CNS penetration. >90 = reduced oral absorption."),
            ("Rotatable Bonds", "<= 10", "Molecular flexibility. >10 = reduced oral bioavailability."),
            ("Bioavailability Score", "0-1", "Composite oral bioavailability estimate (SwissADME formula)."),
        ],
        [45, 30, 95]
    )
    pdf.note(
        "Lipinski's Rule of Five states that a compound is likely orally bioavailable if it "
        "satisfies at least 3 of the 4 Lipinski rules. Violations are highlighted in red."
    )

    pdf.add_page()
    pdf.h2("6.3 Binding Site Analysis Tab")
    pdf.body(
        "Lists all protein residues within contact distance of each best docking pose. "
        "Provides structural context for SAR (Structure-Activity Relationship) analysis."
    )
    pdf.param_table(
        ["Column", "Description"],
        [
            ("Residue", "Three-letter amino acid code (e.g. LYS, ASP, PHE)"),
            ("Chain", "Protein chain identifier (A, B, C, ...)"),
            ("Residue Number", "Sequence position in the PDB file"),
            ("Distance (A)", "Closest atom-atom distance between residue and ligand"),
            ("Interaction Type", "H-Bond Donor, H-Bond Acceptor, Hydrophobic, Charged, Pi-stacking"),
        ],
        [45, 125]
    )
    pdf.body("Pharmacophoric interaction types:")
    pdf.bullet("H-Bond Donor -- residue donates H to ligand acceptor (Ser, Thr, Tyr, Lys, Arg, backbone NH)")
    pdf.bullet("H-Bond Acceptor -- residue accepts H from ligand donor (Asp, Glu, His, backbone C=O)")
    pdf.bullet("Hydrophobic -- nonpolar residue within 4 A (Phe, Leu, Ile, Val, Met, Pro, Trp)")
    pdf.bullet("Charged -- ionic interaction with Arg, Lys (positive) or Asp, Glu (negative)")

    pdf.h2("6.4 3D Viewer & HydroDock Tab")

    pdf.h3("3D Interactive Viewer")
    pdf.body(
        "Uses py3Dmol to render the receptor-ligand complex interactively in the browser. "
        "The receptor is shown as a cartoon ribbon; the ligand as a ball-and-stick model. "
        "The complex PDB can be downloaded directly from this tab."
    )

    pdf.h3("HydroDock -- Water Molecule Analysis")
    pdf.body(
        "HydroDock analyses water molecules present in the binding site (from crystallographic "
        "structures or added during protein preparation) and classifies each as:"
    )
    pdf.param_table(
        ["Classification", "Colour", "Criteria", "Drug Design Implication"],
        [
            ("Stable", "Blue sphere", "Forms H-bonds to >=2 polar protein atoms OR bridges protein and ligand", "Do NOT displace -- these structural waters are part of the binding pharmacophore"),
            ("Displaceable", "Orange sphere", "Only 1 or fewer H-bond partner in protein; not bridging", "TARGET for displacement -- placing a ligand substituent here to displace this water can gain 1-2 kcal/mol"),
        ],
        [30, 25, 55, 60]
    )
    pdf.note(
        "HydroDock is most informative when the receptor PDB contains crystallographic water "
        "molecules (i.e. from an experimental PDB structure). If waters were stripped during "
        "preparation, toggle 'Remove Water' OFF in Settings to retain them."
    )

    pdf.h2("6.5 Lead Optimization R-Group Workbench")
    pdf.body(
        "The Lead Opt tab enables rapid analog generation and mini-docking directly from "
        "docking results. It is designed for fast interactive SAR exploration."
    )
    pdf.h3("Workflow")
    for i, step in enumerate([
        "Select a docked ligand from the dropdown -- its 2D structure preview is displayed",
        "The SMILES string is shown in an editable text box",
        "Apply Quick Modification buttons to append common R-groups:",
        "Or manually edit the SMILES string for arbitrary structural changes",
        "Click Reset to revert to the original ligand structure",
        "Click Run Mini-Dock -- runs Vina (exhaustiveness=4) using the original docking box",
        "Results show: original score vs. analog score, Delta kcal/mol, and pass/fail summary",
    ], 1):
        pdf.bullet(step)

    pdf.h3("Quick Modification Buttons and Their Chemistry")
    pdf.param_table(
        ["Button", "SMILES Fragment", "Medicinal Chemistry Effect"],
        [
            ("?F (Fluorine)", "F", "Improves metabolic stability (blocks CYP oxidation), increases lipophilicity, bioisostere for OH"),
            ("?CH3 (Methyl)", "C", "Methyl scanning -- steric effect for selectivity, slight lipophilicity increase"),
            ("?OH (Hydroxyl)", "O", "Adds H-bond donor/acceptor, improves solubility, can form new contacts with polar residues"),
            ("?NH2 (Amine)", "N", "H-bond donor, can form salt bridges with Asp/Glu, increases basicity and solubility"),
            ("?CF3 (Trifluoromethyl)", "C(F)(F)F", "Strong electron-withdrawing group, high metabolic stability, increases lipophilicity"),
        ],
        [30, 35, 105]
    )
    pdf.note(
        "Mini-Dock uses exhaustiveness=4 for speed. For publication-quality analog scores, "
        "export the modified SMILES and run a full docking job with standard exhaustiveness."
    )

    # ??????????????????????????????????????????????????????????????????????????
    # 7. Score Interpretation & Combinations
    # ??????????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.h1("7. Score Interpretation & Parameter Combinations")

    pdf.h2("7.1 Score Interpretation Matrix")
    pdf.param_table(
        ["Vina Score", "CNN Pose Prob", "CNN Affinity", "Verdict"],
        [
            ("< -8.0", "> 0.7", "> 8.0", "EXCELLENT -- high-priority hit, all metrics agree"),
            ("< -8.0", "< 0.3", "< 5.0", "Suspicious -- good energy but CNN disagrees; may be force-field artefact"),
            ("> -5.0", "> 0.7", "> 7.0", "Interesting -- CNN confident despite poor Vina; common with flexdist"),
            ("Positive", "> 0.5", "> 6.0", "Expected with flexdist; trust CNN Pose Probability only"),
            ("Positive", "< 0.3", "< 4.0", "Poor binder -- discard"),
        ],
        [35, 30, 28, 77]
    )

    pdf.h2("7.2 Recommended Parameter Combinations")
    pdf.param_table(
        ["Use Case", "Engine", "Exhaustiveness", "CNN Mode", "Flexdist", "Notes"],
        [
            ("Fast virtual screening\n(>100 ligands)", "Vina or GNINA", "8", "rescore", "0", "Balance of speed and quality"),
            ("Hit validation\n(10-50 ligands)", "GNINA", "16-32", "rescore", "0", "Better accuracy, manageable time"),
            ("Binding mode study\n(1-5 ligands)", "GNINA", "32", "refinement", "3.5", "Highest accuracy, allow induced fit"),
            ("CPU-only server", "Vina", "8", "N/A", "0", "GNINA too slow without GPU"),
            ("Fragment screening", "Vina", "16", "N/A", "0", "Fragments are fast to dock"),
            ("Covalent candidate\nanalysis", "GNINA", "16", "rescore", "4.0", "Flex pocket for covalent warhead placement"),
            ("Multi-receptor\ncross-dock", "Vina", "8", "N/A", "0", "n_jobs=-1 for full parallelization"),
        ],
        [45, 20, 28, 28, 20, 29]
    )

    pdf.h2("7.3 Protein Preparation Recommendations")
    pdf.param_table(
        ["Scenario", "Remove Water", "Add H", "Fix Loops", "propka", "Minimize", "Rationale"],
        [
            ("Standard docking", "Yes", "Yes", "No", "No", "No", "Fast, sufficient quality for most cases"),
            ("High-accuracy single target", "Kept", "Yes", "Yes", "Yes", "Yes (500)", "Full preparation for reliable results"),
            ("Structure with missing loops", "Yes", "Yes", "Yes", "No", "No", "Fix structural gaps first"),
            ("Metalloprotein", "Yes, keep metals", "Yes", "No", "Yes", "No", "Preserve metal coordination geometry"),
            ("NMR structure (no crystal water)", "N/A", "Yes", "No", "No", "Yes (200)", "NMR structures often need minimization"),
        ],
        [43, 18, 12, 18, 14, 14, 51]
    )

    # ??????????????????????????????????????????????????????????????????????????
    # 8. Help & Session Management
    # ??????????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.h1("8. Session Management")
    pdf.body(
        "StrataDock sessions can be exported and restored via the Help page -> Save/Load Project tab. "
        "A session file (.stratadock) captures all uploaded file data, docking settings, and "
        "box definitions as a base64-encoded pickle archive."
    )
    pdf.h2("8.1 Exported State")
    pdf.body("The following session state keys are saved:")
    exported = [
        "engine, boxes, rec_files_data, lig_files_data",
        "box_padding, multi_pocket, embed_method, max_attempts",
        "force_field, ff_steps, add_hs, keep_chirality, merge_nphs, hydrate",
        "exhaustiveness, num_poses, cnn_model, cnn_weight, min_rmsd",
        "gnina_seed, gpu_device, flexdist, cnn_mode",
        "v_exhaustiveness, v_num_poses, energy_range, scoring_fn, vina_seed",
        "n_jobs, backend",
        "prep_remove_water, prep_remove_het, prep_keep_metals, prep_add_h",
        "prep_fix_atoms, prep_fix_loops, prep_propka, prep_minimize, prep_min_steps",
    ]
    for e in exported:
        pdf.bullet(e, indent=3)

    # ??????????????????????????????????????????????????????????????????????????
    # 9. Troubleshooting
    # ??????????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.h1("9. Troubleshooting")
    issues = [
        ("Ligand failed preparation",
         "Some SMILES cannot be embedded in 3D (unusual valence, charged atoms, large macrocycles). "
         "Increase Max Embed Attempts in Settings. Validate your SMILES at rdkit.org. "
         "Try a different embed method (ETKDG instead of ETKDGv3)."),
        ("GNINA very slow on CPU",
         "Without a CUDA GPU, GNINA's CNN convolutions are glacially slow. "
         "Set CNN Mode to 'none' to use pure Vina scoring (fast). "
         "For full GNINA performance, use an NVIDIA GPU."),
        ("Vina score is positive",
         "This is expected when flexdist > 0 (flexible docking). The deformation penalty "
         "can dominate the force field score. Use CNN Pose Probability as the primary metric."),
        ("ESMFold API timeout",
         "The sequence is likely >600 residues or the free API is under load. "
         "Try under 400 AA. Retry after a short wait. The API has rate limits for research use."),
        ("pLDDT shows 0.X/100 instead of X/100",
         "ESMFold API sometimes returns pLDDT values on a 0-1 scale. "
         "StrataDock auto-detects this: if max(B-factors) <= 1.0, values are multiplied by 100. "
         "If you still see fractional values, the PDB may be malformed."),
        ("Box not computed",
         "The docking box is required. Use either: (A) upload a reference ligand and click "
         "Compute Autobox, or (B) click Suggest Pockets (Auto). Both must be done on Upload page."),
        ("Pocket detection finds no pockets",
         "Some very flat or featureless proteins have no deep cavities. Try adjusting the "
         "alpha-radius parameter, or provide a reference ligand for autobox instead."),
        ("st.DuplicateElementError",
         "Streamlit key conflict -- usually caused by switching pages rapidly. "
         "Hard-refresh the browser (Ctrl+Shift+R). If persistent, restart the Streamlit server."),
    ]
    for q, a in issues:
        pdf.h3(q)
        pdf.body(a)
        pdf.ln(1)

    # ??????????????????????????????????????????????????????????????????????????
    # 10. Quick Reference
    # ??????????????????????????????????????????????????????????????????????????
    pdf.add_page()
    pdf.h1("10. Quick Reference Card")

    pdf.h2("Key Shortcuts")
    pdf.param_table(
        ["Action", "Location"],
        [
            ("Load demo data", "Upload -> ? Demo button"),
            ("Detect binding pockets", "Upload -> Suggest Pocket (fpocket) button"),
            ("Predict structure from FASTA", "Upload -> Fold-to-Dock card"),
            ("Run docking", "Run -> Run Docking button"),
            ("Download all results", "Results -> Download ZIP button"),
            ("Export session", "Help -> Save/Load Project -> Download"),
            ("Restore session", "Help -> Save/Load Project -> Upload .stratadock"),
        ],
        [60, 110]
    )

    pdf.h2("Score Decision Flowchart")
    pdf.body("When evaluating a docking result:")
    pdf.bullet("Step 1: Is Vina score < -7.0 kcal/mol?  -> Continue (else discard unless flexible docking)")
    pdf.bullet("Step 2: Is CNN Pose Probability > 0.5?  -> High-confidence binding mode")
    pdf.bullet("Step 3: Does ADMET pass Lipinski?       -> Drug-like candidate")
    pdf.bullet("Step 4: Check Binding Site residues -- do they match known pharmacophore?")
    pdf.bullet("Step 5: Check HydroDock -- can displaceable waters be targeted by substituents?")
    pdf.bullet("Step 6: Run Lead Opt analogs to rapidly explore R-group SAR")

    pdf.h2("ESMFold Quick Reference")
    pdf.param_table(
        ["Metric", "Good", "Caution", "Unreliable"],
        [
            ("pLDDT", ">= 70 (green)", "50-69 (amber)", "< 50 (red)"),
            ("Sequence length", "< 300 AA", "300-600 AA", "> 600 AA (may timeout)"),
            ("Protein class", "Enzyme, kinase, receptor", "Membrane protein", "Disordered/IDP"),
        ],
        [50, 40, 40, 40]
    )

    pdf.h2("Scoring Summary")
    pdf.param_table(
        ["Score", "Scale", "Good Threshold", "Engine"],
        [
            ("Vina Affinity", "kcal/mol, negative", "< -7 kcal/mol", "Both"),
            ("CNN Pose Probability", "0 to 1.0", "> 0.5", "GNINA only"),
            ("CNN Affinity", "pKd units, positive", "> 7.0", "GNINA only"),
            ("pLDDT", "0 to 100", "> 70", "ESMFold"),
            ("Lipinski violations", "Integer (0-4)", "0 violations", "ADMET"),
        ],
        [50, 40, 40, 40]
    )

    # ?? Write PDF ??????????????????????????????????????????????????????????????
    pdf.output(OUT_FILE)
    print(f"PDF generated successfully: {OUT_FILE}")
    return OUT_FILE, pdf._toc


if __name__ == "__main__":
    out, toc = build()
    print(f"\nTable of Contents:")
    for title, pg, level in toc:
        indent = "  " * (level - 1)
        print(f"  {indent}{title} ............ p.{pg}")
