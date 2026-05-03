from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from stratadock.tools.binaries import script_binary


WATER_NAMES = {"HOH", "WAT", "SOL"}
METAL_NAMES = {"ZN", "MG", "CA", "FE", "MN", "CU", "CO", "NI", "NA", "K", "CL", "CD"}
STANDARD_PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ASH", "GLH", "HID", "HIE", "HIP", "CYX", "MSE",
}


@dataclass(frozen=True)
class ReceptorPrepOptions:
    remove_waters: bool = True
    remove_non_protein_heteroatoms: bool = True
    keep_metals: bool = True
    default_altloc: str = "A"
    add_hydrogens_ph: float | None = None
    repair_with_pdbfixer: bool = False
    minimize_with_openmm: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "remove_waters": self.remove_waters,
            "remove_non_protein_heteroatoms": self.remove_non_protein_heteroatoms,
            "keep_metals": self.keep_metals,
            "default_altloc": self.default_altloc,
            "add_hydrogens_ph": self.add_hydrogens_ph,
            "repair_with_pdbfixer": self.repair_with_pdbfixer,
            "minimize_with_openmm": self.minimize_with_openmm,
        }


@dataclass(frozen=True)
class ReceptorCleanReport:
    source_path: Path
    cleaned_pdb: Path
    source_atom_records: int
    source_hetatm_records: int
    atoms_kept: int
    waters_removed: int
    waters_kept: int
    hetero_removed: int
    hetero_kept: int
    metals_kept: int
    metals_removed: int
    altlocs_normalized: int
    altlocs_dropped: int
    chains: tuple[str, ...]
    residue_count: int
    protein_residue_count: int
    water_residue_count: int
    hetero_residue_counts: dict[str, int]
    kept_hetero_residue_counts: dict[str, int]
    kept_metal_residue_counts: dict[str, int]
    removed_metal_residue_counts: dict[str, int]
    unknown_atom_residues: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ReceptorPrepReport:
    clean_report: ReceptorCleanReport
    options: ReceptorPrepOptions
    prepared_pdb: Path
    pdbqt: Path
    report_json: Path
    report_txt: Path
    prep_steps: tuple[str, ...]


def _residue_id(line: str) -> str:
    residue = line[17:20].strip().upper() or "UNK"
    chain = line[21:22].strip() or "_"
    resseq = line[22:26].strip() or "?"
    icode = line[26:27].strip()
    return f"{residue}:{chain}:{resseq}{icode}"


def _report_dict(
    report: ReceptorCleanReport,
    *,
    options: ReceptorPrepOptions | None = None,
    prepared_pdb: Path | None = None,
    pdbqt: Path | None = None,
    prep_steps: tuple[str, ...] = (),
) -> dict[str, object]:
    data: dict[str, object] = {
        "source_path": str(report.source_path),
        "cleaned_pdb": str(report.cleaned_pdb),
        "source_atom_records": report.source_atom_records,
        "source_hetatm_records": report.source_hetatm_records,
        "atoms_kept": report.atoms_kept,
        "waters_removed": report.waters_removed,
        "waters_kept": report.waters_kept,
        "hetero_removed": report.hetero_removed,
        "hetero_kept": report.hetero_kept,
        "metals_kept": report.metals_kept,
        "metals_removed": report.metals_removed,
        "altlocs_normalized": report.altlocs_normalized,
        "altlocs_dropped": report.altlocs_dropped,
        "chains": list(report.chains),
        "residue_count": report.residue_count,
        "protein_residue_count": report.protein_residue_count,
        "water_residue_count": report.water_residue_count,
        "hetero_residue_counts": report.hetero_residue_counts,
        "kept_hetero_residue_counts": report.kept_hetero_residue_counts,
        "kept_metal_residue_counts": report.kept_metal_residue_counts,
        "removed_metal_residue_counts": report.removed_metal_residue_counts,
        "unknown_atom_residues": list(report.unknown_atom_residues),
        "warnings": list(report.warnings),
    }
    if options is not None:
        data["options"] = options.as_dict()
    if prepared_pdb is not None:
        data["prepared_pdb"] = str(prepared_pdb)
    if pdbqt is not None:
        data["pdbqt"] = str(pdbqt)
    if prep_steps:
        data["prep_steps"] = list(prep_steps)
    return data


def write_receptor_reports(
    report: ReceptorCleanReport,
    *,
    options: ReceptorPrepOptions,
    prepared_pdb: Path,
    pdbqt: Path,
    prep_steps: tuple[str, ...],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _report_dict(report, options=options, prepared_pdb=prepared_pdb, pdbqt=pdbqt, prep_steps=prep_steps)
    report_json = output_dir / "receptor_report.json"
    report_txt = output_dir / "receptor_report.txt"
    report_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    lines = [
        "StrataDock receptor preparation report",
        "=" * 39,
        f"Source: {report.source_path}",
        f"Cleaned PDB: {report.cleaned_pdb}",
        f"Prepared PDB: {prepared_pdb}",
        f"PDBQT: {pdbqt}",
        "",
        "Options:",
        f"- Remove waters: {options.remove_waters}",
        f"- Remove non-protein HETATM records: {options.remove_non_protein_heteroatoms}",
        f"- Keep metals/ions: {options.keep_metals}",
        f"- Default altloc: {options.default_altloc}",
        f"- Add hydrogens pH: {options.add_hydrogens_ph if options.add_hydrogens_ph is not None else '(disabled)'}",
        f"- PDBFixer repair: {options.repair_with_pdbfixer}",
        f"- OpenMM minimization: {options.minimize_with_openmm}",
        "",
        f"Chains: {', '.join(report.chains) if report.chains else '(none)'}",
        f"Protein residues: {report.protein_residue_count}",
        f"Residues total: {report.residue_count}",
        f"Atoms kept: {report.atoms_kept}",
        f"Waters removed: {report.waters_removed}",
        f"Waters kept: {report.waters_kept}",
        f"Non-protein HETATM records removed: {report.hetero_removed}",
        f"Non-protein HETATM records kept: {report.hetero_kept}",
        f"Metals kept: {report.metals_kept}",
        f"Metals removed: {report.metals_removed}",
        f"Altlocs normalized: {report.altlocs_normalized}",
        f"Altlocs dropped: {report.altlocs_dropped}",
        f"Prep steps: {', '.join(prep_steps) if prep_steps else 'clean_only'}",
    ]
    if report.hetero_residue_counts:
        lines.append("")
        lines.append("Removed hetero residues:")
        for name, count in sorted(report.hetero_residue_counts.items()):
            lines.append(f"- {name}: {count} atom(s)")
    if report.kept_hetero_residue_counts:
        lines.append("")
        lines.append("Kept hetero residues:")
        for name, count in sorted(report.kept_hetero_residue_counts.items()):
            lines.append(f"- {name}: {count} atom(s)")
    if report.kept_metal_residue_counts:
        lines.append("")
        lines.append("Kept metal/ion residues:")
        for name, count in sorted(report.kept_metal_residue_counts.items()):
            lines.append(f"- {name}: {count} atom(s)")
    if report.removed_metal_residue_counts:
        lines.append("")
        lines.append("Removed metal/ion residues:")
        for name, count in sorted(report.removed_metal_residue_counts.items()):
            lines.append(f"- {name}: {count} atom(s)")
    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in report.warnings:
            lines.append(f"- {warning}")
    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_json, report_txt


def _normalize_altloc(line: str, default_altloc: str = "A") -> tuple[str | None, bool]:
    if not line.startswith(("ATOM", "HETATM")) or len(line) <= 16:
        return line, False
    altloc = line[16]
    if altloc in {" ", default_altloc}:
        if altloc == default_altloc:
            return line[:16] + " " + line[17:], True
        return line, False
    return None, False


def _pdb_residue_sort_key(index_line: tuple[int, str]) -> tuple[str, int, str, str, int]:
    index, line = index_line
    chain = line[21:22].strip()
    resseq_text = line[22:26].strip()
    try:
        resseq = int(resseq_text)
    except ValueError:
        resseq = 0
    icode = line[26:27].strip()
    residue = line[17:20].strip()
    return (chain, resseq, icode, residue, index)


def group_pdb_atoms_by_residue(input_pdb: Path, output_pdb: Path) -> Path:
    lines = input_pdb.read_text(encoding="utf-8", errors="ignore").splitlines()
    atom_lines = [(idx, line) for idx, line in enumerate(lines) if line.startswith(("ATOM", "HETATM"))]
    other_lines = [line for line in lines if not line.startswith(("ATOM", "HETATM", "END"))]
    grouped = [line for _idx, line in sorted(atom_lines, key=_pdb_residue_sort_key)]
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text("\n".join([*other_lines, *grouped, "END"]) + "\n", encoding="utf-8")
    return output_pdb


def clean_receptor_pdb(
    input_pdb: Path,
    output_pdb: Path,
    *,
    remove_waters: bool = True,
    remove_non_protein_heteroatoms: bool = True,
    keep_metals: bool = True,
    default_altloc: str = "A",
) -> ReceptorCleanReport:
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    atoms_kept = waters_removed = waters_kept = hetero_removed = hetero_kept = 0
    metals_kept = metals_removed = altlocs_normalized = altlocs_dropped = 0
    source_atom_records = source_hetatm_records = 0
    chains: set[str] = set()
    all_residues: set[str] = set()
    protein_residues: set[str] = set()
    water_residues: set[str] = set()
    unknown_atom_residues: set[str] = set()
    hetero_residue_counts: Counter[str] = Counter()
    kept_hetero_residue_counts: Counter[str] = Counter()
    kept_metal_residue_counts: Counter[str] = Counter()
    removed_metal_residue_counts: Counter[str] = Counter()

    for line in input_pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        residue = line[17:20].strip().upper()
        residue_id = _residue_id(line)
        all_residues.add(residue_id)
        if line.startswith("ATOM"):
            source_atom_records += 1
            chains.add(line[21:22].strip() or "_")
            protein_residues.add(residue_id)
            if residue and residue not in STANDARD_PROTEIN_RESIDUES:
                unknown_atom_residues.add(residue)
        else:
            source_hetatm_records += 1

        normalized, changed = _normalize_altloc(line, default_altloc=default_altloc)
        if normalized is None:
            altlocs_dropped += 1
            continue
        line = normalized
        altlocs_normalized += int(changed)

        atom_name = line[12:16].strip().upper()
        element = (line[76:78].strip() or atom_name[:2].strip()).upper()

        if residue in WATER_NAMES:
            water_residues.add(residue_id)
            if remove_waters:
                waters_removed += 1
                continue
            kept.append(line)
            atoms_kept += 1
            waters_kept += 1
            continue
        if line.startswith("HETATM"):
            is_metal = residue in METAL_NAMES or element in METAL_NAMES or atom_name in METAL_NAMES
            if is_metal:
                if keep_metals:
                    kept.append(line)
                    atoms_kept += 1
                    metals_kept += 1
                    kept_metal_residue_counts[residue] += 1
                else:
                    metals_removed += 1
                    removed_metal_residue_counts[residue] += 1
                continue
            if remove_non_protein_heteroatoms:
                hetero_removed += 1
                hetero_residue_counts[residue] += 1
                continue
            kept.append(line)
            atoms_kept += 1
            hetero_kept += 1
            kept_hetero_residue_counts[residue] += 1
            continue

        kept.append(line)
        atoms_kept += 1

    if not kept:
        raise ValueError(f"No receptor atoms retained from {input_pdb}")
    warnings: list[str] = []
    if hetero_removed:
        warnings.append("Non-protein HETATM records were removed; check whether any cofactors should be retained.")
    if hetero_kept:
        warnings.append("Non-protein HETATM records were kept; ensure they are intended receptor components.")
    if metals_kept:
        warnings.append("Metal/ion HETATM records were kept.")
    if metals_removed:
        warnings.append("Metal/ion HETATM records were removed.")
    if waters_kept:
        warnings.append("Water HETATM records were kept.")
    if altlocs_dropped:
        warnings.append(f"Alternate locations other than {default_altloc!r} were dropped.")
    if unknown_atom_residues:
        warnings.append("Unknown/non-standard ATOM residues were detected.")
    if not chains:
        warnings.append("No chain identifiers were found in ATOM records.")

    output_pdb.write_text("\n".join(kept) + "\nEND\n", encoding="utf-8")
    return ReceptorCleanReport(
        source_path=input_pdb,
        cleaned_pdb=output_pdb,
        source_atom_records=source_atom_records,
        source_hetatm_records=source_hetatm_records,
        atoms_kept=atoms_kept,
        waters_removed=waters_removed,
        waters_kept=waters_kept,
        hetero_removed=hetero_removed,
        hetero_kept=hetero_kept,
        metals_kept=metals_kept,
        metals_removed=metals_removed,
        altlocs_normalized=altlocs_normalized,
        altlocs_dropped=altlocs_dropped,
        chains=tuple(sorted(chains)),
        residue_count=len(all_residues),
        protein_residue_count=len(protein_residues),
        water_residue_count=len(water_residues),
        hetero_residue_counts=dict(sorted(hetero_residue_counts.items())),
        kept_hetero_residue_counts=dict(sorted(kept_hetero_residue_counts.items())),
        kept_metal_residue_counts=dict(sorted(kept_metal_residue_counts.items())),
        removed_metal_residue_counts=dict(sorted(removed_metal_residue_counts.items())),
        unknown_atom_residues=tuple(sorted(unknown_atom_residues)),
        warnings=tuple(warnings),
    )


def prepare_receptor_pdbqt(input_pdb: Path, output_pdbqt: Path) -> Path:
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    output_base = output_pdbqt.with_suffix("")
    mk_prepare_receptor = script_binary("mk_prepare_receptor")
    cmd = [
        str(mk_prepare_receptor),
        "--read_pdb",
        str(input_pdb),
        "-o",
        str(output_base),
        "-p",
        str(output_pdbqt),
        "-a",
        "--default_altloc",
        "A",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not output_pdbqt.exists():
        raise RuntimeError(
            "Receptor PDBQT preparation failed.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return output_pdbqt


def add_hydrogens_with_obabel(input_pdb: Path, output_pdb: Path, *, ph: float = 7.4) -> Path:
    obabel = shutil.which("obabel")
    if obabel is None:
        raise RuntimeError("OpenBabel 'obabel' was not found on PATH; cannot add receptor hydrogens.")
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output_pdb.with_suffix(".raw.pdb")
    cmd = [obabel, "-ipdb", str(input_pdb), "-opdb", "-O", str(raw_output), "-h", "-p", str(ph)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not raw_output.exists():
        raise RuntimeError(f"OpenBabel hydrogen addition failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    group_pdb_atoms_by_residue(raw_output, output_pdb)
    return output_pdb


def repair_with_pdbfixer(input_pdb: Path, output_pdb: Path, *, ph: float | None = None) -> Path:
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError as exc:
        raise RuntimeError("PDBFixer/OpenMM is not installed; cannot repair receptor structure.") from exc
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    fixer = PDBFixer(filename=str(input_pdb))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    if ph is not None:
        fixer.addMissingHydrogens(pH=ph)
    with output_pdb.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle)
    return output_pdb


def minimize_with_openmm(input_pdb: Path, output_pdb: Path) -> Path:
    try:
        from openmm import LangevinIntegrator, unit
        from openmm.app import ForceField, PDBFile, Simulation
    except ImportError as exc:
        raise RuntimeError("OpenMM is not installed; cannot minimize receptor structure.") from exc
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    pdb = PDBFile(str(input_pdb))
    forcefield = ForceField("amber14-all.xml")
    system = forcefield.createSystem(pdb.topology)
    integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=200)
    positions = simulation.context.getState(getPositions=True).getPositions()
    with output_pdb.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(pdb.topology, positions, handle)
    return output_pdb


def prepare_receptor_input(
    input_pdb: Path,
    output_dir: Path,
    *,
    remove_waters: bool = True,
    remove_non_protein_heteroatoms: bool = True,
    keep_metals: bool = True,
    default_altloc: str = "A",
    add_hydrogens_ph: float | None = None,
    repair_with_pdbfixer_enabled: bool = False,
    minimize_with_openmm_enabled: bool = False,
) -> ReceptorPrepReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    options = ReceptorPrepOptions(
        remove_waters=remove_waters,
        remove_non_protein_heteroatoms=remove_non_protein_heteroatoms,
        keep_metals=keep_metals,
        default_altloc=default_altloc,
        add_hydrogens_ph=add_hydrogens_ph,
        repair_with_pdbfixer=repair_with_pdbfixer_enabled,
        minimize_with_openmm=minimize_with_openmm_enabled,
    )
    cleaned = output_dir / f"{input_pdb.stem}.cleaned.pdb"
    clean_report = clean_receptor_pdb(
        input_pdb,
        cleaned,
        remove_waters=remove_waters,
        remove_non_protein_heteroatoms=remove_non_protein_heteroatoms,
        keep_metals=keep_metals,
        default_altloc=default_altloc,
    )
    working_pdb = cleaned
    prep_steps: list[str] = ["clean"]
    if repair_with_pdbfixer_enabled:
        working_pdb = repair_with_pdbfixer(working_pdb, output_dir / f"{input_pdb.stem}.repaired.pdb", ph=add_hydrogens_ph)
        prep_steps.append("pdbfixer_repair")
    if add_hydrogens_ph is not None and not repair_with_pdbfixer_enabled:
        working_pdb = add_hydrogens_with_obabel(working_pdb, output_dir / f"{input_pdb.stem}.hydrogens.pdb", ph=add_hydrogens_ph)
        prep_steps.append(f"obabel_hydrogens_ph_{add_hydrogens_ph:g}")
    elif add_hydrogens_ph is not None and repair_with_pdbfixer_enabled:
        prep_steps.append(f"pdbfixer_hydrogens_ph_{add_hydrogens_ph:g}")
    if minimize_with_openmm_enabled:
        working_pdb = minimize_with_openmm(working_pdb, output_dir / f"{input_pdb.stem}.minimized.pdb")
        prep_steps.append("openmm_minimize")
    pdbqt = prepare_receptor_pdbqt(working_pdb, output_dir / f"{input_pdb.stem}.pdbqt")
    report_json, report_txt = write_receptor_reports(
        clean_report,
        options=options,
        prepared_pdb=working_pdb,
        pdbqt=pdbqt,
        prep_steps=tuple(prep_steps),
        output_dir=output_dir,
    )
    return ReceptorPrepReport(
        clean_report=clean_report,
        options=options,
        prepared_pdb=working_pdb,
        pdbqt=pdbqt,
        report_json=report_json,
        report_txt=report_txt,
        prep_steps=tuple(prep_steps),
    )
