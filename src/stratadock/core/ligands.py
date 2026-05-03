from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from stratadock.tools.binaries import script_binary


VALID_FORCE_FIELDS = {"MMFF94", "MMFF94s", "UFF"}


def _normalize_force_field(force_field: str) -> str:
    normalized = force_field.strip()
    upper = normalized.upper()
    if upper == "MMFF94":
        return "MMFF94"
    if upper == "MMFF94S":
        return "MMFF94s"
    if upper == "UFF":
        return "UFF"
    raise ValueError(f"Unsupported force field: {force_field}")


@dataclass(frozen=True)
class LigandPrepOptions:
    force_field: str = "MMFF94"
    protonation_ph: float | None = None
    strip_salts: bool = False
    neutralize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "force_field", _normalize_force_field(self.force_field))

    def as_dict(self) -> dict[str, object]:
        return {
            "force_field": self.force_field,
            "protonation_ph": self.protonation_ph,
            "strip_salts": self.strip_salts,
            "neutralize": self.neutralize,
        }


@dataclass(frozen=True)
class LigandPrepReport:
    name: str
    source_path: Path
    prepared_sdf: Path
    pdbqt: Path
    heavy_atoms: int
    embedding_used: bool
    force_field: str
    requested_force_field: str = "MMFF94"
    protonation_ph: float | None = None
    protonation_status: str = "disabled"
    salt_stripping_status: str = "disabled"
    neutralization_status: str = "disabled"
    options: LigandPrepOptions = field(default_factory=LigandPrepOptions)


@dataclass(frozen=True)
class LigandRecord:
    name: str
    mol: Chem.Mol
    source_path: Path
    source_index: int


@dataclass(frozen=True)
class LigandLoadError:
    source_path: Path
    source_index: int
    message: str


def load_sdf(path: Path) -> list[Chem.Mol]:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    return [mol for mol in supplier if mol is not None]


def load_smiles(path: Path) -> list[LigandRecord]:
    records: list[LigandRecord] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        smiles = parts[0]
        if _is_smiles_header(parts):
            continue
        name = " ".join(parts[1:]) if len(parts) > 1 else f"ligand_{idx}"
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES on line {idx}: {smiles}")
        mol.SetProp("_Name", name)
        records.append(LigandRecord(name=name, mol=mol, source_path=path, source_index=idx))
    if not records:
        raise ValueError(f"No SMILES records found in {path}")
    return records


def load_ligand_records(path: Path) -> list[LigandRecord]:
    records, errors = load_ligand_records_with_errors(path)
    if errors:
        raise ValueError(errors[0].message)
    if not records:
        raise ValueError(f"No molecules found in {path}")
    return records


def load_ligand_records_with_errors(path: Path) -> tuple[list[LigandRecord], list[LigandLoadError]]:
    suffix = path.suffix.lower()
    errors: list[LigandLoadError] = []
    if suffix == ".zip":
        return _load_zip_ligands(path)
    if suffix == ".sdf":
        records: list[LigandRecord] = []
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
        for idx, mol in enumerate(supplier, start=1):
            if mol is None:
                errors.append(LigandLoadError(path, idx, f"Invalid SDF molecule at record {idx}"))
                continue
            name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else f"{path.stem}_{idx}"
            records.append(
                LigandRecord(
                    name=name or f"{path.stem}_{idx}",
                    mol=mol,
                    source_path=path,
                    source_index=idx,
                )
            )
        return records, errors
    if suffix in {".smi", ".smiles", ".txt"}:
        records = []
        for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if _is_smiles_header(parts):
                continue
            smiles = parts[0]
            name = " ".join(parts[1:]) if len(parts) > 1 else f"ligand_{idx}"
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                errors.append(LigandLoadError(path, idx, f"Invalid SMILES on line {idx}: {smiles}"))
                continue
            mol.SetProp("_Name", name)
            records.append(LigandRecord(name=name, mol=mol, source_path=path, source_index=idx))
        return records, errors
    return [], [LigandLoadError(path, 0, f"Unsupported ligand file type: {path.suffix}")]


def _load_zip_ligands(path: Path) -> tuple[list[LigandRecord], list[LigandLoadError]]:
    records: list[LigandRecord] = []
    errors: list[LigandLoadError] = []
    supported = {".sdf", ".smi", ".smiles", ".txt"}
    source_index = 1
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        ligand_names = [name for name in names if Path(name).suffix.lower() in supported and not _is_metadata_member(name)]
        if not ligand_names:
            return [], [LigandLoadError(path, 0, "ZIP file contains no supported ligand files.")]
        for name in ligand_names:
            suffix = Path(name).suffix.lower()
            data = archive.read(name)
            if suffix == ".sdf":
                supplier = Chem.ForwardSDMolSupplier(io.BytesIO(data), removeHs=False, sanitize=True)
                for mol in supplier:
                    if mol is None:
                        errors.append(LigandLoadError(path, source_index, f"Invalid SDF molecule in ZIP member {name}"))
                        source_index += 1
                        continue
                    mol_name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else f"{Path(name).stem}_{source_index}"
                    records.append(
                        LigandRecord(
                            name=mol_name or f"{Path(name).stem}_{source_index}",
                            mol=mol,
                            source_path=path,
                            source_index=source_index,
                        )
                    )
                    source_index += 1
                continue

            text = data.decode("utf-8", errors="ignore")
            for line_no, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if _is_smiles_header(parts):
                    continue
                smiles = parts[0]
                mol_name = " ".join(parts[1:]) if len(parts) > 1 else f"{Path(name).stem}_{line_no}"
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    errors.append(LigandLoadError(path, source_index, f"Invalid SMILES in ZIP member {name} line {line_no}: {smiles}"))
                    source_index += 1
                    continue
                mol.SetProp("_Name", mol_name)
                records.append(LigandRecord(name=mol_name, mol=mol, source_path=path, source_index=source_index))
                source_index += 1
    return records, errors


def _is_smiles_header(parts: list[str]) -> bool:
    if not parts:
        return False
    first = parts[0].strip().lower()
    second = parts[1].strip().lower() if len(parts) > 1 else ""
    return first in {"smiles", "smile"} or (first in {"canonical_smiles", "isosmiles"} and second in {"name", "id", "identifier"})


def _is_metadata_member(name: str) -> bool:
    path = Path(name)
    lowered_parts = {part.lower() for part in path.parts}
    lowered_name = path.name.lower()
    if path.suffix.lower() != ".txt":
        return False
    return (
        lowered_name.startswith(("readme", "license", "notes"))
        or "notes" in lowered_parts
        or "docs" in lowered_parts
        or "documentation" in lowered_parts
    )


def write_prepared_sdf(
    mol: Chem.Mol,
    output_sdf: Path,
    *,
    embed_if_needed: bool = True,
    force_field: str = "MMFF94",
    seed: int = 42,
) -> tuple[Path, bool, str]:
    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    working = Chem.Mol(mol)
    embedded = False
    force_field = _normalize_force_field(force_field)
    if working.GetNumConformers() == 0:
        if not embed_if_needed:
            raise ValueError("Ligand has no 3D conformer.")
        working = Chem.AddHs(working)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(working, params) != 0:
            raise ValueError("RDKit failed to embed ligand in 3D.")
        embedded = True
    else:
        working = Chem.AddHs(working, addCoords=True)

    used_force_field = force_field
    if force_field.upper().startswith("MMFF"):
        props = AllChem.MMFFGetMoleculeProperties(working, mmffVariant=force_field)
        if props is not None:
            AllChem.MMFFOptimizeMolecule(working, mmffVariant=force_field, maxIters=500)
        else:
            used_force_field = "UFF"
            AllChem.UFFOptimizeMolecule(working, maxIters=500)
    elif force_field.upper() == "UFF":
        AllChem.UFFOptimizeMolecule(working, maxIters=500)
    else:
        raise ValueError(f"Unsupported force field: {force_field}")

    writer = Chem.SDWriter(str(output_sdf))
    writer.write(working)
    writer.close()
    return output_sdf, embedded, used_force_field


def _copy_mol_metadata(source: Chem.Mol, target: Chem.Mol) -> Chem.Mol:
    for name in source.GetPropNames(includePrivate=True, includeComputed=False):
        target.SetProp(name, source.GetProp(name))
    return target


def _strip_salts_if_requested(mol: Chem.Mol, *, enabled: bool) -> tuple[Chem.Mol, str]:
    if not enabled:
        return Chem.Mol(mol), "disabled"
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(fragments) <= 1:
        return Chem.Mol(mol), "single_fragment"
    largest = max(fragments, key=lambda fragment: (fragment.GetNumHeavyAtoms(), fragment.GetNumAtoms()))
    return _copy_mol_metadata(mol, Chem.Mol(largest)), "stripped"


def _neutralize_if_requested(mol: Chem.Mol, *, enabled: bool) -> tuple[Chem.Mol, str]:
    if not enabled:
        return Chem.Mol(mol), "disabled"
    working = Chem.RWMol(mol)
    initial_charge = Chem.GetFormalCharge(working)
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    for (atom_index,) in working.GetMol().GetSubstructMatches(pattern):
        atom = working.GetAtomWithIdx(atom_index)
        charge = atom.GetFormalCharge()
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(max(0, atom.GetTotalNumHs() - charge))
        atom.UpdatePropertyCache(strict=False)
    neutralized = working.GetMol()
    Chem.SanitizeMol(neutralized)
    neutralized = _copy_mol_metadata(mol, neutralized)
    return neutralized, "neutralized" if Chem.GetFormalCharge(neutralized) != initial_charge else "unchanged"


def _apply_ligand_cleanup_options(mol: Chem.Mol, options: LigandPrepOptions) -> tuple[Chem.Mol, str, str]:
    cleaned, salt_status = _strip_salts_if_requested(mol, enabled=options.strip_salts)
    cleaned, neutralization_status = _neutralize_if_requested(cleaned, enabled=options.neutralize)
    return cleaned, salt_status, neutralization_status


def _apply_obabel_protonation_if_requested(sdf_path: Path, *, ph: float | None) -> tuple[Path, str]:
    if ph is None:
        return sdf_path, "disabled"
    obabel = shutil.which("obabel")
    if obabel is None:
        return sdf_path, "obabel_unavailable"
    protonated_sdf = sdf_path.with_name(f"{sdf_path.stem}.obabel.sdf")
    cmd = [obabel, "-isdf", str(sdf_path), "-osdf", "-O", str(protonated_sdf), "-p", f"{ph:g}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not protonated_sdf.exists():
        raise RuntimeError(f"OpenBabel ligand protonation failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    protonated_sdf.replace(sdf_path)
    return sdf_path, f"obabel_ph_{ph:g}"


def first_ligand_from_path(path: Path) -> tuple[str, Chem.Mol]:
    record = load_ligand_records(path)[0]
    return record.name, record.mol


def prepare_ligand_pdbqt(input_sdf: Path, output_pdbqt: Path) -> Path:
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    mk_prepare_ligand = script_binary("mk_prepare_ligand")
    mols = load_sdf(input_sdf)
    if not mols:
        raise ValueError(f"No molecules found in {input_sdf}")

    mol = Chem.AddHs(mols[0], addCoords=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        hydrated_sdf = Path(tmp_dir) / "ligand_with_h.sdf"
        writer = Chem.SDWriter(str(hydrated_sdf))
        writer.write(mol)
        writer.close()

        cmd = [
            str(mk_prepare_ligand),
            "-i",
            str(hydrated_sdf),
            "-o",
            str(output_pdbqt),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not output_pdbqt.exists():
        raise RuntimeError(
            "Ligand PDBQT preparation failed.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return output_pdbqt


def prepare_ligand_input(
    input_path: Path,
    output_dir: Path,
    *,
    force_field: str = "MMFF94",
    options: LigandPrepOptions | None = None,
) -> LigandPrepReport:
    return prepare_ligand_record(load_ligand_records(input_path)[0], output_dir, force_field=force_field, options=options)


def prepare_ligand_record(
    record: LigandRecord,
    output_dir: Path,
    *,
    force_field: str = "MMFF94",
    options: LigandPrepOptions | None = None,
) -> LigandPrepReport:
    options = options or LigandPrepOptions(force_field=force_field)
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in record.name) or "ligand"
    safe_name = f"{record.source_index:05d}_{safe_name}"
    prepared_sdf = output_dir / f"{safe_name}.prepared.sdf"
    pdbqt = output_dir / f"{safe_name}.pdbqt"
    prepared_mol, salt_status, neutralization_status = _apply_ligand_cleanup_options(record.mol, options)
    sdf_path, embedded, used_force_field = write_prepared_sdf(
        prepared_mol,
        prepared_sdf,
        embed_if_needed=True,
        force_field=options.force_field,
    )
    sdf_path, protonation_status = _apply_obabel_protonation_if_requested(sdf_path, ph=options.protonation_ph)
    prepare_ligand_pdbqt(sdf_path, pdbqt)
    return LigandPrepReport(
        name=safe_name,
        source_path=record.source_path,
        prepared_sdf=sdf_path,
        pdbqt=pdbqt,
        heavy_atoms=sum(1 for atom in prepared_mol.GetAtoms() if atom.GetAtomicNum() > 1),
        embedding_used=embedded,
        force_field=used_force_field,
        requested_force_field=options.force_field,
        protonation_ph=options.protonation_ph,
        protonation_status=protonation_status,
        salt_stripping_status=salt_status,
        neutralization_status=neutralization_status,
        options=options,
    )
