import json

from stratadock.core.pdb import parse_pdb_atoms
from stratadock.core.receptors import clean_receptor_pdb, group_pdb_atoms_by_residue, prepare_receptor_input


def _pdb_line(
    record: str,
    serial: int,
    atom: str,
    residue: str,
    chain: str,
    seq: int,
    element: str,
    *,
    altloc: str = " ",
) -> str:
    return (
        f"{record:<6}{serial:5d} {atom:^4}{altloc}{residue:>3} {chain}{seq:4d}    "
        f"{float(serial):8.3f}{0.0:8.3f}{0.0:8.3f}{1.00:6.2f}{20.00:6.2f}          {element:>2}"
    )


def _write_tiny_receptor(path):
    path.write_text(
        "\n".join(
            [
                _pdb_line("ATOM", 1, "N", "ALA", "A", 1, "N"),
                _pdb_line("ATOM", 2, "CA", "ALA", "A", 1, "C"),
                _pdb_line("HETATM", 3, "O", "HOH", "A", 2, "O"),
                _pdb_line("HETATM", 4, "C1", "FAD", "A", 3, "C"),
                _pdb_line("HETATM", 5, "ZN", "ZN", "A", 4, "ZN"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_clean_receptor_can_keep_or_remove_waters(tmp_path):
    receptor = tmp_path / "receptor.pdb"
    _write_tiny_receptor(receptor)

    removed_report = clean_receptor_pdb(receptor, tmp_path / "removed.pdb")
    kept_report = clean_receptor_pdb(receptor, tmp_path / "kept.pdb", remove_waters=False)

    removed_atoms = parse_pdb_atoms((tmp_path / "removed.pdb").read_text())
    kept_atoms = parse_pdb_atoms((tmp_path / "kept.pdb").read_text())
    assert all(atom.residue_name != "HOH" for atom in removed_atoms)
    assert any(atom.residue_name == "HOH" for atom in kept_atoms)
    assert removed_report.waters_removed == 1
    assert kept_report.waters_removed == 0
    assert kept_report.waters_kept == 1


def test_clean_receptor_can_keep_or_remove_non_protein_heteroatoms(tmp_path):
    receptor = tmp_path / "receptor.pdb"
    _write_tiny_receptor(receptor)

    removed_report = clean_receptor_pdb(receptor, tmp_path / "removed.pdb")
    kept_report = clean_receptor_pdb(
        receptor,
        tmp_path / "kept.pdb",
        remove_non_protein_heteroatoms=False,
    )

    removed_atoms = parse_pdb_atoms((tmp_path / "removed.pdb").read_text())
    kept_atoms = parse_pdb_atoms((tmp_path / "kept.pdb").read_text())
    assert all(atom.residue_name != "FAD" for atom in removed_atoms)
    assert any(atom.residue_name == "FAD" for atom in kept_atoms)
    assert removed_report.hetero_removed == 1
    assert kept_report.hetero_removed == 0
    assert kept_report.hetero_kept == 1
    assert kept_report.kept_hetero_residue_counts == {"FAD": 1}


def test_clean_receptor_can_keep_or_remove_metals(tmp_path):
    receptor = tmp_path / "receptor.pdb"
    _write_tiny_receptor(receptor)

    kept_report = clean_receptor_pdb(receptor, tmp_path / "kept.pdb")
    removed_report = clean_receptor_pdb(receptor, tmp_path / "removed.pdb", keep_metals=False)

    kept_atoms = parse_pdb_atoms((tmp_path / "kept.pdb").read_text())
    removed_atoms = parse_pdb_atoms((tmp_path / "removed.pdb").read_text())
    assert any(atom.residue_name == "ZN" for atom in kept_atoms)
    assert all(atom.residue_name != "ZN" for atom in removed_atoms)
    assert kept_report.metals_kept == 1
    assert removed_report.metals_kept == 0
    assert removed_report.metals_removed == 1


def test_clean_receptor_uses_requested_default_altloc(tmp_path):
    receptor = tmp_path / "altloc.pdb"
    receptor.write_text(
        "\n".join(
            [
                _pdb_line("ATOM", 1, "N", "ALA", "A", 1, "N"),
                _pdb_line("ATOM", 2, "CA", "ALA", "A", 1, "C", altloc="A"),
                _pdb_line("ATOM", 3, "CA", "ALA", "A", 1, "C", altloc="B"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = clean_receptor_pdb(receptor, tmp_path / "cleaned.pdb", default_altloc="B")
    atoms = parse_pdb_atoms((tmp_path / "cleaned.pdb").read_text())

    assert [atom.atom_name for atom in atoms].count("CA") == 1
    assert atoms[1].line[16] == " "
    assert float(atoms[1].x) == 3.0
    assert report.altlocs_normalized == 1
    assert report.altlocs_dropped == 1


def test_group_pdb_atoms_by_residue_makes_interrupted_residues_contiguous(tmp_path):
    raw = tmp_path / "raw.pdb"
    raw.write_text(
        "\n".join(
            [
                _pdb_line("ATOM", 1, "N", "ALA", "A", 1, "N"),
                _pdb_line("ATOM", 2, "N", "GLY", "A", 2, "N"),
                _pdb_line("ATOM", 3, "H", "ALA", "A", 1, "H"),
                _pdb_line("ATOM", 4, "H", "GLY", "A", 2, "H"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    grouped = group_pdb_atoms_by_residue(raw, tmp_path / "grouped.pdb")
    atoms = parse_pdb_atoms(grouped.read_text())

    assert [(atom.residue_name, atom.atom_name) for atom in atoms] == [
        ("ALA", "N"),
        ("ALA", "H"),
        ("GLY", "N"),
        ("GLY", "H"),
    ]


def test_receptor_report_json_reflects_prep_options_and_counts(tmp_path, monkeypatch):
    receptor = tmp_path / "receptor.pdb"
    _write_tiny_receptor(receptor)

    def fake_prepare_receptor_pdbqt(input_pdb, output_pdbqt):
        output_pdbqt.write_text("REMARK fake pdbqt\n", encoding="utf-8")
        return output_pdbqt

    monkeypatch.setattr(
        "stratadock.core.receptors.prepare_receptor_pdbqt",
        fake_prepare_receptor_pdbqt,
    )

    report = prepare_receptor_input(
        receptor,
        tmp_path / "out",
        remove_waters=False,
        remove_non_protein_heteroatoms=False,
        keep_metals=False,
        default_altloc="B",
    )

    data = json.loads(report.report_json.read_text())
    assert data["options"]["remove_waters"] is False
    assert data["options"]["remove_non_protein_heteroatoms"] is False
    assert data["options"]["keep_metals"] is False
    assert data["options"]["default_altloc"] == "B"
    assert data["waters_removed"] == 0
    assert data["waters_kept"] == 1
    assert data["hetero_removed"] == 0
    assert data["hetero_kept"] == 1
    assert data["metals_kept"] == 0
    assert data["metals_removed"] == 1
    assert data["kept_hetero_residue_counts"] == {"FAD": 1}
