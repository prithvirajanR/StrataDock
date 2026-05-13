from stratadock.core.pdb import compute_box_from_pdb_ligand, extract_ligand_pdb, ligand_candidates_from_pdb, parse_ligand_selector


def _pdb_line(record: str, serial: int, atom: str, residue: str, chain: str, seq: int, x: float, element: str) -> str:
    return (
        f"{record:<6}{serial:5d} {atom:^4} {residue:>3} {chain}{seq:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.00:6.2f}{20.00:6.2f}          {element:>2}"
    )


def test_ligand_candidates_ignore_waters_ions_and_ligand_halogen_atoms():
    pdb_text = "\n".join(
        [
            _pdb_line("ATOM", 1, "N", "ALA", "A", 1, 0.0, "N"),
            _pdb_line("HETATM", 2, "NA", "NA", "A", 10, 1.0, "NA"),
            _pdb_line("HETATM", 3, "O", "HOH", "A", 11, 2.0, "O"),
            _pdb_line("HETATM", 4, "C1", "LIG", "A", 12, 3.0, "C"),
            _pdb_line("HETATM", 5, "C2", "LIG", "A", 12, 4.0, "C"),
            _pdb_line("HETATM", 6, "C3", "LIG", "A", 12, 5.0, "C"),
            _pdb_line("HETATM", 7, "C4", "LIG", "A", 12, 6.0, "C"),
            _pdb_line("HETATM", 8, "C5", "LIG", "A", 12, 7.0, "C"),
            _pdb_line("HETATM", 9, "CL1", "LIG", "A", 12, 8.0, "CL"),
            "END",
        ]
    )

    candidates = ligand_candidates_from_pdb(pdb_text)

    assert [candidate.selector for candidate in candidates] == ["LIG:A:12"]
    assert candidates[0].atom_count == 6


def test_extract_detected_ligand_can_define_reference_box():
    pdb_text = "\n".join(
        [
            _pdb_line("ATOM", 1, "N", "ALA", "A", 1, 0.0, "N"),
            _pdb_line("HETATM", 2, "C1", "LIG", "A", 12, 3.0, "C"),
            _pdb_line("HETATM", 3, "C2", "LIG", "A", 12, 7.0, "C"),
            "END",
        ]
    )

    ligand_pdb = extract_ligand_pdb(pdb_text, parse_ligand_selector("LIG:A:12"))
    box = compute_box_from_pdb_ligand(ligand_pdb, padding=4.0)

    assert "LIG" in ligand_pdb
    assert "ALA" not in ligand_pdb
    assert box.center_x == 5.0
    assert box.size_x == 8.0
