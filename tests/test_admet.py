from rdkit import Chem

from stratadock.core.admet import compute_basic_admet, compute_ligand_admet_batch
from stratadock.core.ligands import load_ligand_records_with_errors


def test_basic_admet_for_aspirin_is_in_expected_range():
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    admet = compute_basic_admet(mol)

    assert 179 < admet.molecular_weight < 181
    assert 1.0 < admet.logp < 2.0
    assert 60 < admet.tpsa < 70
    assert admet.hbd == 1
    assert admet.hba == 3
    assert 0 <= admet.qed <= 1
    assert admet.lipinski_failures == 0


def test_expanded_admet_for_aspirin_keeps_basic_keys_and_adds_druglike_flags():
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    admet = compute_basic_admet(mol)
    payload = admet.as_dict()
    extended = admet.as_dict(include_extended=True)

    assert payload == {
        "molecular_weight": admet.molecular_weight,
        "logp": admet.logp,
        "tpsa": admet.tpsa,
        "hbd": admet.hbd,
        "hba": admet.hba,
        "qed": admet.qed,
        "lipinski_failures": admet.lipinski_failures,
    }

    assert admet.heavy_atom_count == 13
    assert admet.aromatic_rings == 1
    assert admet.formal_charge == 0
    assert admet.rotatable_bonds >= 2
    assert 40 < admet.molar_refractivity < 50
    assert admet.rule_of_five_pass is True
    assert extended["rule_of_five_classification"] == "pass"
    assert admet.veber_pass is True
    assert 0 <= admet.fraction_csp3 <= 1
    assert admet.bbb_penetration in {"likely", "unlikely"}
    assert admet.herg_risk in {"low", "medium", "high"}
    assert admet.hepatotoxicity_risk in {"low", "medium", "high"}
    assert admet.mutagenicity_risk in {"low", "medium", "high"}
    assert admet.pains_alert_count >= 0
    assert admet.brenk_alert_count >= 0


def test_expanded_admet_for_paracetamol_is_in_expected_range():
    mol = Chem.MolFromSmiles("CC(=O)NC1=CC=C(O)C=C1")
    admet = compute_basic_admet(mol)

    assert 150 < admet.molecular_weight < 152
    assert 1 <= admet.rotatable_bonds <= 2
    assert admet.heavy_atom_count == 11
    assert admet.aromatic_rings == 1
    assert admet.formal_charge == 0
    assert admet.rule_of_five_pass is True
    assert admet.veber_pass is True


def test_rule_flags_fail_for_flexible_hydrophobic_ligand():
    mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCCCCCCCCCCCC")
    admet = compute_basic_admet(mol)

    assert admet.lipinski_failures >= 1
    assert admet.rule_of_five_pass is False
    assert admet.rule_of_five_classification == "fail"
    assert admet.rotatable_bonds > 10
    assert admet.veber_pass is False


def test_compute_ligand_admet_batch_includes_records_and_load_errors(tmp_path):
    ligand_file = tmp_path / "ligands.smi"
    ligand_file.write_text(
        "\n".join(
            [
                "CC(=O)OC1=CC=CC=C1C(=O)O aspirin",
                "C1CC bad_ring",
                "CC(=O)NC1=CC=C(O)C=C1 paracetamol",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = compute_ligand_admet_batch(ligand_file)

    assert [row["source_index"] for row in rows] == [1, 2, 3]
    assert rows[0]["name"] == "aspirin"
    assert rows[0]["error"] is None
    assert rows[0]["heavy_atom_count"] == 13
    assert rows[0]["bbb_penetration"] in {"likely", "unlikely"}
    assert rows[0]["herg_risk"] in {"low", "medium", "high"}
    assert rows[1]["name"] == "record_2"
    assert "Invalid SMILES" in rows[1]["error"]
    assert rows[1]["molecular_weight"] is None
    assert rows[2]["name"] == "paracetamol"
    assert rows[2]["rule_of_five_pass"] is True


def test_compute_ligand_admet_batch_accepts_loaded_records_and_errors(tmp_path):
    ligand_file = tmp_path / "ligands.smi"
    ligand_file.write_text("CCO ethanol\nC1CC bad_ring\n", encoding="utf-8")
    records, errors = load_ligand_records_with_errors(ligand_file)

    rows = compute_ligand_admet_batch(records, errors=errors)

    assert [row["source_index"] for row in rows] == [1, 2]
    assert rows[0]["name"] == "ethanol"
    assert rows[0]["error"] is None
    assert rows[1]["name"] == "record_2"
    assert "Invalid SMILES" in rows[1]["error"]
