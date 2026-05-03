import pytest

from stratadock.ui.preview import ligand_preview


def test_ligand_preview_returns_png_and_stable_labels(tmp_path):
    ligand_file = tmp_path / "ligands.smi"
    ligand_file.write_text(
        "\n".join(
            [
                "CCO duplicate",
                "CC(=O)O duplicate",
                "c1ccccc1 benzene",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    preview = ligand_preview(ligand_file, max_mols=2)

    assert preview.png.startswith(b"\x89PNG\r\n\x1a\n")
    assert preview.labels == ["1: duplicate", "2: duplicate"]
    assert preview.total_mols == 3
    assert preview.shown_mols == 2
    assert preview.omitted_count == 1
    assert preview.errors == []


def test_ligand_preview_rejects_non_positive_max_mols(tmp_path):
    ligand_file = tmp_path / "ligands.smi"
    ligand_file.write_text("CCO ethanol\n", encoding="utf-8")

    with pytest.raises(ValueError, match="max_mols"):
        ligand_preview(ligand_file, max_mols=0)
