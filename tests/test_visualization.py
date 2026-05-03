from pathlib import Path

from stratadock.core.visualization import write_3dmol_viewer_html
from stratadock.ui.preview import ligand_preview_png


def test_ligand_preview_png_returns_image_bytes(tmp_path):
    ligand_file = tmp_path / "ligands.smi"
    ligand_file.write_text("CCO ethanol\nCC(=O)O acetate\n", encoding="utf-8")

    blob = ligand_preview_png(ligand_file)

    assert blob.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(blob) > 1000


def test_write_3dmol_viewer_html_embeds_pdb(tmp_path):
    complex_pdb = tmp_path / "complex.pdb"
    complex_pdb.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
        "HETATM    2  C1  LIG L   1       1.000   0.000   0.000  1.00 20.00           C\n"
        "END\n",
        encoding="utf-8",
    )

    html_path = write_3dmol_viewer_html(complex_pdb, tmp_path / "viewer.html")
    text = html_path.read_text(encoding="utf-8")

    assert "3Dmol" in text
    assert "2.5.3/3Dmol-min.js" in text
    assert "container.style.width" in text
    assert "viewerAttempts += 1" in text
    assert "WebGL is disabled or unavailable" in text
    assert "ALA A" in text
    assert "LIG L" in text


def test_write_3dmol_viewer_html_accepts_viewer_options_and_metadata(tmp_path):
    complex_pdb = tmp_path / "complex.pdb"
    complex_pdb.write_text(
        "ATOM      1  N   ASP A  10       0.000   0.000   0.000  1.00 20.00           N\n"
        "HETATM    2  C1  LIG L   1       1.000   0.000   0.000  1.00 20.00           C\n"
        "END\n",
        encoding="utf-8",
    )

    html_path = write_3dmol_viewer_html(
        complex_pdb,
        tmp_path / "viewer.html",
        title="Pose Viewer",
        receptor_style="stick",
        ligand_style="sphere",
        surface_opacity=0.42,
        show_interactions=True,
        interactions=[{"type": "hbond", "ligand_atom": "N1", "receptor_residue": "ASP 10"}],
        title_metadata={"Ligand": "benzamidine", "Score": -7.1},
    )
    text = html_path.read_text(encoding="utf-8")

    assert "Pose Viewer" in text
    assert "stick" in text
    assert "sphere" in text
    assert "0.42" in text
    assert "ASP 10" in text
    assert "benzamidine" in text


def test_write_3dmol_viewer_html_accepts_surface_and_current_interaction_schema(tmp_path):
    complex_pdb = tmp_path / "complex.pdb"
    complex_pdb.write_text(
        "ATOM      1  N   ASP A  10       0.000   0.000   0.000  1.00 20.00           N\n"
        "HETATM    2  C1  LIG Z   1       1.000   0.000   0.000  1.00 20.00           C\n"
        "END\n",
        encoding="utf-8",
    )

    html_path = write_3dmol_viewer_html(
        complex_pdb,
        tmp_path / "viewer.html",
        receptor_style="surface",
        ligand_style="sphere",
        show_interactions=True,
        interactions=[
            {
                "interaction_type": "polar_contact",
                "residue_name": "ASP",
                "chain_id": "A",
                "residue_seq": "10",
                "ligand_atom_index": 2,
                "ligand_element": "O",
                "distance_angstrom": 2.91,
            }
        ],
    )
    text = html_path.read_text(encoding="utf-8")

    assert "surface" not in text.lower().split("receptor_style must be one of")
    assert "polar contact | O2 | ASP A 10 | 2.91 A" in text
    assert "Interactions (1)" in text
