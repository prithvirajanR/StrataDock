import importlib.util
import io
import re
import zipfile
from pathlib import Path

from stratadock.ui.export import zip_run_outputs, zip_selected_hit_outputs, zip_selected_run_outputs
from stratadock.ui.export import csv_bytes
from stratadock.ui.state import normalize_uploaded_ligands, normalize_uploaded_receptors, selected_pocket_names


ROOT = Path(__file__).resolve().parents[1]


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _load_streamlit_module():
    spec = importlib.util.spec_from_file_location("stratadock_streamlit_app", ROOT / "streamlit_app.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pdb_line(record: str, serial: int, atom: str, residue: str, chain: str, seq: int, x: float, element: str) -> str:
    return (
        f"{record:<6}{serial:5d} {atom:^4} {residue:>3} {chain}{seq:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.00:6.2f}{20.00:6.2f}          {element:>2}"
    )


def test_streamlit_app_compiles():
    spec = importlib.util.spec_from_file_location("stratadock_streamlit_app", ROOT / "streamlit_app.py")
    assert spec is not None
    assert spec.loader is not None


def test_read_results_table_accepts_json_string():
    module = _load_streamlit_module()

    df = module.read_results_table('[{"ligand_name":"x","vina_score":-7.1}]')

    assert df.to_dict(orient="records") == [{"ligand_name": "x", "vina_score": -7.1}]


def test_write_upload_bytes_extracts_detected_receptor_reference_ligand(tmp_path, monkeypatch):
    module = _load_streamlit_module()
    pdb_text = "\n".join(
        [
            _pdb_line("ATOM", 1, "N", "ALA", "A", 1, 0.0, "N"),
            _pdb_line("HETATM", 2, "NA", "NA", "A", 10, 1.0, "NA"),
            _pdb_line("HETATM", 3, "C1", "LIG", "A", 12, 3.0, "C"),
            _pdb_line("HETATM", 4, "C2", "LIG", "A", 12, 4.0, "C"),
            _pdb_line("HETATM", 5, "C3", "LIG", "A", 12, 5.0, "C"),
            _pdb_line("HETATM", 6, "C4", "LIG", "A", 12, 6.0, "C"),
            _pdb_line("HETATM", 7, "C5", "LIG", "A", 12, 7.0, "C"),
            _pdb_line("HETATM", 8, "CL1", "LIG", "A", 12, 8.0, "CL"),
            "END",
        ]
    )
    state = AttrDict(
        receptor_uploads=[{"name": "complex.pdb", "bytes": pdb_text.encode("utf-8")}],
        receptor_name="complex.pdb",
        receptor_bytes=pdb_text.encode("utf-8"),
        ligand_name="ligands.smi",
        ligand_bytes=b"CCO ethanol\n",
        reference_source="Use ligand detected in receptor",
        reference_receptor_name="complex.pdb",
        reference_ligand_selector="LIG:A:12",
        reference_name=None,
        reference_bytes=None,
    )
    monkeypatch.setattr(module.st, "session_state", state)

    _receptors, _ligand, reference = module.write_upload_bytes(tmp_path)

    assert reference is not None
    text = reference.read_text(encoding="utf-8")
    assert "LIG" in text
    assert "ALA" not in text
    assert " NA " not in text


def test_persistent_streamlit_checkboxes_use_stable_keys():
    source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    stateful_checkbox_keys = [
        "fold_allow_network",
        "prep_remove_waters",
        "prep_remove_hetero",
        "prep_keep_metals",
        "prep_add_h",
        "prep_repair",
        "prep_minimize",
        "ligand_strip_salts",
        "ligand_neutralize",
        "ligand_use_ph",
        "resume",
        "viewer_show_interactions",
    ]

    for key in stateful_checkbox_keys:
        assert f'key="{key}"' in source
        assert not re.search(rf"st\.session_state\.{key}\s*=\s*[\w.]+\.checkbox\(", source)


def test_zip_run_outputs_contains_files(tmp_path):
    (tmp_path / "results.csv").write_text("ligand,score\nx,-1.0\n", encoding="utf-8")
    (tmp_path / "poses").mkdir()
    (tmp_path / "poses" / "x_pose.pdbqt").write_text("MODEL 1\nENDMDL\n", encoding="utf-8")

    blob = zip_run_outputs(tmp_path)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = set(archive.namelist())

    assert "results.csv" in names
    assert "poses/x_pose.pdbqt" in names
    assert "session_manifest.json" in names


def test_zip_selected_run_outputs_contains_only_selected_files(tmp_path):
    (tmp_path / "results.csv").write_text("ligand,score\nx,-1.0\n", encoding="utf-8")
    (tmp_path / "run_summary.txt").write_text("summary\n", encoding="utf-8")

    blob = zip_selected_run_outputs(tmp_path, ["results.csv"])
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = set(archive.namelist())

    assert names == {"results.csv", "session_manifest.json"}


def test_zip_selected_hit_outputs_collects_filtered_row_artifacts(tmp_path):
    (tmp_path / "results.csv").write_text("ligand,score\nx,-1.0\n", encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "poses").mkdir()
    (tmp_path / "complexes").mkdir()
    (tmp_path / "interactions").mkdir()
    (tmp_path / "poses" / "x_pose.pdbqt").write_text("MODEL 1\nENDMDL\n", encoding="utf-8")
    (tmp_path / "poses" / "y_pose.pdbqt").write_text("MODEL 1\nENDMDL\n", encoding="utf-8")
    (tmp_path / "complexes" / "x_complex.pdb").write_text("ATOM\n", encoding="utf-8")
    (tmp_path / "interactions" / "x.json").write_text("{}", encoding="utf-8")

    blob = zip_selected_hit_outputs(
        tmp_path,
        [
            {
                "pose_pdbqt": "poses/x_pose.pdbqt",
                "complex_pdb": "complexes/x_complex.pdb",
                "interactions_json": "interactions/x.json",
            }
        ],
        include_summary=True,
    )
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = set(archive.namelist())

    assert names == {
        "session_manifest.json",
        "run_manifest.json",
        "results.csv",
        "poses/x_pose.pdbqt",
        "complexes/x_complex.pdb",
        "interactions/x.json",
    }


def test_zip_selected_hit_outputs_accepts_absolute_paths_inside_run(tmp_path):
    (tmp_path / "poses").mkdir()
    pose = tmp_path / "poses" / "x_pose.pdbqt"
    pose.write_text("MODEL 1\nENDMDL\n", encoding="utf-8")

    blob = zip_selected_hit_outputs(tmp_path, [{"pose_pdbqt": str(pose)}], include_summary=False)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = set(archive.namelist())

    assert names == {"session_manifest.json", "poses/x_pose.pdbqt"}


def test_csv_bytes_exports_selected_rows():
    rows = [
        {"ligand_name": "a", "vina_score": -7.0},
        {"ligand_name": "b", "vina_score": -5.0},
    ]

    blob = csv_bytes(rows)

    assert b"ligand_name,vina_score" in blob
    assert b"a,-7.0" in blob


def test_normalize_uploaded_receptors_preserves_order():
    uploads = [
        {"name": "a.pdb", "bytes": b"ATOM a\n"},
        {"name": "b.pdb", "bytes": b"ATOM b\n"},
    ]

    assert normalize_uploaded_receptors(uploads) == uploads


def test_normalize_uploaded_ligands_keeps_single_file():
    upload = {"name": "ligand.sdf", "bytes": b"sdf-data"}

    assert normalize_uploaded_ligands([upload]) == upload


def test_normalize_uploaded_ligands_packages_multiple_files_as_zip():
    uploads = [
        {"name": "a.sdf", "bytes": b"sdf-a"},
        {"name": "b.sdf", "bytes": b"sdf-b"},
    ]

    packaged = normalize_uploaded_ligands(uploads)

    assert packaged is not None
    assert packaged["name"] == "stratadock_ligands.zip"
    with zipfile.ZipFile(io.BytesIO(packaged["bytes"])) as archive:
        assert archive.read("a.sdf") == b"sdf-a"
        assert archive.read("b.sdf") == b"sdf-b"


def test_selected_pocket_names_uses_rank_selection_when_present():
    rows = [
        {"rank": 1, "name": "pocket_1"},
        {"rank": 2, "name": "pocket_2"},
        {"rank": 5, "name": "pocket_5"},
    ]

    assert selected_pocket_names(rows, [2, 5]) == {"pocket_2", "pocket_5"}
    assert selected_pocket_names(rows, []) == {"pocket_1", "pocket_2", "pocket_5"}
