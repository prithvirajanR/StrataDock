from pathlib import Path

import pytest

from stratadock.core.batch import run_batch_screen
from stratadock.core.boxes import box_from_ligand_pdb
from stratadock.core.models import NamedDockingBox
from stratadock.core.pockets import _run_fpocket, fpocket_available, fpocket_metadata_table, parse_fpocket_output, suggest_pockets_with_fpocket
from stratadock.tools.binaries import vina_binary


ROOT = Path(__file__).resolve().parents[1]


def _requires_vina() -> None:
    try:
        vina_binary(ROOT)
    except FileNotFoundError as exc:
        pytest.skip(f"AutoDock Vina binary is not installed for integration test: {exc}")


def _write_smiles(path: Path) -> Path:
    path.write_text("NC(=N)c1ccccc1 benzamidine\n", encoding="utf-8")
    return path


def test_parse_fpocket_output_reads_pocket_boxes_and_scores(tmp_path):
    case_dir = ROOT / "data" / "validation" / "trypsin_3ptb"
    native_pocket = (case_dir / "native_ligand.pdb").read_text()
    out_dir = tmp_path / "receptor_out"
    pockets_dir = out_dir / "pockets"
    pockets_dir.mkdir(parents=True)
    (pockets_dir / "pocket1_atm.pdb").write_text(native_pocket, encoding="utf-8")
    (pockets_dir / "pocket2_atm.pdb").write_text(native_pocket, encoding="utf-8")
    (out_dir / "receptor_info.txt").write_text(
        "\n".join(
            [
                "Pocket 1 :",
                "\tScore : 42.5",
                "\tPocket volume (Monte Carlo) : 123.4",
                "\tDruggability Score : 0.71",
                "Pocket 2 :",
                "\tScore : 12.0",
                "\tPocket volume (Monte Carlo) : 88.0",
                "\tDruggability Score : 0.33",
            ]
        ),
        encoding="utf-8",
    )

    pockets = parse_fpocket_output(out_dir, top_n=1)

    assert len(pockets) == 1
    assert pockets[0].name == "pocket_1"
    assert pockets[0].source == "fpocket"
    assert pockets[0].score == 42.5
    assert pockets[0].druggability_score == 0.71
    assert pockets[0].box.size_x > 0

    table = fpocket_metadata_table(out_dir, pockets)
    assert table[0]["name"] == "pocket_1"
    assert table[0]["score"] == 42.5
    assert table[0]["volume"] == 123.4
    assert table[0]["druggability_score"] == 0.71
    assert table[0]["size_x"] == pockets[0].box.size_x


def test_fpocket_metadata_table_falls_back_without_info_file(tmp_path):
    case_dir = ROOT / "data" / "validation" / "trypsin_3ptb"
    native_pocket = (case_dir / "native_ligand.pdb").read_text()
    out_dir = tmp_path / "receptor_out"
    pockets_dir = out_dir / "pockets"
    pockets_dir.mkdir(parents=True)
    (pockets_dir / "pocket1_atm.pdb").write_text(native_pocket, encoding="utf-8")

    pockets = parse_fpocket_output(out_dir)
    table = fpocket_metadata_table(out_dir, pockets)

    assert table == [
        {
            "rank": 1,
            "name": "pocket_1",
            "source": "fpocket",
            "score": None,
            "volume": None,
            "druggability_score": None,
            "center_x": pockets[0].box.center_x,
            "center_y": pockets[0].box.center_y,
            "center_z": pockets[0].box.center_z,
            "size_x": pockets[0].box.size_x,
            "size_y": pockets[0].box.size_y,
            "size_z": pockets[0].box.size_z,
        }
    ]


def test_suggest_pockets_with_fpocket_reuses_cached_receptor_output(tmp_path, monkeypatch):
    case_dir = ROOT / "data" / "validation" / "trypsin_3ptb"
    native_pocket = (case_dir / "native_ligand.pdb").read_text()
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\nEND\n", encoding="utf-8")
    calls = {"count": 0}

    def fake_run_fpocket(*, receptor_pdb, output_dir):
        calls["count"] += 1
        out_dir = output_dir / f"{receptor_pdb.stem}_out"
        pockets_dir = out_dir / "pockets"
        pockets_dir.mkdir(parents=True)
        (pockets_dir / "pocket1_atm.pdb").write_text(native_pocket, encoding="utf-8")
        (out_dir / "receptor_info.txt").write_text(
            "Pocket 1 :\n\tScore : 9.5\n\tPocket volume (Monte Carlo) : 55.0\n",
            encoding="utf-8",
        )
        return type("Run", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("stratadock.core.pockets.fpocket_available", lambda: True)
    monkeypatch.setattr("stratadock.core.pockets._run_fpocket", fake_run_fpocket)

    first = suggest_pockets_with_fpocket(receptor, tmp_path / "fpocket", top_n=1)
    second = suggest_pockets_with_fpocket(receptor, tmp_path / "fpocket", top_n=1)

    assert calls["count"] == 1
    assert first[0].score == 9.5
    assert second[0].score == 9.5
    assert (tmp_path / "fpocket" / "fpocket_cache.json").exists()
    assert (tmp_path / "fpocket" / "pockets_table.json").exists()


def test_run_fpocket_uses_absolute_receptor_path_for_relative_output(tmp_path, monkeypatch):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\nEND\n", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return type("Run", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("stratadock.core.pockets.shutil.which", lambda name: "/usr/bin/fpocket" if name == "fpocket" else None)
    monkeypatch.setattr("stratadock.core.pockets.subprocess.run", fake_run)

    _run_fpocket(receptor_pdb=Path("../receptor.pdb"), output_dir=Path("runs/fpocket"))

    args, kwargs = calls[0]
    assert args[:2] == ["/usr/bin/fpocket", "-f"]
    assert Path(args[2]).is_absolute()
    assert Path(args[2]).exists()
    assert kwargs["cwd"].is_absolute()


def test_fpocket_available_accepts_direct_bundled_binary(monkeypatch, tmp_path):
    binary = tmp_path / "fpocket"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr("stratadock.core.pockets.shutil.which", lambda name: None)
    monkeypatch.setattr("stratadock.core.pockets._bundled_fpocket_binary", lambda: binary)

    assert fpocket_available() is True


def test_batch_screen_can_dock_multiple_named_pockets(tmp_path):
    _requires_vina()
    case_dir = ROOT / "data" / "validation" / "trypsin_3ptb"
    box = box_from_ligand_pdb(case_dir / "native_ligand.pdb")
    ligands = _write_smiles(tmp_path / "one_ligand.smi")

    result = run_batch_screen(
        project_root=ROOT,
        receptor_pdb=case_dir / "3ptb_complex.pdb",
        ligand_file=ligands,
        box=box,
        boxes=[
            NamedDockingBox(name="pocket_a", box=box, source="test", rank=1),
            NamedDockingBox(name="pocket_b", box=box, source="test", rank=2),
        ],
        output_dir=tmp_path / "multi_pocket",
        exhaustiveness=1,
    )

    assert len(result.results) == 2
    assert {row.pocket_name for row in result.results} == {"pocket_a", "pocket_b"}
    assert sum(row.docking_status == "success" for row in result.results) == 2
    assert result.boxes_json.exists()
