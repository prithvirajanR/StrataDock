from pathlib import Path
import subprocess

import pytest

from stratadock.core.models import DockingBox


def test_locate_gnina_prefers_explicit_executable(tmp_path):
    from stratadock.core.gnina import locate_gnina

    executable = tmp_path / "gnina-custom"
    executable.write_text("")

    assert locate_gnina(executable=executable) == executable


def test_locate_gnina_falls_back_to_path(monkeypatch):
    from stratadock.core import gnina

    monkeypatch.setattr(gnina.shutil, "which", lambda name: "/usr/bin/gnina" if name == "gnina" else None)

    assert gnina.locate_gnina() == Path("/usr/bin/gnina")


def test_build_gnina_command_includes_box_and_gpu_options():
    from stratadock.core.gnina import build_gnina_command

    command = build_gnina_command(
        executable="gnina",
        receptor=Path("receptor.pdbqt"),
        ligand=Path("ligand.sdf"),
        box=DockingBox(1.0, 2.0, 3.0, 20.0, 21.0, 22.0),
        output=Path("pose.sdf"),
        exhaustiveness=16,
        num_modes=5,
        seed=123,
        device=1,
    )

    assert command == [
        "gnina",
        "--receptor",
        "receptor.pdbqt",
        "--ligand",
        "ligand.sdf",
        "--center_x",
        "1.0",
        "--center_y",
        "2.0",
        "--center_z",
        "3.0",
        "--size_x",
        "20.0",
        "--size_y",
        "21.0",
        "--size_z",
        "22.0",
        "--exhaustiveness",
        "16",
        "--num_modes",
        "5",
        "--seed",
        "123",
        "--device",
        "1",
        "--out",
        "pose.sdf",
    ]


def test_build_gnina_command_validates_but_omits_vina_only_energy_range():
    from stratadock.core.gnina import build_gnina_command

    command = build_gnina_command(
        executable="gnina",
        receptor=Path("receptor.pdbqt"),
        ligand=Path("ligand.sdf"),
        box=DockingBox(1, 2, 3, 4, 5, 6),
        output=Path("pose.sdf"),
        energy_range=7.5,
    )

    assert "--energy_range" not in command

    with pytest.raises(ValueError, match="energy_range"):
        build_gnina_command(
            executable="gnina",
            receptor=Path("receptor.pdbqt"),
            ligand=Path("ligand.sdf"),
            box=DockingBox(1, 2, 3, 4, 5, 6),
            output=Path("pose.sdf"),
            energy_range=0,
        )


def test_build_gnina_command_uses_no_gpu_flag_instead_of_device():
    from stratadock.core.gnina import build_gnina_command

    command = build_gnina_command(
        executable="gnina",
        receptor=Path("receptor.pdbqt"),
        ligand=Path("ligand.sdf"),
        box=DockingBox(1, 2, 3, 4, 5, 6),
        output=Path("pose.sdf"),
        device=2,
        cpu_only=True,
    )

    assert "--no_gpu" in command
    assert "--cpu" not in command
    assert "--device" not in command


def test_gnina_command_uses_cpu_only_for_amd_hardware_recommendation(monkeypatch):
    from stratadock.core import hardware
    from stratadock.core.gnina import build_gnina_command

    monkeypatch.setattr(
        hardware.shutil,
        "which",
        lambda name: "/opt/rocm/bin/rocm-smi" if name == "rocm-smi" else None,
    )
    monkeypatch.setattr(
        hardware.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="GPU[0] : Card series: AMD Radeon RX 7900 XTX\n",
            stderr="",
        ),
    )

    summary = hardware.detect_hardware()
    command = build_gnina_command(
        executable="gnina",
        receptor=Path("receptor.pdbqt"),
        ligand=Path("ligand.sdf"),
        box=DockingBox(1, 2, 3, 4, 5, 6),
        output=Path("pose.sdf"),
        device=0,
        cpu_only=summary.recommended_backend == "cpu",
    )

    assert summary.amd.available is True
    assert summary.recommended_backend == "cpu"
    assert "--no_gpu" in command
    assert "--device" not in command


def test_parse_gnina_scores_from_stdout_table():
    from stratadock.core.gnina import parse_gnina_scores

    scores = parse_gnina_scores(
        """
        mode | affinity | CNNscore | CNNaffinity
        -----+----------+----------+------------
           1    -8.40      0.8123       7.10
           2    -7.20      0.6123       6.30
        """
    )

    assert [score.affinity for score in scores] == [-8.40, -7.20]
    assert [score.cnn_score for score in scores] == [0.8123, 0.6123]
    assert [score.cnn_affinity for score in scores] == [7.10, 6.30]


def test_parse_gnina_scores_from_sdf_tags():
    from stratadock.core.gnina import parse_gnina_scores

    scores = parse_gnina_scores(
        """
        > <minimizedAffinity>
        -9.25

        > <CNNscore>
        0.923

        > <CNNaffinity>
        7.52
        """
    )

    assert len(scores) == 1
    assert scores[0].affinity == -9.25
    assert scores[0].cnn_score == 0.923
    assert scores[0].cnn_affinity == 7.52


def test_parse_gnina_scores_prefers_pose_remarks_over_table_ruler():
    from stratadock.core.gnina import parse_gnina_scores

    scores = parse_gnina_scores(
        """
        1 10 20 30
        MODEL 1
        REMARK minimizedAffinity -11.6793633
        REMARK CNNscore 0.935761213
        REMARK CNNaffinity 8.57409286
        """
    )

    assert len(scores) == 1
    assert scores[0].affinity == -11.6793633
    assert scores[0].cnn_score == 0.935761213
    assert scores[0].cnn_affinity == 8.57409286


def test_parse_gnina_scores_from_multiple_pose_remarks():
    from stratadock.core.gnina import parse_gnina_scores

    scores = parse_gnina_scores(
        """
        MODEL 1
        REMARK minimizedAffinity -9.0
        REMARK CNNscore 0.90
        REMARK CNNaffinity 7.1
        ENDMDL
        MODEL 2
        REMARK minimizedAffinity -8.0
        REMARK CNNscore 0.80
        REMARK CNNaffinity 6.2
        ENDMDL
        """
    )

    assert [score.mode for score in scores] == [1, 2]
    assert [score.affinity for score in scores] == [-9.0, -8.0]
    assert [score.cnn_score for score in scores] == [0.90, 0.80]


def test_run_gnina_uses_mocked_subprocess_and_parses_scores(tmp_path, monkeypatch):
    from stratadock.core import gnina

    executable = tmp_path / "gnina"
    executable.write_text("")
    receptor = tmp_path / "receptor.pdbqt"
    ligand = tmp_path / "ligand.sdf"
    output = tmp_path / "out" / "pose.sdf"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("> <CNNscore>\n0.88\n\n> <CNNaffinity>\n6.7\n")
        return subprocess.CompletedProcess(command, 0, stdout="1 -8.1 0.88 6.7\n", stderr="")

    monkeypatch.setattr(gnina.subprocess, "run", fake_run)

    result = gnina.run_gnina(
        executable=executable,
        receptor=receptor,
        ligand=ligand,
        box=DockingBox(0, 0, 0, 10, 10, 10),
        output=output,
        exhaustiveness=4,
        num_modes=1,
        seed=99,
        cpu_only=True,
    )

    assert calls
    command, kwargs = calls[0]
    assert command[0] == str(executable)
    assert "--no_gpu" in command
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert result.score == -8.1
    assert result.scores[0].cnn_score == 0.88
    assert result.scores[0].cnn_affinity == 6.7


def test_run_gnina_raises_on_process_failure(tmp_path, monkeypatch):
    from stratadock.core import gnina

    executable = tmp_path / "gnina"
    executable.write_text("")

    monkeypatch.setattr(
        gnina.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 2, stdout="", stderr="bad input"),
    )

    with pytest.raises(RuntimeError, match="GNINA failed with code 2"):
        gnina.run_gnina(
            executable=executable,
            receptor=tmp_path / "receptor.pdbqt",
            ligand=tmp_path / "ligand.sdf",
            box=DockingBox(0, 0, 0, 10, 10, 10),
            output=tmp_path / "pose.sdf",
        )
