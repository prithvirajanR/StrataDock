from pathlib import Path

import pytest

from stratadock.core.docking import run_vina
from stratadock.core.models import DockingBox


def _dummy_path(name: str) -> Path:
    return Path(name)


def test_run_vina_rejects_invalid_docking_parameters(tmp_path):
    kwargs = {
        "project_root": tmp_path,
        "receptor_pdbqt": _dummy_path("receptor.pdbqt"),
        "ligand_pdbqt": _dummy_path("ligand.pdbqt"),
        "box": DockingBox(0, 0, 0, 20, 20, 20),
        "output_pdbqt": tmp_path / "pose.pdbqt",
    }

    with pytest.raises(ValueError, match="num_modes"):
        run_vina(**kwargs, num_modes=0)

    with pytest.raises(ValueError, match="energy_range"):
        run_vina(**kwargs, energy_range=0)

    with pytest.raises(ValueError, match="scoring"):
        run_vina(**kwargs, scoring="ad4")
