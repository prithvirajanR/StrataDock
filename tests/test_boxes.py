import json
import math

import pytest

from pathlib import Path

from stratadock.core.boxes import box_from_center_size, box_from_residues_pdb, load_box_json, validate_box, write_box_json
from stratadock.core.models import DockingBox


ROOT = Path(__file__).resolve().parents[1]


def test_manual_box_from_center_size_validates_and_casts_values():
    box = box_from_center_size(center=["1.5", 2, 3.25], size=[18, "19.5", 20])

    assert box.as_dict() == {
        "center_x": 1.5,
        "center_y": 2.0,
        "center_z": 3.25,
        "size_x": 18.0,
        "size_y": 19.5,
        "size_z": 20.0,
    }


@pytest.mark.parametrize(
    "box, error",
    [
        (DockingBox(0, 0, 0, 0, 20, 20), "positive"),
        (DockingBox(0, 0, 0, 81, 20, 20), "suspiciously large"),
        (DockingBox(0, math.inf, 0, 20, 20, 20), "finite"),
    ],
)
def test_validate_box_rejects_bad_values(box, error):
    with pytest.raises(ValueError, match=error):
        validate_box(box)


def test_box_json_roundtrip(tmp_path):
    path = tmp_path / "box.json"
    box = DockingBox(10.0, 11.0, 12.0, 20.0, 21.0, 22.0)

    write_box_json(box, path)

    assert json.loads(path.read_text()) == box.as_dict()
    assert load_box_json(path) == box


def test_box_json_requires_all_fields(tmp_path):
    path = tmp_path / "box.json"
    path.write_text('{"center_x": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="missing required"):
        load_box_json(path)


def test_box_from_residue_selectors_matches_atoms():
    receptor = ROOT / "data" / "validation" / "trypsin_3ptb" / "receptor.pdb"

    box = box_from_residues_pdb(receptor, ["A:189", "A:190"])

    assert box.size_x > 0
    assert box.size_y > 0
    assert box.size_z > 0
