from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DockingBox:
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float

    def as_dict(self) -> dict[str, float]:
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "center_z": self.center_z,
            "size_x": self.size_x,
            "size_y": self.size_y,
            "size_z": self.size_z,
        }


@dataclass(frozen=True)
class NamedDockingBox:
    name: str
    box: DockingBox
    source: str = "manual"
    rank: int | None = None
    score: float | None = None
    druggability_score: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "rank": self.rank,
            "score": self.score,
            "druggability_score": self.druggability_score,
            "box": self.box.as_dict(),
        }


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    pdb_id: str
    ligand_resname: str
    ligand_chain: str
    ligand_resseq: str
    complex_pdb: Path
    receptor_pdb: Path
    native_ligand_pdb: Path
    box: DockingBox
