from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


_BASIC_KEYS = (
    "molecular_weight",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "qed",
    "lipinski_failures",
)
_EXTENDED_KEYS = (
    *_BASIC_KEYS,
    "rotatable_bonds",
    "heavy_atom_count",
    "aromatic_rings",
    "formal_charge",
    "fraction_csp3",
    "molar_refractivity",
    "rule_of_five_pass",
    "rule_of_five_classification",
    "veber_pass",
    "bbb_penetration",
    "herg_risk",
    "hepatotoxicity_risk",
    "mutagenicity_risk",
    "pains_alert_count",
    "brenk_alert_count",
    "structural_alert_count",
)


@dataclass(frozen=True)
class BasicAdmet:
    molecular_weight: float
    logp: float
    tpsa: float
    hbd: int
    hba: int
    qed: float
    lipinski_failures: int
    rotatable_bonds: int
    heavy_atom_count: int
    aromatic_rings: int
    formal_charge: int
    fraction_csp3: float
    molar_refractivity: float | None
    rule_of_five_pass: bool
    rule_of_five_classification: str
    veber_pass: bool
    bbb_penetration: str
    herg_risk: str
    hepatotoxicity_risk: str
    mutagenicity_risk: str
    pains_alert_count: int | None
    brenk_alert_count: int | None
    structural_alert_count: int | None

    def as_dict(self, *, include_extended: bool = False) -> dict[str, float | int | bool | str | None]:
        basic: dict[str, float | int | bool | str | None] = {
            "molecular_weight": self.molecular_weight,
            "logp": self.logp,
            "tpsa": self.tpsa,
            "hbd": self.hbd,
            "hba": self.hba,
            "qed": self.qed,
            "lipinski_failures": self.lipinski_failures,
        }
        if not include_extended:
            return basic
        return {
            **basic,
            "rotatable_bonds": self.rotatable_bonds,
            "heavy_atom_count": self.heavy_atom_count,
            "aromatic_rings": self.aromatic_rings,
            "formal_charge": self.formal_charge,
            "fraction_csp3": self.fraction_csp3,
            "molar_refractivity": self.molar_refractivity,
            "rule_of_five_pass": self.rule_of_five_pass,
            "rule_of_five_classification": self.rule_of_five_classification,
            "veber_pass": self.veber_pass,
            "bbb_penetration": self.bbb_penetration,
            "herg_risk": self.herg_risk,
            "hepatotoxicity_risk": self.hepatotoxicity_risk,
            "mutagenicity_risk": self.mutagenicity_risk,
            "pains_alert_count": self.pains_alert_count,
            "brenk_alert_count": self.brenk_alert_count,
            "structural_alert_count": self.structural_alert_count,
        }


def compute_basic_admet(mol: Chem.Mol) -> BasicAdmet:
    mw = float(Descriptors.MolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    qed = float(QED.qed(mol))
    lipinski_failures = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    rotatable_bonds = int(Lipinski.NumRotatableBonds(mol))
    pains_alert_count = _filter_catalog_match_count(mol, "PAINS")
    brenk_alert_count = _filter_catalog_match_count(mol, "BRENK")
    structural_alert_count = None
    if pains_alert_count is not None and brenk_alert_count is not None:
        structural_alert_count = pains_alert_count + brenk_alert_count
    return BasicAdmet(
        molecular_weight=round(mw, 3),
        logp=round(logp, 3),
        tpsa=round(tpsa, 3),
        hbd=hbd,
        hba=hba,
        qed=round(qed, 3),
        lipinski_failures=lipinski_failures,
        rotatable_bonds=rotatable_bonds,
        heavy_atom_count=int(mol.GetNumHeavyAtoms()),
        aromatic_rings=int(Lipinski.NumAromaticRings(mol)),
        formal_charge=int(Chem.GetFormalCharge(mol)),
        fraction_csp3=round(float(rdMolDescriptors.CalcFractionCSP3(mol)), 3),
        molar_refractivity=_molar_refractivity(mol),
        rule_of_five_pass=lipinski_failures == 0,
        rule_of_five_classification="pass" if lipinski_failures == 0 else "fail",
        veber_pass=rotatable_bonds <= 10 and tpsa <= 140,
        bbb_penetration=_bbb_penetration(mw=mw, logp=logp, tpsa=tpsa, hbd=hbd),
        herg_risk=_herg_risk(mw=mw, logp=logp, aromatic_rings=int(Lipinski.NumAromaticRings(mol))),
        hepatotoxicity_risk=_hepatotoxicity_risk(logp=logp, alerts=structural_alert_count),
        mutagenicity_risk=_mutagenicity_risk(pains_alerts=pains_alert_count, brenk_alerts=brenk_alert_count),
        pains_alert_count=pains_alert_count,
        brenk_alert_count=brenk_alert_count,
        structural_alert_count=structural_alert_count,
    )


def compute_ligand_admet_batch(
    ligands: Path | Sequence[object],
    *,
    errors: Sequence[object] | None = None,
) -> list[dict[str, object]]:
    from stratadock.core.ligands import LigandLoadError, LigandRecord, load_ligand_records_with_errors

    if isinstance(ligands, Path):
        records, load_errors = load_ligand_records_with_errors(ligands)
    else:
        records = [record for record in ligands if isinstance(record, LigandRecord)]
        load_errors = [error for error in errors or [] if isinstance(error, LigandLoadError)]

    rows: list[dict[str, object]] = []
    for record in records:
        try:
            admet = compute_basic_admet(record.mol).as_dict(include_extended=True)
            error = None
        except Exception as exc:
            admet = _empty_admet_dict()
            error = str(exc)
        rows.append(
            {
                "name": record.name,
                "source_path": str(record.source_path),
                "source_index": record.source_index,
                "error": error,
                **admet,
            }
        )

    for error in load_errors:
        rows.append(
            {
                "name": f"record_{error.source_index}",
                "source_path": str(error.source_path),
                "source_index": error.source_index,
                "error": error.message,
                **_empty_admet_dict(),
            }
        )

    return sorted(rows, key=lambda row: int(row["source_index"]))


def _molar_refractivity(mol: Chem.Mol) -> float | None:
    try:
        return round(float(Crippen.MolMR(mol)), 3)
    except Exception:
        return None


def _bbb_penetration(*, mw: float, logp: float, tpsa: float, hbd: int) -> str:
    return "likely" if mw <= 450 and 1 <= logp <= 4 and tpsa <= 90 and hbd <= 3 else "unlikely"


def _herg_risk(*, mw: float, logp: float, aromatic_rings: int) -> str:
    if logp >= 4.5 and mw >= 350 and aromatic_rings >= 2:
        return "high"
    if logp >= 3.5 and mw >= 250 and aromatic_rings >= 1:
        return "medium"
    return "low"


def _hepatotoxicity_risk(*, logp: float, alerts: int | None) -> str:
    alert_count = alerts or 0
    if logp > 5 or alert_count >= 2:
        return "high"
    if logp > 3.5 or alert_count == 1:
        return "medium"
    return "low"


def _mutagenicity_risk(*, pains_alerts: int | None, brenk_alerts: int | None) -> str:
    alerts = (pains_alerts or 0) + (brenk_alerts or 0)
    if alerts >= 2:
        return "high"
    if alerts == 1:
        return "medium"
    return "low"


def _filter_catalog_match_count(mol: Chem.Mol, catalog_name: str) -> int | None:
    try:
        catalog = _filter_catalog(catalog_name)
        return int(len(catalog.GetMatches(mol)))
    except Exception:
        return None


@lru_cache(maxsize=None)
def _filter_catalog(catalog_name: str):
    from rdkit.Chem import FilterCatalog

    catalog_enum = getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, catalog_name)
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(catalog_enum)
    return FilterCatalog.FilterCatalog(params)


def _empty_admet_dict() -> dict[str, object]:
    return {key: None for key in _EXTENDED_KEYS}
