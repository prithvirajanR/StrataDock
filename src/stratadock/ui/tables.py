from __future__ import annotations

import pandas as pd


ADMET_DISPLAY_COLUMNS = [
    "ligand_name",
    "vina_score",
    "molecular_weight",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "qed",
    "lipinski_failures",
    "rotatable_bonds",
    "heavy_atom_count",
    "aromatic_rings",
    "formal_charge",
    "fraction_csp3",
    "molar_refractivity",
    "rule_of_five_pass",
    "veber_pass",
    "bbb_penetration",
    "herg_risk",
    "hepatotoxicity_risk",
    "mutagenicity_risk",
    "pains_alert_count",
    "brenk_alert_count",
    "structural_alert_count",
]


def result_choice_labels(df: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    seen: dict[str, int] = {}
    for _, row in df.reset_index(drop=True).iterrows():
        ligand = str(row.get("ligand_name", "ligand"))
        pocket = str(row.get("pocket_name", "pocket"))
        score = row.get("vina_score", None)
        score_text = "n/a" if pd.isna(score) else f"{float(score):.3f}"
        base = f"{ligand} | {pocket} | {score_text}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        labels.append(base if count == 1 else f"{base} #{count}")
    return labels


def admet_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in ADMET_DISPLAY_COLUMNS if column in df.columns]


def filter_results(
    df: pd.DataFrame,
    *,
    ligand_query: str = "",
    pockets: list[str] | None = None,
    statuses: list[str] | None = None,
    max_score: float | None = None,
    admet_mode: str = "all",
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    filtered = df.copy()
    if ligand_query and "ligand_name" in filtered:
        filtered = filtered[
            filtered["ligand_name"].fillna("").astype(str).str.contains(ligand_query, case=False, regex=False)
        ]
    if pockets and "pocket_name" in filtered:
        filtered = filtered[filtered["pocket_name"].isin(pockets)]
    if statuses and "docking_status" in filtered:
        filtered = filtered[filtered["docking_status"].isin(statuses)]
    if max_score is not None and "vina_score" in filtered:
        scores = pd.to_numeric(filtered["vina_score"], errors="coerce")
        filtered = filtered[scores.notna() & (scores <= max_score)]
    if admet_mode != "all" and "rule_of_five_pass" in filtered:
        passes = filtered["rule_of_five_pass"].fillna(False).astype(bool)
        filtered = filtered[passes] if admet_mode == "pass" else filtered[~passes]
    return filtered.reset_index(drop=True)


def score_filter_bounds(scores: pd.Series) -> tuple[float, float] | None:
    numeric = pd.to_numeric(scores, errors="coerce").dropna()
    if numeric.empty:
        return None
    low = float(numeric.min())
    high = float(numeric.max())
    if low == high:
        return None
    return low, high


def best_rows(df: pd.DataFrame, group_field: str) -> pd.DataFrame:
    required = {group_field, "vina_score", "docking_status"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=list(df.columns))
    successes = df[df["docking_status"].eq("success")].copy()
    successes["vina_score"] = pd.to_numeric(successes["vina_score"], errors="coerce")
    successes = successes[successes["vina_score"].notna()]
    if successes.empty:
        return pd.DataFrame(columns=list(df.columns))
    idx = successes.groupby(group_field)["vina_score"].idxmin()
    return successes.loc[idx].sort_values("vina_score").reset_index(drop=True)


def row_warning_flags(row: pd.Series) -> list[str]:
    flags: list[str] = []
    status = str(row.get("docking_status", ""))
    score = row.get("docking_score", row.get("vina_score"))
    if status != "success":
        flags.append("not_docked")
    if pd.isna(score):
        flags.append("missing_score")
    else:
        try:
            if float(score) >= 0:
                flags.append("unfavorable_score")
        except (TypeError, ValueError):
            pass
    if row.get("rule_of_five_pass") is False:
        flags.append("admet_rule_fail")
    if row.get("veber_pass") is False:
        flags.append("veber_fail")
    alerts = row.get("structural_alert_count")
    if not pd.isna(alerts) and alerts and int(alerts) > 0:
        flags.append("structural_alerts")
    interactions = row.get("interactions_csv")
    if status == "success" and (pd.isna(interactions) or not interactions):
        flags.append("missing_interactions")
    return flags


def add_warning_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    flagged = df.copy()
    flagged["warnings"] = ["; ".join(row_warning_flags(row)) for _, row in flagged.iterrows()]
    return flagged


def admet_summary(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {"pass": 0, "warn": 0, "fail": 0, "total": 0}
    total = len(df)
    rule_pass = df.get("rule_of_five_pass", pd.Series([False] * total)).fillna(False).astype(bool)
    veber_pass = df.get("veber_pass", pd.Series([False] * total)).fillna(False).astype(bool)
    alerts = pd.to_numeric(df.get("structural_alert_count", pd.Series([0] * total)), errors="coerce").fillna(0)
    fail = int((~rule_pass | ~veber_pass).sum())
    warn = int((rule_pass & veber_pass & (alerts > 0)).sum())
    passed = int((rule_pass & veber_pass & (alerts <= 0)).sum())
    return {"pass": passed, "warn": warn, "fail": fail, "total": total}
