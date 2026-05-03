import pandas as pd

from stratadock.ui.tables import (
    admet_columns,
    admet_summary,
    best_rows,
    filter_results,
    result_choice_labels,
    row_warning_flags,
    score_filter_bounds,
)


def test_result_choice_labels_disambiguate_repeated_ligands():
    df = pd.DataFrame(
        [
            {"ligand_name": "Ibuprofen_2", "pocket_name": "pocket_1", "vina_score": -6.1},
            {"ligand_name": "Ibuprofen_2", "pocket_name": "pocket_2", "vina_score": -5.9},
            {"ligand_name": "Ibuprofen_2", "pocket_name": "pocket_2", "vina_score": -5.9},
        ]
    )

    assert result_choice_labels(df) == [
        "Ibuprofen_2 | pocket_1 | -6.100",
        "Ibuprofen_2 | pocket_2 | -5.900",
        "Ibuprofen_2 | pocket_2 | -5.900 #2",
    ]


def test_admet_columns_keeps_only_available_columns_in_display_order():
    df = pd.DataFrame(
        [
            {
                "ligand_name": "aspirin",
                "qed": 0.55,
                "rotatable_bonds": 3,
                "unknown": "x",
            }
        ]
    )

    assert admet_columns(df) == ["ligand_name", "qed", "rotatable_bonds"]


def test_filter_results_applies_text_pocket_status_score_and_admet():
    df = pd.DataFrame(
        [
            {"ligand_name": "aspirin", "pocket_name": "p1", "docking_status": "success", "vina_score": -7.2, "rule_of_five_pass": True},
            {"ligand_name": "ethanol", "pocket_name": "p2", "docking_status": "failed", "vina_score": None, "rule_of_five_pass": True},
            {"ligand_name": "greasy", "pocket_name": "p1", "docking_status": "success", "vina_score": -4.0, "rule_of_five_pass": False},
        ]
    )

    filtered = filter_results(
        df,
        ligand_query="as",
        pockets=["p1"],
        statuses=["success"],
        max_score=-6.0,
        admet_mode="pass",
    )

    assert filtered["ligand_name"].tolist() == ["aspirin"]


def test_score_filter_bounds_returns_none_for_identical_scores():
    scores = pd.Series([-5.694, -5.694])

    assert score_filter_bounds(scores) is None


def test_score_filter_bounds_returns_range_for_varied_scores():
    scores = pd.Series([-8.0, -5.5, None])

    assert score_filter_bounds(scores) == (-8.0, -5.5)


def test_best_rows_picks_lowest_score_per_group():
    df = pd.DataFrame(
        [
            {"ligand_name": "a", "pocket_name": "p1", "vina_score": -5.0, "docking_status": "success"},
            {"ligand_name": "a", "pocket_name": "p2", "vina_score": -7.0, "docking_status": "success"},
            {"ligand_name": "b", "pocket_name": "p1", "vina_score": None, "docking_status": "failed"},
        ]
    )

    best = best_rows(df, "ligand_name")

    assert best[["ligand_name", "pocket_name", "vina_score"]].to_dict("records") == [
        {"ligand_name": "a", "pocket_name": "p2", "vina_score": -7.0}
    ]


def test_row_warning_flags_marks_problematic_rows():
    row = pd.Series(
        {
            "docking_status": "success",
            "vina_score": None,
            "rule_of_five_pass": False,
            "interactions_csv": None,
        }
    )

    assert row_warning_flags(row) == ["missing_score", "admet_rule_fail", "missing_interactions"]


def test_row_warning_flags_marks_non_negative_docking_scores():
    row = pd.Series(
        {
            "docking_status": "success",
            "vina_score": 3.2,
            "rule_of_five_pass": True,
            "veber_pass": True,
            "interactions_csv": "interactions.csv",
        }
    )

    assert row_warning_flags(row) == ["unfavorable_score"]


def test_admet_summary_counts_pass_warn_fail():
    df = pd.DataFrame(
        [
            {"rule_of_five_pass": True, "veber_pass": True, "structural_alert_count": 0},
            {"rule_of_five_pass": True, "veber_pass": True, "structural_alert_count": 2},
            {"rule_of_five_pass": False, "veber_pass": True, "structural_alert_count": 0},
        ]
    )

    assert admet_summary(df) == {"pass": 1, "warn": 1, "fail": 1, "total": 3}
