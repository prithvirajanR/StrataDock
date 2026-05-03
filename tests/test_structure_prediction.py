import json

import pytest

from stratadock.core.structure_prediction import (
    PredictionConfig,
    RemotePredictionResponse,
    predict_structure_from_sequence,
    validate_amino_acid_sequence,
)


PDB_TEXT = """\
ATOM      1  N   MET A   1      11.104  13.207   9.447  1.00 92.00           N
ATOM      2  CA  MET A   1      12.560  13.150   9.447  1.00 88.00           C
TER
END
"""


def test_validate_amino_acid_sequence_accepts_fasta_and_plain_text():
    raw = ">example protein\nmqi fvk\nTLT\n"

    assert validate_amino_acid_sequence(raw) == "MQIFVKTLT"


@pytest.mark.parametrize("bad", ["", ">only header\n", "MZQ", "MQ1F", "M-QF"])
def test_validate_amino_acid_sequence_rejects_empty_or_noncanonical_residues(bad):
    with pytest.raises(ValueError):
        validate_amino_acid_sequence(bad)


def test_predict_structure_reuses_existing_pdb_without_remote_call(tmp_path):
    output_path = tmp_path / "cached.pdb"
    output_path.write_text(PDB_TEXT, encoding="utf-8")
    report_path = tmp_path / "cached_report.json"

    def unexpected_post(*args, **kwargs):
        raise AssertionError("remote endpoint should not be called for cache hits")

    result = predict_structure_from_sequence(
        "MQIFVKTLT",
        output_path,
        config=PredictionConfig(allow_network=True, report_path=report_path),
        post_fn=unexpected_post,
    )

    assert result.ok is True
    assert result.cached is True
    assert result.output_path == output_path
    assert json.loads(report_path.read_text(encoding="utf-8"))["cached"] is True


def test_predict_structure_writes_successful_remote_pdb_and_report(tmp_path):
    output_path = tmp_path / "predicted.pdb"
    report_path = tmp_path / "prediction.json"
    calls = []

    def fake_post(sequence, endpoint_url, timeout_seconds):
        calls.append((sequence, endpoint_url, timeout_seconds))
        return RemotePredictionResponse(status_code=200, text=PDB_TEXT)

    result = predict_structure_from_sequence(
        ">villin\nMQIFVKTLT",
        output_path,
        config=PredictionConfig(
            endpoint_url="https://example.test/fold",
            allow_network=True,
            timeout_seconds=12,
            report_path=report_path,
        ),
        post_fn=fake_post,
    )

    assert result.ok is True
    assert result.cached is False
    assert output_path.read_text(encoding="utf-8") == PDB_TEXT
    assert calls == [("MQIFVKTLT", "https://example.test/fold", 12)]
    assert result.report["mean_plddt"] == 90.0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert report["sequence_length"] == 9
    assert report["output_path"] == str(output_path)
    assert report["endpoint_url"] == "https://example.test/fold"


def test_predict_structure_returns_structured_error_for_remote_failure(tmp_path):
    output_path = tmp_path / "failed.pdb"
    report_path = tmp_path / "failed.json"

    def fake_post(sequence, endpoint_url, timeout_seconds):
        return RemotePredictionResponse(status_code=503, text="service unavailable")

    result = predict_structure_from_sequence(
        "MQIFVKTLT",
        output_path,
        config=PredictionConfig(allow_network=True, report_path=report_path),
        post_fn=fake_post,
    )

    assert result.ok is False
    assert result.status == "remote_error"
    assert "503" in result.error
    assert "service unavailable" in result.error
    assert not output_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["status"] == "remote_error"
    assert report["error"] == result.error


def test_predict_structure_reports_network_disabled_without_crashing(tmp_path):
    result = predict_structure_from_sequence(
        "MQIFVKTLT",
        tmp_path / "disabled.pdb",
        config=PredictionConfig(allow_network=False),
    )

    assert result.ok is False
    assert result.status == "network_disabled"
    assert "Network access is disabled" in result.error
