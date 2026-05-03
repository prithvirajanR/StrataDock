import pytest

from stratadock.core.pdb_download import validate_pdb_id


def test_validate_pdb_id_accepts_standard_ids():
    assert validate_pdb_id("3PTB") == "3ptb"


@pytest.mark.parametrize("bad", ["PTB", "ABCDE", "!!!!", "A123"])
def test_validate_pdb_id_rejects_invalid_ids(bad):
    with pytest.raises(ValueError):
        validate_pdb_id(bad)
