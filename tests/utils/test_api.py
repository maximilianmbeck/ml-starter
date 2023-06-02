"""Tests that the API file can be imported."""


def test_import() -> None:
    import ml.api

    assert ml.api is not None
