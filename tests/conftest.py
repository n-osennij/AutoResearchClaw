# conftest.py — shared pytest fixtures for researchclaw tests


import pytest


@pytest.fixture(autouse=True)
def _isolate_literature_cache(tmp_path, monkeypatch):
    """Keep the on-disk literature cache out of the repo during tests."""
    from researchclaw.literature import cache

    monkeypatch.setattr(cache, "_DEFAULT_CACHE_DIR", tmp_path / "litcache")
