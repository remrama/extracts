"""Unit tests for the CacheParquet processor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from extracts._fetchers import CacheParquet


def _make_source_tsv(tmp_path: Path) -> Path:
    src = tmp_path / "source.tsv"
    src.write_text("id\tval\n1\t10\n2\t20\n", encoding="utf-8")
    return src


def _identity_build(source_path: Path) -> pd.DataFrame:
    return pd.read_table(source_path, index_col=0)


def test_first_call_parses_and_writes_parquet(tmp_path: Path) -> None:
    src = _make_source_tsv(tmp_path)
    processor = CacheParquet(_identity_build, "cached.parquet")
    out = processor(str(src), "download", pup=None)  # type: ignore[arg-type]

    cache_path = Path(out)
    assert cache_path == src.parent / "cached.parquet"
    assert cache_path.exists()

    df = pd.read_parquet(cache_path)
    assert list(df["val"]) == [10, 20]


def test_second_fetch_uses_cache_without_calling_build(tmp_path: Path) -> None:
    src = _make_source_tsv(tmp_path)
    build = Mock(side_effect=_identity_build)
    processor = CacheParquet(build, "cached.parquet")

    processor(str(src), "download", pup=None)  # type: ignore[arg-type]
    assert build.call_count == 1

    # Second call with action="fetch" — cache exists, build should NOT run again.
    processor(str(src), "fetch", pup=None)  # type: ignore[arg-type]
    assert build.call_count == 1


def test_update_action_rebuilds(tmp_path: Path) -> None:
    src = _make_source_tsv(tmp_path)
    build = Mock(side_effect=_identity_build)
    processor = CacheParquet(build, "cached.parquet")

    processor(str(src), "download", pup=None)  # type: ignore[arg-type]
    processor(str(src), "update", pup=None)  # type: ignore[arg-type]
    assert build.call_count == 2


def test_multiindex_columns_roundtrip(tmp_path: Path) -> None:
    """MultiIndex headers must survive the parquet round-trip."""
    src = tmp_path / "mi.tsv"
    src.write_text(
        "\tA\tA\tB\n\tx\ty\tz\n0\t1\t2\t3\n1\t4\t5\t6\n",
        encoding="utf-8",
    )

    def _build(p: Path) -> pd.DataFrame:
        return pd.read_table(p, index_col=0, header=[0, 1])

    out = CacheParquet(_build, "mi.parquet")(str(src), "download", pup=None)  # type: ignore[arg-type]
    df = pd.read_parquet(out)
    assert df.columns.nlevels == 2
    assert list(df.iloc[0]) == [1, 2, 3]
