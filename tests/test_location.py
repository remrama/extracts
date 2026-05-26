"""Tests for get_location / set_location."""

from __future__ import annotations

from pathlib import Path

import pooch
import pytest

import extracts


def test_get_location_default_is_pooch_os_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXTRACTS_DATA_DIR", raising=False)
    assert extracts.get_location() == pooch.os_cache("extracts")


def test_get_location_honors_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EXTRACTS_DATA_DIR", str(tmp_path))
    assert extracts.get_location() == tmp_path


def test_set_location_resolves_relative(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EXTRACTS_DATA_DIR", raising=False)
    extracts.set_location("./mycache")
    assert extracts.get_location() == (tmp_path / "mycache").resolve()


def test_set_location_expands_tilde(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXTRACTS_DATA_DIR", raising=False)
    extracts.set_location("~/somewhere")
    loc = extracts.get_location()
    assert "~" not in str(loc)
    assert loc.is_absolute()


def test_set_location_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("EXTRACTS_DATA_DIR", raising=False)
    extracts.set_location(tmp_path)
    assert extracts.get_location() == tmp_path.resolve()
