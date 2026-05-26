"""Package-level smoke tests: version, public API surface."""

from __future__ import annotations

import extracts


def test_version_is_nonempty_string() -> None:
    assert isinstance(extracts.__version__, str)
    assert extracts.__version__


def test_all_names_are_importable() -> None:
    for name in extracts.__all__:
        assert hasattr(extracts, name), f"extracts.__all__ lists {name!r} but missing on module"


def test_public_callables_are_callable() -> None:
    for name in (
        "fetch_path",
        "set_location",
        "get_location",
        "list_available_datasets",
        "list_available_tables",
        "fetch_text",
        "fetch_reference",
    ):
        assert callable(getattr(extracts, name)), f"{name} should be callable"


def test_public_surface_matches_all() -> None:
    """Names exposed on the module (minus dunders) equal the declared __all__."""
    public = {n for n in dir(extracts) if not n.startswith("_")}
    declared = set(extracts.__all__) - {"__version__"}
    assert public == declared, (
        f"public surface drift: only-in-public={public - declared}, only-in-all={declared - public}"
    )
