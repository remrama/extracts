"""Regenerate ``docs/datasets/<name>.md`` from each fetcher's docstring.

Run after adding, renaming, or editing a dataset:

    uv run python docs/scripts/gen_dataset_pages.py

Then ``git status`` will surface the diff. Commit it alongside the source change.

The script extracts these sections from each fetcher's docstring (canonical
template defined in AGENTS.md): the summary line, the Citation block, the
Table captions list, and the optional Notes block. It emits a myst-nb-flavored
markdown file with executable preview cells.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from textwrap import dedent

import extracts
from extracts._common import DATASETS, DOI_PREFIX

DOCS_DATASETS = Path(__file__).resolve().parents[1] / "datasets"


def _split_sections(doc: str) -> dict[str, str]:
    """Split a numpy-style docstring into a {section_name: body} mapping.

    The summary line is stored under the key ``"summary"``. Body text appearing
    before any section header goes under ``"intro"``. Section headers are
    detected by an underline-of-dashes line, the standard numpy convention.
    """
    # Drop the first/last triple-quote whitespace and dedent.
    text = dedent(doc).strip("\n")

    sections: dict[str, str] = {}
    lines = text.splitlines()

    # First line is the summary.
    sections["summary"] = lines[0].strip()
    body = lines[1:]

    # Walk through; whenever we see a header (line followed by `---` of same length),
    # start a new section.
    cur_name = "intro"
    cur_body: list[str] = []
    i = 0
    while i < len(body):
        line = body[i]
        next_line = body[i + 1] if i + 1 < len(body) else ""
        if (
            next_line
            and set(next_line.strip()) == {"-"}
            and len(next_line.strip()) >= 3
            and len(line.strip()) > 0
            and line.strip() == line.lstrip()  # no leading indent
        ):
            # Flush previous section.
            sections[cur_name] = "\n".join(cur_body).strip("\n")
            cur_name = line.strip()
            cur_body = []
            i += 2
            continue
        cur_body.append(line)
        i += 1
    sections[cur_name] = "\n".join(cur_body).strip("\n")
    return sections


def _format_citation(raw: str) -> str:
    """Turn a docstring Citation block into well-formed markdown.

    Converts reST inline link syntax to markdown links and preserves the
    paragraph structure (blank lines) of the source block.
    """
    # ``doi:`label <url>`_``           -> ``doi: [label](url)``
    out = re.sub(r"doi:`([^<]+?)\s*<([^>]+)>`_", r"doi: [\1](\2)", raw)
    # ``url:`<url>`_``                  -> ``url: <url>``
    out = re.sub(r"url:`<([^>]+)>`_", r"url: <\1>", out)
    # Generic reST link ``\`text <url>\`_``  -> ``[text](url)``
    # Use a non-greedy match across newlines for text that wraps.
    out = re.sub(r"`([^`]+?)\s*<([^>]+)>`_", r"[\1](\2)", out, flags=re.DOTALL)
    # Strip a trailing comma on each non-empty line for readability, but
    # preserve blank lines so paragraphs render separately in markdown.
    cleaned_lines = []
    for line in out.splitlines():
        stripped = line.rstrip().rstrip(",")
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines).strip("\n")


def _format_table_captions(raw: str) -> str:
    """Pass through the bullet list verbatim (already in markdown-compatible shape)."""
    return raw.strip()


def _format_notes(raw: str) -> str:
    return raw.strip()


def _render_page(name: str) -> str:
    fetcher = getattr(extracts, f"fetch_{name}")
    sections = _split_sections(fetcher.__doc__ or "")

    latest_id = DATASETS[name]["latest"]
    zenodo_doi = f"{DOI_PREFIX}{latest_id}"

    citation = _format_citation(sections.get("Citation", ""))
    captions = _format_table_captions(sections.get("Table captions", ""))
    notes = sections.get("Notes", "")
    intro = sections.get("intro", "").strip()

    parts = [
        "---",
        "file_format: mystnb",
        "kernelspec:",
        "  name: python3",
        "  display_name: Python 3",
        "---",
        "",
        f"# {name}",
        "",
        citation,
        "",
        f"Zenodo deposit: [{zenodo_doi}](https://doi.org/{zenodo_doi})",
        "",
    ]
    if intro:
        parts.extend([intro, ""])

    parts.extend(["## Available tables", "", captions, ""])

    parts.extend(
        [
            "## Preview",
            "",
            "```{code-cell} python",
            "import extracts",
            "",
            f'df = extracts.fetch_{name}("table1")',
            "df.head()",
            "```",
            "",
            "```{code-cell} python",
            "df.shape",
            "```",
            "",
            "## Versions",
            "",
            "```{code-cell} python",
            "from extracts._common import DATASETS",
            "",
            f'DATASETS["{name}"]',
            "```",
            "",
        ]
    )

    if notes:
        parts.extend(["## Notes", "", _format_notes(notes), ""])

    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    DOCS_DATASETS.mkdir(exist_ok=True)
    written = []
    for name in extracts.list_available_datasets():
        fetcher = getattr(extracts, f"fetch_{name}", None)
        if fetcher is None or not inspect.isfunction(fetcher):
            continue
        page = _render_page(name)
        out = DOCS_DATASETS / f"{name}.md"
        out.write_text(page, encoding="utf-8")
        written.append(out.name)
    print(f"Wrote {len(written)} dataset pages: {written}")


if __name__ == "__main__":
    main()
