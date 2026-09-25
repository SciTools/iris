# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Pre-commit hook: validate the metadata every documentation page must carry.

Two requirements are checked, both of which are answerable from the page's
own reStructuredText source:

* every non-exempt page carries a ``.. readingtime::`` directive;
* every page under a Diataxis-typed directory carries exactly one
  correctly-configured sphinx-needs item.

The same rules are enforced during the documentation build, by the Sphinx
extensions in ``docs/src/sphinxext/``, which import this module rather than
reimplementing it.  That is the point of putting the logic here: a rule added
below is enforced in both places at once, and the two can never disagree about
whether a page is valid.  The build additionally covers the pages under
``generated/`` — written by autosummary and sphinx-gallery while the build
runs, so they have no source file for this hook to read.

Usage (the hook passes no filenames; the whole source tree is walked)::

    python .hooks/check_docs_page_metadata.py [docs-src-directory]

Exit codes:
    0 – all clear
    1 – one or more pages are missing required metadata
"""

import fnmatch
from pathlib import Path, PurePosixPath
import re
import sys

DOCS_SRC = Path(__file__).parent.parent / "docs" / "src"
"""The documentation source root, relative to this file."""

READINGTIME_EXCEPTIONS = {
    "index",
    "*/index",
    "voted_issues",
    "sg_execution_times",
    "copyright",
    "user_manual/section_indexes/*",
    "user_manual/reference/glossary",
    "developers_guide/contributing_getting_involved",
    "developers_guide/contributing_changes",
    "developers_guide/contributing_codebase_index",
    "developers_guide/contributing_documentation",
    "developers_guide/contributing_testing_index",
    "developers_guide/release_do_nothing",
    "developers_guide/gitwash/**",
    "generated/**",
    "whatsnew/**",
}
"""
Pages exempt from readingtime validation.

Supports glob patterns using standard wildcards.
"""

EXPECTED_TYPE_BY_PARENT = (
    ("generated/api", "z_reference"),
    ("generated/gallery", "how-to"),
    ("user_manual/tutorial", "tutorial"),
    ("user_manual/explanation", "explanation"),
    ("user_manual/how_to", "how-to"),
    ("user_manual/reference", "z_reference"),
)
"""
The sphinx-needs item type each Diataxis directory is expected to contain.

Ordered; the first matching parent wins.  The type names are the values of
``Diataxis`` in ``docs/src/sphinxext/user_manual_directives.py``, which is
where the corresponding directives are configured.
"""

MAX_NEED_LINE = 25
"""Items must start within this many lines, so links land at the page top."""

GENERATED_PARENT = PurePosixPath("generated")
"""Pages below here are written during the build; this hook cannot see them."""

_READINGTIME_PATTERN = re.compile(r"^\.\.\s+readingtime::", re.MULTILINE)
_NEED_PATTERN = re.compile(
    r"^\.\.\s+(tutorial|explanation|how-to|z_reference)::\s*(?P<title>.*)$"
)
_OPTION_PATTERN = re.compile(r"^\s+:(?P<name>[^:]+):\s*(?P<value>.*)$")
_TAG_SEPARATORS = re.compile(r"[;,]")


def is_readingtime_exception(docname: str) -> bool:
    """Return True if *docname* is exempt from the readingtime requirement."""
    return any(fnmatch.fnmatch(docname, pattern) for pattern in READINGTIME_EXCEPTIONS)


def is_generated_page(docname: str) -> bool:
    """Return True if *docname* is written during the build, not by an author.

    Such pages have no source file outside a build, so this hook cannot see
    them; the documentation build validates them from the sphinx-needs items
    instead.
    """
    return GENERATED_PARENT in PurePosixPath(docname).parents


def expected_diataxis_type(docname: str) -> str | None:
    """Return the sphinx-needs item type *docname* must carry, if any."""
    doc_path = PurePosixPath(docname)
    if doc_path.name == "sg_execution_times":
        return None
    for parent, diataxis in EXPECTED_TYPE_BY_PARENT:
        if PurePosixPath(parent) in doc_path.parents:
            if parent == "generated/gallery" and doc_path.name == "index":
                return None
            return diataxis
    return None


def _parse_directive(lines: list[str], index: int) -> tuple[dict[str, str], list[str]]:
    """Return the options and content of the directive starting at *index*."""
    options: dict[str, str] = {}
    content: list[str] = []
    # Options must directly follow the directive; the first blank line ends
    #  them, and everything indented after that is content.
    in_options = True
    for line in lines[index + 1 :]:
        if not line.strip():
            in_options = False
            continue
        if not line.startswith((" ", "\t")):
            # Dedent: the directive has ended.
            break
        option = _OPTION_PATTERN.match(line) if in_options else None
        if option:
            options[option["name"].strip()] = option["value"].strip()
        else:
            in_options = False
            content.append(line.strip())
    return options, content


def check_page(docname: str, text: str) -> list[str]:
    """Return a list of metadata problems with a page, empty if it is clean."""
    problems: list[str] = []

    if not is_readingtime_exception(docname) and not _READINGTIME_PATTERN.search(text):
        problems.append(
            "Missing '.. readingtime::' directive. Documentation pages should "
            "include a readingtime directive for consistency. If this page "
            f"should be exempt, add '{docname}' to READINGTIME_EXCEPTIONS in "
            ".hooks/check_docs_page_metadata.py"
        )

    expected_type = expected_diataxis_type(docname)
    if expected_type is None:
        return problems

    lines = text.splitlines()
    found = [
        (number, match)
        for number, line in enumerate(lines, start=1)
        if (match := _NEED_PATTERN.match(line))
    ]
    if len(found) != 1:
        problems.append(
            f"Page expected to have exactly 1 sphinx-needs item; found {len(found)}."
        )
        return problems

    line_number, match = found[0]
    if (page_type := match.group(1)) != expected_type:
        problems.append(
            f"sphinx-needs item expected to have type '{expected_type}'; "
            f"found type '{page_type}'."
        )
    if line_number > MAX_NEED_LINE:
        problems.append(
            "sphinx-needs item expected to be defined within first "
            f"{MAX_NEED_LINE} lines; found at line {line_number}."
        )

    # Title is not validated as it is always populated.

    options, content = _parse_directive(lines, line_number - 1)
    if not content:
        problems.append("sphinx-needs item must have non-empty content section.")

    tags = _TAG_SEPARATORS.split(options.get("tags", ""))
    if not any(tag.strip().startswith("topic_") for tag in tags):
        problems.append(
            "sphinx-needs item must have at least one 'topic_xxx' tag in its "
            "'tags' field."
        )

    return problems


def source_pages(docs_src: Path) -> list[tuple[str, Path]]:
    """Return the ``(docname, path)`` of every page this hook can check.

    Pages below ``generated/`` are excluded: they are build artifacts, present
    only after a documentation build, so including them would make the result
    depend on whether the caller happens to have built the docs.
    """
    pages = []
    for path in sorted(docs_src.rglob("*.rst")):
        relative = path.relative_to(docs_src)
        docname = relative.with_suffix("").as_posix()
        if is_generated_page(docname):
            continue
        pages.append((docname, path))
    return pages


def check_tree(docs_src: Path) -> list[str]:
    """Return a list of problem strings for every page under *docs_src*."""
    problems: list[str] = []
    for docname, path in source_pages(docs_src):
        text = path.read_text(encoding="utf-8")
        problems.extend(f"{path}: {problem}" for problem in check_page(docname, text))
    return problems


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    docs_src = Path(argv[0]) if argv else DOCS_SRC
    problems = check_tree(docs_src)

    if problems:
        print(
            "Documentation page metadata check failed.\n"
            "These problems would otherwise only surface during a docs build.\n"
        )
        for problem in problems:
            print(problem)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
