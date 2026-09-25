# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Pre-commit hook: validate the metadata every documentation page must carry.

Two requirements are checked:

* every non-exempt page carries a ``.. readingtime::`` directive;
* every page with a Diataxis type carries exactly one correctly-configured
  sphinx-needs item.

Every page's metadata is authored in a version-controlled file, so none of
this needs a documentation build to answer.  Three kinds of file carry it:

* pages under ``docs/src`` carry theirs in their own reStructuredText;
* gallery pages are built from the scripts in ``docs/gallery_code``, and
  carry theirs in the script's module docstring;
* API pages are built from the library itself, and carry theirs in the
  module's own docstring.

The last two are the reason this hook does not simply walk ``.rst`` files:
their pages appear under ``docs/src/generated`` during a build, and the
generated reStructuredText contains an ``automodule`` directive rather than
the item, which is only resolved later.  Checking the docstrings instead
reaches the same conclusion from files that are always present.

The same rules are enforced during the documentation build, by the Sphinx
extension in ``docs/src/sphinxext/page_metadata_validator.py``, which imports
this module rather than reimplementing it.  That is the point of putting the
logic here: a rule added below is enforced in both places at once, and the two
can never disagree about whether a page is valid.

Usage (the hook passes no filenames; the whole repository is walked)::

    python .hooks/check_docs_page_metadata.py [repository-root]

Exit codes:
    0 – all clear
    1 – one or more pages are missing required metadata
"""

import ast
import fnmatch
from pathlib import Path, PurePosixPath
import re
import sys

REPO_ROOT = Path(__file__).parent.parent
"""The repository root, relative to this file."""

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
    "whatsnew/**",
}
"""
Pages exempt from readingtime validation.

Supports glob patterns using standard wildcards.
"""

EXPECTED_TYPE_BY_PARENT = (
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

GALLERY_TYPE = "how-to"
"""The item type every gallery script's docstring is expected to carry."""

API_TYPE = "z_reference"
"""The item type every documented library module's docstring is expected to carry."""

MAX_NEED_LINE = 25
"""Items must start within this many lines, so links land at the page top."""

_READINGTIME_PATTERN = re.compile(r"^\.\.\s+readingtime::", re.MULTILINE)
_NEED_PATTERN = re.compile(
    r"^\.\.\s+(tutorial|explanation|how-to|z_reference)::\s*(?P<title>.*)$"
)
_OPTION_PATTERN = re.compile(r"^\s+:(?P<name>[^:]+):\s*(?P<value>.*)$")
_TAG_SEPARATORS = re.compile(r"[;,]")


def is_readingtime_exception(docname: str) -> bool:
    """Return True if *docname* is exempt from the readingtime requirement."""
    return any(fnmatch.fnmatch(docname, pattern) for pattern in READINGTIME_EXCEPTIONS)


def expected_diataxis_type(docname: str) -> str | None:
    """Return the sphinx-needs item type *docname* must carry, if any."""
    doc_path = PurePosixPath(docname)
    for parent, diataxis in EXPECTED_TYPE_BY_PARENT:
        if PurePosixPath(parent) in doc_path.parents:
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


def check_item(text: str, expected_type: str) -> list[str]:
    """Return the problems with the sphinx-needs item in *text*, empty if clean.

    *text* is whatever will be rendered at the top of the page - a page's own
    reStructuredText, or the module docstring a page is generated from.
    """
    problems: list[str] = []

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
    if expected_type is not None:
        problems.extend(check_item(text, expected_type))

    return problems


def check_docstring(text: str, expected_type: str) -> list[str]:
    """Return a list of metadata problems with a Python file's docstring.

    The module docstring is rendered at the top of the page generated from
    *text*, and is therefore where that page's item is authored.
    """
    try:
        docstring = ast.get_docstring(ast.parse(text))
    except SyntaxError as error:
        return [f"Could not be parsed, so its docstring cannot be checked: {error}"]

    if docstring is None:
        return [
            "Missing a module docstring, so the page generated from it has no "
            "sphinx-needs item."
        ]

    return check_item(docstring, expected_type)


def source_pages(docs_src: Path) -> list[tuple[str, Path]]:
    """Return the ``(docname, path)`` of every authored documentation page.

    Anything below ``generated/`` is a build artifact - present only after a
    documentation build, and only until the next ``make clean``.  Including it
    would make the result depend on whether the caller happens to have built
    the docs; the files those pages are generated from are checked instead, by
    :func:`gallery_scripts` and :func:`library_modules`.
    """
    pages = []
    for path in sorted(docs_src.rglob("*.rst")):
        relative = path.relative_to(docs_src)
        if relative.parts[0] == "generated":
            continue
        pages.append((relative.with_suffix("").as_posix(), path))
    return pages


def gallery_scripts(gallery_code: Path) -> list[Path]:
    """Return every gallery script that becomes a documentation page.

    sphinx-gallery builds a page from each script matching its
    ``filename_pattern``, configured in ``docs/src/conf.py``.
    """
    return sorted(gallery_code.rglob("plot_*.py"))


def library_modules(library: Path) -> list[Path]:
    """Return every library module that becomes an API page.

    ``sphinxcontrib.apidoc`` writes one page per module - see the ``apidoc_*``
    settings in ``docs/src/conf.py`` - skipping the tests, and skipping
    anything private.  ``iris.experimental.raster`` is also skipped there, to
    avoid a gdal conflict, but is deliberately not skipped here: it carries
    its item like any other module, and an exception would be more to explain
    than it saves.
    """
    modules = []
    for path in sorted(library.rglob("*.py")):
        parts = path.relative_to(library).with_suffix("").parts
        if parts[1:2] == ("tests",):
            continue
        if any(part.startswith("_") and part != "__init__" for part in parts):
            continue
        modules.append(path)
    return modules


def check_tree(repo_root: Path) -> list[str]:
    """Return a list of problem strings for every page *repo_root* produces."""
    problems: list[str] = []

    def record(path: Path, found: list[str]) -> None:
        problems.extend(f"{path}: {problem}" for problem in found)

    for docname, path in source_pages(repo_root / "docs" / "src"):
        record(path, check_page(docname, path.read_text(encoding="utf-8")))

    for path in gallery_scripts(repo_root / "docs" / "gallery_code"):
        record(path, check_docstring(path.read_text(encoding="utf-8"), GALLERY_TYPE))

    for path in library_modules(repo_root / "lib"):
        record(path, check_docstring(path.read_text(encoding="utf-8"), API_TYPE))

    return problems


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    repo_root = Path(argv[0]) if argv else REPO_ROOT
    problems = check_tree(repo_root)

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
