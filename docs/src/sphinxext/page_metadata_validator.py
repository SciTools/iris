# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Sphinx extension to validate the metadata every documentation page carries.

The rules themselves live in ``.hooks/check_docs_page_metadata.py``, which is
also a pre-commit hook.  Keeping them there means an author gets told about a
missing ``readingtime`` directive or sphinx-needs item in milliseconds, rather
than several minutes into a documentation build — while this extension keeps
the build itself authoritative, so nothing depends on a hook having been run.

Pages are validated by whichever means can see them:

* authored pages are checked from their source text, by the shared rules;
* pages below ``generated/`` are written by autosummary and sphinx-gallery
  while the build runs, so they have no source for the hook to read.  They are
  checked here instead, from the resolved sphinx-needs items.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import typing

from sphinx.util import logging as sphinx_logging
from sphinx_needs.api import get_needs_view

if typing.TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.builders import Builder
    from sphinx_needs.api.need import NeedsInfoType

logger = sphinx_logging.getLogger(__name__)


def _load_page_metadata_rules():
    """Load the validation rules shared with the pre-commit hook.

    The rules live outside the documentation tree, in ``.hooks/``, alongside
    Iris' other pre-commit hooks; that directory is not importable, so the
    module is loaded by path.  ``lib/iris/tests/test_coding_standards.py``
    loads the other hooks the same way.
    """
    hook_path = Path(__file__).parents[3] / ".hooks" / "check_docs_page_metadata.py"
    spec = importlib.util.spec_from_file_location("check_docs_page_metadata", hook_path)
    if spec is None or spec.loader is None:
        message = f"Failed to load the page metadata rules from: {hook_path}"
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rules = _load_page_metadata_rules()


def _validate_source_page(docname: str, source_path: Path) -> None:
    """Report any metadata problems with an authored page."""
    text = source_path.read_text(encoding="utf-8")
    for problem in rules.check_page(docname, text):
        logger.error(problem, location=docname)


def _validate_generated_page(
    docname: str,
    expected_type: str,
    page_needs: list[NeedsInfoType],
) -> None:
    """Report any metadata problems with a page written during the build."""
    problem_prefix = "Page expected to have exactly 1 sphinx-needs item;"
    if len(page_needs) != 1:
        problem = f"{problem_prefix} found {len(page_needs)}."
        logger.error(problem, location=docname)
        return

    (page_need,) = page_needs

    if (page_type := page_need["type"]) != expected_type:
        problem = (
            "sphinx-needs item expected to have type "
            f"'{expected_type}'; found type '{page_type}'."
        )
        logger.error(problem, location=docname)

    if (line_no := page_need.get("lineno")) > rules.MAX_NEED_LINE:
        # Ensures that links to the needs directive take reader to the
        #  start of the page.
        problem = (
            "sphinx-needs item expected to be defined within "
            f"first {rules.MAX_NEED_LINE} lines; found at line {line_no}."
        )
        logger.error(problem, location=docname)

    # Title is not validated as it is always populated.

    if page_need["content"] == "":
        problem = "sphinx-needs item must have non-empty content section."
        logger.error(problem, location=docname)

    tags = page_need.get("tags", [])
    if [tag for tag in tags if tag.startswith("topic_")] == []:
        problem = (
            "sphinx-needs item must have at least one 'topic_xxx' tag "
            "in its 'tags' field."
        )
        logger.error(problem, location=docname)


def validate_pages(app: Sphinx, builder: Builder) -> None:
    """Validate that every page carries the metadata it is required to have."""
    env = app.env

    # Read-only iterable of all sphinx-needs items; only valid in the write phase.
    needs_view = get_needs_view(app)
    # Group needs by docname.
    by_doc: dict[str, list[NeedsInfoType]] = {}
    for need_id in needs_view:
        need = needs_view[need_id]
        doc_name = need.get("docname")
        if not doc_name:
            # External/imported needs may have no docname; skip page accounting.
            continue
        by_doc.setdefault(doc_name, []).append(need)

    for docname in env.found_docs:
        if rules.is_generated_page(docname):
            expected_type = rules.expected_diataxis_type(docname)
            if expected_type is not None:
                _validate_generated_page(
                    docname, expected_type, by_doc.get(docname, [])
                )
        else:
            _validate_source_page(docname, Path(env.doc2path(docname)))


def setup(app: Sphinx) -> dict:
    # Connect at write-started so needs are fully collected & resolved.
    app.connect("write-started", validate_pages)

    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
