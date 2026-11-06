# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Sphinx extension to validate the metadata every documentation page carries.

The rules themselves live in ``.hooks/check_docs_page_metadata.py``, which is
also a pre-commit hook.  Keeping them there means an author gets told about a
missing ``readingtime`` directive or sphinx-needs item in milliseconds, rather
than several minutes into a documentation build - while this extension keeps
the build itself authoritative, so nothing depends on a hook having been run.

Every rule is answerable from version-controlled source, including the rules
for pages generated from gallery scripts and library modules, so this
extension does not inspect anything the build produces.  It runs the same
check over the same files that the hook does; only the trigger differs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import typing

from sphinx.util import logging as sphinx_logging

if typing.TYPE_CHECKING:
    from sphinx.application import Sphinx

logger = sphinx_logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parents[3]


def _load_page_metadata_rules():
    """Load the validation rules shared with the pre-commit hook.

    The rules live outside the documentation tree, in ``.hooks/``, alongside
    Iris' other pre-commit hooks; that directory is not importable, so the
    module is loaded by path.  ``lib/iris/tests/test_coding_standards.py``
    loads the other hooks the same way.
    """
    hook_path = REPO_ROOT / ".hooks" / "check_docs_page_metadata.py"
    spec = importlib.util.spec_from_file_location("check_docs_page_metadata", hook_path)
    if spec is None or spec.loader is None:
        message = f"Failed to load the page metadata rules from: {hook_path}"
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rules = _load_page_metadata_rules()


def validate_pages(app: Sphinx) -> None:
    """Report every page whose metadata is missing or malformed."""
    for problem in rules.check_tree(REPO_ROOT):
        logger.error(problem)


def setup(app: Sphinx) -> dict:
    # Connect at builder-inited: the check reads source files only, so there
    #  is nothing to wait for, and reporting before the build starts is
    #  quicker for the author than reporting at the end of it.
    app.connect("builder-inited", validate_pages)

    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
