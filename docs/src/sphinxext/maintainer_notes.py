# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""The ``maintainer-notes`` directive: docstring prose that is never published.

A module docstring is the only prose read on every visit to a module, checked
in the diff, and impossible to lose track of. That makes it the right home for
subsystem explanation - but Sphinx publishes it, so prose written for whoever
maintains the module arrives in front of whoever uses it. This directive is
the way out: docutils collects its content and this returns nothing, so the
prose stays in the file and reaches no page.

The content is deliberately never parsed. :meth:`MaintainerNotes.run` returns
no nodes and never calls ``nested_parse``, so anything inside is free text.
Broken markup, unbalanced ``>>>`` and stray asterisks are all safe, and the
doctest builder does not see them either. That is the point of the design, not
an oversight: an agent writing notes cannot break a docs build with them.

Getting the name wrong fails loudly. ``make html`` runs with ``-W``, so an
unknown directive stops the build, and the content does not render on the way
out. A plain reST comment fails the other way: dedent one line and the prose
is silently published.
"""

from docutils.parsers.rst import Directive  # type: ignore[import-untyped]
from sphinx.application import Sphinx


class MaintainerNotes(Directive):
    """Swallow the directive's content, contributing nothing to the page."""

    has_content = True
    optional_arguments = 1
    final_argument_whitespace = True

    def run(self):
        """Return no nodes, leaving the content unparsed and unpublished."""
        return []


def setup(app: Sphinx):
    """Set up the Sphinx extension.

    This function is expected by Sphinx to register the extension.
    """
    app.add_directive("maintainer-notes", MaintainerNotes)

    return {"version": "0.1"}
