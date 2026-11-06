# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Sphinx customisations for a Diataxis User Manual (see diataxis.fr)."""

import enum
from pathlib import Path
import re
from textwrap import indent

from docutils import nodes  # type: ignore[import-untyped]
from docutils.parsers.rst import Directive  # type: ignore[import-untyped]
from docutils.statemachine import StringList  # type: ignore[import-untyped]
from sphinx.application import Sphinx


class Diataxis(enum.StrEnum):
    """The Diataxis-inspired sphinx-needs directives configured in conf.py.

    Which pages are expected to carry which of these is recorded in
    ``.hooks/check_docs_page_metadata.py`` - by directory in
    ``EXPECTED_TYPE_BY_PARENT``, and in ``GALLERY_TYPE`` and ``API_TYPE`` for
    the generated pages; renaming a member here means editing those too.
    """

    ALL = "all"
    TUTORIAL = "tutorial"
    EXPLANATION = "explanation"
    HOW_TO = "how-to"
    # z_ prefix to force to the end of sorted lists.
    REFERENCE = "z_reference"


DIATAXIS_CAPTIONS = {
    Diataxis.TUTORIAL: "Guided lessons for understanding a topic.\n\n(Supports **study**, via **action**)",
    Diataxis.EXPLANATION: "In-depth discussion for understanding concepts.\n\n(Supports **study**, via **theory**)",
    Diataxis.HOW_TO: "Step by step instructions for achieving a specific goal.\n\n(Supports **work**, via **action**)",
    Diataxis.REFERENCE: "Concise information to look up when needed.\n\n(Supports **work**, via **theory**)",
}
"""Text to be displayed at the top of each Diataxis tab."""


class DiataxisDirective(Directive):
    """A topic-filtered tab-set block with Diataxis tab-items and topic navigation badges."""

    has_content = True
    """Content = the topic tag to filter by, e.g. `topic_about`."""

    @staticmethod
    def _indent(text: str) -> str:
        return indent(text, "   ")

    def _needtable(self, types: Diataxis, tags: str) -> str:
        """Construct a single sphinx-needs needtable directive string."""
        options = [
            ':columns: title as " ";content as " "',
            ":colwidths: 30;60",
            ":style: table",
            ":sort: type",
            ":filter_warning: No pages for this filter.",
        ]
        if types is not Diataxis.ALL:
            options.append(f":types: {types}")
        if tags != "topic_all":
            options.append(f":tags: {tags}")
        options_str = "\n".join(options)
        needtable = "\n".join(
            [
                ".. needtable::",
                self._indent(options_str),
            ]
        )
        return needtable

    def _tab_item(self, diataxis: Diataxis, tags: str) -> str:
        """Construct a single tab-item string for the given Diataxis type."""
        needtable = self._needtable(types=diataxis, tags=tags)

        # Convert the Diataxis directive name to a pretty title.
        tab_item_title = str(diataxis)
        tab_item_title = tab_item_title.removeprefix("z_")
        tab_item_title = tab_item_title.capitalize()

        caption = DIATAXIS_CAPTIONS.get(diataxis, "")
        content = [
            # sync means all tab-sets on this page switch tabs together.
            f":sync: {diataxis}",
            "",
            caption,
            "",
            needtable,
        ]
        content_str = "\n".join(content)
        tab_item = "\n".join(
            [
                f".. tab-item:: {tab_item_title}",
                self._indent(content_str),
            ]
        )
        return tab_item

    def run(self):
        """Construct the navigation badges followed by the Diataxis tab-set."""
        # Enforce the only valid location for this directive.
        rst_path = Path(self.state.document["source"])
        if not (rst_path.parent.name == "user_manual" and rst_path.name == "index.rst"):
            message = "Expected directive to only be used in user_manual/index.rst"
            error = self.state_machine.reporter.error(message, line=self.lineno)
            return [error]

        # Find all the topic labels in this file and construct navigation badges
        #  for them.
        label_pattern = re.compile(r"^\.\. _(topic_.+):$", re.MULTILINE)
        topic_labels = label_pattern.findall(rst_path.read_text())
        # The 'current' topic is highlighted differently.
        badges = {
            label: "bdg-ref-primary"
            if label == self.content[0]
            else "bdg-ref-primary-line"
            for label in topic_labels
        }
        # Parse the badges as RST.
        node = nodes.Element()
        self.state.nested_parse(
            StringList([f":{badge}:`{label}`" for label, badge in badges.items()]),
            self.content_offset,
            node,
        )

        # Construct the Diataxis tab-set.
        tab_items = [
            self._tab_item(diataxis=diataxis, tags=self.content[0])
            for diataxis in Diataxis
        ]
        tab_items_str = "\n\n".join(tab_items)
        tab_set = "\n".join(
            [
                ".. tab-set::",
                "",
                self._indent(tab_items_str),
            ]
        )
        # Parse the tab set as RST.
        self.state.nested_parse(
            StringList(tab_set.splitlines()), self.content_offset, node
        )

        return node.children


def setup(app: Sphinx):
    """Set up the Sphinx extension.

    This function is expected by Sphinx to register the extension.
    """
    app.add_directive("diataxis-page-list", DiataxisDirective)

    return {"version": "0.1"}
