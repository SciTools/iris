# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Sphinx customisations for a Diataxis User Manual (see diataxis.fr)."""

import enum
from pathlib import Path
import re
import typing

from docutils import nodes  # type: ignore[import-untyped]
from docutils.parsers.rst import Directive  # type: ignore[import-untyped]
from docutils.statemachine import StringList  # type: ignore[import-untyped]
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.util import logging as sphinx_logging
from sphinx_needs.api import get_needs_view

if typing.TYPE_CHECKING:
    from sphinx_needs.api.need import NeedsInfoType

logger = sphinx_logging.getLogger(__name__)


class Diataxis(enum.StrEnum):
    """The Diataxis-inspired sphinx-needs directives configured in conf.py."""

    # TODO: should user manual section indexes also get their own Diataxis tab?
    #  This would allow topic-based filtering, and allow all pages to be found
    #  through the same route.
    ALL = "all"
    TUTORIAL = "tutorial"
    EXPLANATION = "explanation"
    HOW_TO = "how-to"
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
        indented = ["   " + line for line in text.splitlines()]
        return "\n".join(indented)

    def _needtable(self, types: Diataxis, tags: str) -> str:
        """Construct a single sphinx-needs needtable directive string."""
        options = [
            ':columns: title;content as " "',
            ":colwidths: 30;60",
            ":style: table",
            ":sort: type",
            ":filter_warning: No pages for this filter.",
        ]
        # TODO: should the table somehow include what section the page belongs
        #  to? This isn't standard sphinx-needs metadata so would need
        #  `needs_extra_options` in conf.py.
        if types is not Diataxis.ALL:
            options.append(f":types: {types}")
        # TODO: is looking for `topic_all` brittle hard-coding?
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

        # TODO: should there be a caption for ALL as well? Even if that's just
        #  for visual consistency.
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


def validate_items(app: Sphinx, builder: Builder) -> None:
    """Validate that each user manual page has a single correctly configured item."""
    env = app.env
    found_docs: typing.Iterable[str] = env.found_docs

    # Read-only iterable of all sphinx-needs items; only valid in the write phase.
    needs_view = get_needs_view(app)
    # Group needs by docname
    by_doc: dict[Path, list[NeedsInfoType]] = {}
    for need_id in needs_view:
        need = needs_view[need_id]
        doc_name = need.get("docname")
        if not doc_name:
            # External/imported needs may have no docname; skip page accounting
            continue
        by_doc.setdefault(Path(doc_name), []).append(need)

    def _get_expected_type(doc_path: Path) -> typing.Optional[Diataxis]:
        """Get the expected Diataxis type for the given document path."""
        parents_and_diataxis = [
            (Path("generated/api"), Diataxis.REFERENCE),
            (Path("generated/gallery"), Diataxis.HOW_TO),
            (Path("user_manual/tutorial"), Diataxis.TUTORIAL),
            (Path("user_manual/explanation"), Diataxis.EXPLANATION),
            (Path("user_manual/how_to"), Diataxis.HOW_TO),
            (Path("user_manual/reference"), Diataxis.REFERENCE),
        ]
        expected = None
        for parent, diataxis in parents_and_diataxis:
            if parent in doc_path.parents:
                expected = diataxis
                break
        if Path("generated/gallery") in doc_path.parents and doc_path.name == "index":
            expected = None
        if doc_path.name == "sg_execution_times":
            expected = None
        return expected

    for doc_name in found_docs:
        doc_path = Path(doc_name)
        expected_type = _get_expected_type(doc_path)
        if expected_type is not None:
            problem_prefix = "Page expected to have exactly 1 sphinx-needs item;"
            try:
                (page_need,) = by_doc[doc_path]
            except KeyError:
                problem = f"{problem_prefix} found 0."
                logger.error(problem, location=doc_name)
                continue
            except ValueError:
                count = len(by_doc[doc_path])
                problem = f"{problem_prefix} found {count}."
                logger.error(problem, location=doc_name)
                continue

            if (page_type := page_need["type"]) != expected_type:
                problem = (
                    "sphinx-needs item expected to have type "
                    f"'{expected_type}'; found type '{page_type}'."
                )
                logger.error(problem, location=doc_name)

            if (line_no := page_need.get("lineno")) > 25:
                # Ensures that links to the needs directive take reader to the
                #  start of the page.
                problem = (
                    "sphinx-needs item expected to be defined within "
                    f"first 25 lines; found at line {line_no}."
                )
                logger.error(problem, location=doc_name)

            # Title is not validated as it is always populated.

            if page_need["content"] == "":
                problem = "sphinx-needs item must have non-empty content section."
                logger.error(problem, location=doc_name)

            tags = page_need.get("tags", [])
            if [tag for tag in tags if tag.startswith("topic_")] == []:
                problem = (
                    "sphinx-needs item must have at least one 'topic_xxx' tag "
                    "in its 'tags' field."
                )
                logger.error(problem, location=doc_name)


def setup(app: Sphinx):
    """Set up the Sphinx extension.

    This function is expected by Sphinx to register the extension.
    """
    # Connect at write-started so needs are fully collected & resolved.
    app.connect("write-started", validate_items)

    app.add_directive("diataxis-page-list", DiataxisDirective)

    return {"version": "0.1"}
