# TODO: header
# TODO: docstrings/comments. How to document an extension/directive?
# TODO: linting

# TODO: include a validation routine, calling to sphinx-needs to confirm that
#  all User Manual pages have the minimum metadata.
#  - sphinx-need directive
#    - :tags: including a topic_xxx tag
#    - think sphinx-needs will already insist on a valid type?
#    - think sphinx-needs will already insist on a title?
#    - think sphinx-needs will already insist on a description?
#  - as it stands this would exclude all section indexes, unless the section
#    indexes get their own Diataxis tab.

from pathlib import Path
import enum
import re

from docutils import nodes  # type: ignore[import-untyped]
from docutils.parsers.rst import Directive  # type: ignore[import-untyped]
from docutils.statemachine import StringList  # type: ignore[import-untyped]


class Diataxis(enum.StrEnum):
    # TODO: should user manual section indexes also get their own Diataxis tab?
    #  This would allow topic-based filtering, and allow all pages to be found
    #  through the same route.

    ALL = "all"
    TUTORIAL = "tutorial"
    HOW_TO = "how-to"
    REFERENCE = "reference"
    EXPLANATION = "explanation"


DIATAXIS_CAPTIONS = {
    Diataxis.TUTORIAL: "Guided lessons for understanding a topic. (Supports **study**, via **action**)",
    Diataxis.HOW_TO: "Step by step instructions for achieving a specific goal. (Supports **work**, via **action**)",
    Diataxis.EXPLANATION: "In-depth discussion for understanding concepts. (Supports **study**, via **theory**)",
    Diataxis.REFERENCE: "Concise information to look up when needed. (Supports **work**, via **theory**)",
}


class DiataxisDirective(Directive):
    has_content = True

    @staticmethod
    def _indent(text: str) -> str:
        indented = ["   " + line for line in text.splitlines()]
        return "\n".join(indented)

    def _needtable(self, types: Diataxis, tags: str) -> str:
        options = [
            ':columns: id as "Link";title;content as " "',
            ":colwidths: 10;30;60",
            ":style: table",
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
        needtable = "\n".join([
            ".. needtable::",
            self._indent(options_str),
        ])
        return needtable

    def _tab_item(self, diataxis: Diataxis, tags: str) -> str:
        needtable = self._needtable(types=diataxis, tags=tags)
        tab_item_title = diataxis.capitalize()
        # TODO: should there be a caption for ALL as well? Even if that's just
        #  for visual consistency.
        caption = DIATAXIS_CAPTIONS.get(diataxis, "")
        if diataxis is not Diataxis.ALL:
            tab_item_title += "s"
        content = [
            f":sync: {diataxis}",
            "",
            caption,
            "",
            needtable,
        ]
        content_str = "\n".join(content)
        tab_item = "\n".join([
            f".. tab-item:: {tab_item_title}",
            self._indent(content_str),
        ])
        return tab_item

    def run(self):
        rst_path = Path(self.state.document['source'])
        # user_manual.index.rst
        if not (rst_path.parent.name == "user_manual" and rst_path.name == "index.rst"):
            message = "Expected directive to only be used in user_manual/index.rst"
            error = self.state_machine.reporter.error(
                message, line=self.lineno
            )
            return [error]
        label_pattern = re.compile(r"^\.\. _(topic_.+):$", re.MULTILINE)
        topic_labels = label_pattern.findall(rst_path.read_text())
        badges = {
            label: "bdg-ref-primary" if label == self.content[0] else "bdg-ref-primary-line"
            for label in topic_labels
        }
        # Parse the badges as RST.
        node = nodes.Element()
        self.state.nested_parse(
            StringList([f":{badge}:`{label}`" for label, badge in badges.items()]),
            self.content_offset,
            node
        )

        tab_items = [
            self._tab_item(diataxis=diataxis, tags=self.content[0])
            for diataxis in Diataxis
        ]
        tab_items_str = "\n\n".join(tab_items)
        tab_set = "\n".join([
            ".. tab-set::",
            "",
            self._indent(tab_items_str),
        ])
        # Parse the tab set as RST.
        self.state.nested_parse(
            StringList(tab_set.splitlines()),
            self.content_offset,
            node
        )

        return node.children


def setup(app):
    app.add_directive("diataxis-page-list", DiataxisDirective)
    return {"version": "0.1"}
