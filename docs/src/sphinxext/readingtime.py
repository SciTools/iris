# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Sphinx extension to estimate reading time for documentation pages."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import TYPE_CHECKING

from docutils import nodes  # type: ignore[import-untyped]
from sphinx.util.docutils import SphinxDirective

if TYPE_CHECKING:
    from sphinx.application import Sphinx

PATTERN = re.compile(r"^(\d+)wpm", re.IGNORECASE)
"""Words-per-minute pattern matcher."""

WPM = 125  # typically 100-150wpm
"""Default words-per-minute reading speed for technical documentation."""


def count_words(text: str) -> int:
    """Count the number of words in a given text.

    Parameters
    ----------
    text : str
        The text to count words in.

    Returns
    -------
    int
        The number of words in the text.

    """
    words = re.findall(r"\w+", text)
    return len(words)


class ReadingTimeDirective(SphinxDirective):
    """Directive to estimate reading time for a documentation page."""

    has_content = False
    optional_arguments = 1
    final_argument_whitespace = False

    def run(self) -> list[nodes.Node]:
        """Estimate the reading time for a documentation page.

        Directive accepts a single optional argument which may be either the
        proposed reading-time i.e., do not calculate an estimate instead use
        the value provided e.g. "10". Alternatively, the number of words per
        minute (override ``WPM`` default) to be used for the estimation
        e.g., "100wpm".

        Returns
        -------
        list[nodes.Node]
            A list containing a single raw HTML node with the estimated reading time.

        """
        env = self.env
        docname = env.docname
        source_path = env.doc2path(docname)

        minutes = wpm = None

        def convert(value: str) -> int:
            """Convert argument from string to integer."""
            try:
                result = int(value)
            except ValueError:
                result = 0
            return result

        if self.arguments:
            argument = self.arguments[0]

            match = PATTERN.match(argument)

            if match:
                wpm = convert(match.group(1))
            else:
                minutes = convert(argument)

        # optional wpm argument overrides the default for estimation
        wpm = wpm or WPM

        if minutes is None:
            with Path(source_path).open("r", encoding="utf-8") as fi:
                text = fi.read()

            # common reading speed baselines for technical docs is 100-150 wpm
            # let's roll with the lower end to be conservative and account for
            # code snippets, figures, etc.
            words = count_words(text)
            minutes = max(1, math.ceil(words / wpm))

        html = (
            f'<div class="reading-time">'
            f'<i class="fa-solid fa-clock"></i> '
            f"Estimated reading time: {minutes} minute{'s' if minutes != 1 else ''}"
            f"</div>"
        )

        return [nodes.raw("", html, format="html")]


def setup(app: Sphinx) -> dict:
    """Set up the reading time Sphinx extension.

    Parameters
    ----------
    app : Sphinx
        The Sphinx application object.

    Returns
    -------
    dict
        A dictionary containing the extension version.

    """
    app.add_directive("readingtime", ReadingTimeDirective)
    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
