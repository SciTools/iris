# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.io.format_picker.FormatAgent` class."""

from iris.fileformats import FORMAT_AGENT
import iris.tests as tests


class TestFormatAgent(tests.IrisTest):
    def test_copy_is_equal(self):
        format_agent_copy = FORMAT_AGENT.copy()
        self.assertEqual(format_agent_copy, FORMAT_AGENT)

    def test_modified_copy_not_equal(self):
        format_agent_copy = FORMAT_AGENT.copy()
        format_agent_copy._format_specs.pop()
        self.assertNotEqual(format_agent_copy, FORMAT_AGENT)


if __name__ == "__main__":
    tests.main()
