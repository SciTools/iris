# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""The CF layer must not branch on which storage implementation it holds.

`isinstance(thing, CFDataset)` tests the *interface*, which is how generic
code is supposed to accept any backend, and `CFReader` does exactly that.
What is banned is testing an *implementation*: that is what makes a third
backend a rewrite rather than a new module. Constructing one is a third
thing again, and legitimate - `CFReader` opens a `NetCDFDataset` for a bare
path, which is a default, not a branch on something it was handed.
"""

from pathlib import Path

import iris.fileformats.cf

#: Implementation class names that may not appear in a type test.
IMPLEMENTATION_NAMES = (
    "NetCDFDataset",
    "NetCDFDatasetVariable",
    "ZarrDataset",
    "ZarrDatasetVariable",
)

#: The spellings of a type test. `__class__` catches the hand-rolled form.
TYPE_TESTS = ("isinstance", "issubclass", "__class__", "type(")


def cf_package_sources():
    """Yield `(path, text)` for every module in `iris.fileformats.cf`."""
    package = Path(iris.fileformats.cf.__file__).parent
    for path in sorted(package.glob("*.py")):
        yield path, path.read_text(encoding="utf-8")


def test_no_module_type_tests_an_implementation_class():
    offenders = []
    for path, text in cf_package_sources():
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            names_one = any(name in line for name in IMPLEMENTATION_NAMES)
            tests_type = any(spelling in line for spelling in TYPE_TESTS)
            if names_one and tests_type:
                offenders.append(f"{path.name}:{line_number}: {line.strip()}")
    assert offenders == []
