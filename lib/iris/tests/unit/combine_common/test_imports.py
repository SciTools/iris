# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test the import-freedom of :mod:`iris._combine_common`."""

import ast
from pathlib import Path

from iris import _combine_common


class RuntimeImports(ast.NodeVisitor):
    """Collect modules imported at runtime, skipping ``TYPE_CHECKING`` blocks."""

    def __init__(self):
        self.modules: set[str] = set()

    def visit_If(self, node: ast.If) -> None:
        guard = node.test
        type_checking = (
            isinstance(guard, ast.Name) and guard.id == "TYPE_CHECKING"
        ) or (isinstance(guard, ast.Attribute) and guard.attr == "TYPE_CHECKING")
        if type_checking:
            # Only the ``else`` branch runs at runtime.
            for statement in node.orelse:
                self.visit(statement)
        else:
            self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.modules.update(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            # A relative import is necessarily an Iris import.
            self.modules.add("." * node.level + (node.module or ""))
        elif node.module is not None:
            self.modules.add(node.module)


def test_no_runtime_iris_imports():
    """The substrate sits below merge and concatenate, so imports no Iris."""
    source = Path(_combine_common.__file__).read_text(encoding="utf-8")
    visitor = RuntimeImports()
    visitor.visit(ast.parse(source))
    offenders = {
        module
        for module in visitor.modules
        if module == "iris" or module.startswith(("iris.", "."))
    }
    assert offenders == set(), (
        "iris._combine_common must import nothing from Iris at runtime; "
        f"found {sorted(offenders)}. Put annotation-only imports behind "
        "`if TYPE_CHECKING:`."
    )


def test_type_checking_imports_are_detected():
    """Guard the guard: a runtime Iris import must be seen as one."""
    source = "\n".join(
        [
            "from typing import TYPE_CHECKING",
            "if TYPE_CHECKING:",
            "    from iris.coords import DimCoord",
            "from iris.cube import Cube",
        ]
    )
    visitor = RuntimeImports()
    visitor.visit(ast.parse(source))
    assert "iris.cube" in visitor.modules
    assert "iris.coords" not in visitor.modules
