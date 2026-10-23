# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Pre-commit hook: ensure all warnings are categorised.

All UserWarnings raised by Iris must be categorised (not use the base
UserWarning class), to allow users to filter them.

Usage (pre-commit passes staged file paths as arguments)::

    python .hooks/check_categorised_warnings.py [file ...]

Exit codes:
    0 – all clear
    1 – one or more files have categorisation issues
"""

import ast
from pathlib import Path
import sys


def _get_warnings_issues(path: Path) -> list[str]:
    """Return a list of warning categorisation issues in *path*."""
    try:
        file_text = path.read_text()
    except (OSError, UnicodeDecodeError):
        # Let other tools handle file read errors
        return []

    try:
        tree = ast.parse(source=file_text, filename=str(path))
    except SyntaxError:
        # Let other hooks (check-ast) handle syntax errors
        return []

    violations = []

    # Find all .warn() calls
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Check if this is a .warn(...) call
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "warn"):
            continue

        warn_ref = f"{path}:{node.lineno}"

        # Check for 'category' keyword argument
        category_kwargs = [k for k in node.keywords if k.arg == "category"]

        if not category_kwargs:
            violations.append(
                f"{warn_ref}: warning raised without category= kwarg — "
                "use a specific IrisWarning subclass"
            )
        else:
            category_kwarg = category_kwargs[0]
            # Check if it's using the base UserWarning
            category_name = None
            if isinstance(category_kwarg.value, ast.Name):
                category_name = category_kwarg.value.id
            elif isinstance(category_kwarg.value, ast.Attribute):
                category_name = category_kwarg.value.attr

            if category_name == "UserWarning":
                violations.append(
                    f"{warn_ref}: warning uses base UserWarning — "
                    "use a specific IrisWarning subclass"
                )

    return violations


def check_file(path: Path) -> list[str]:
    """Return a list of violation strings for *path*, empty if clean."""
    if not path.suffix == ".py":
        return []

    return _get_warnings_issues(path)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    paths = [Path(p) for p in argv]
    all_violations: list[str] = []

    for path in paths:
        all_violations.extend(check_file(path))

    if all_violations:
        print(
            "Warning categorisation check failed.\n"
            "All warnings raised by Iris must be categorised with a specific "
            "IrisWarning subclass.\n"
        )
        for v in all_violations:
            print(v)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
