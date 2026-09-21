# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Pre-commit hook: ensure netCDF4 is only imported via _thread_safe_nc.

Direct imports of the netCDF4 library are forbidden throughout Iris (except in
the permitted files listed below).  All access must go via the thread-safe
wrapper at ``iris.fileformats.netcdf._thread_safe_nc``.

Usage (pre-commit passes staged file paths as arguments)::

    python .hooks/check_netcdf4_imports.py [file ...]

Exit codes:
    0 – all clear
    1 – one or more files contain a forbidden netCDF4 import
"""

import ast
from pathlib import Path
import sys

# Files that are allowed to import netCDF4 directly.
_PERMITTED_SUFFIXES = (
    # The thread-safe wrapper itself.
    "iris/fileformats/netcdf/_thread_safe_nc.py",
    # The test for the wrapper.
    "iris/tests/unit/fileformats/netcdf/_thread_safe_nc/test_NetCDFWriteProxy.py",
    # The tests for the bytecoding dataset wrapper.
    "iris/tests/unit/fileformats/netcdf/test_bytecoding_datasets.py",
    # The system test that checks netCDF4 is importable.
    "iris/tests/system_test.py",
)


def _is_permitted(path: Path) -> bool:
    """Return True if *path* is one of the permitted files."""
    as_posix = path.as_posix()
    return any(as_posix.endswith(suffix) for suffix in _PERMITTED_SUFFIXES)


def _has_netcdf4_import(node: ast.AST) -> bool:
    """Return True if *node* is a direct netCDF4 import statement."""
    if isinstance(node, ast.Import):
        return any(alias.name == "netCDF4" for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return node.module is not None and node.module.startswith("netCDF4")
    return False


def check_file(path: Path) -> list[str]:
    """Return a list of violation strings for *path*, empty if clean."""
    if _is_permitted(path):
        return []

    file_text = path.read_text()

    # Fast pre-filter: skip parsing if "netCDF4" doesn't appear at all.
    if "netCDF4" not in file_text:
        return []

    try:
        tree = ast.parse(source=file_text, filename=str(path))
    except SyntaxError:
        # Let other hooks (check-ast) handle syntax errors.
        return []

    violations = []
    for imp in ast.walk(tree):
        if not isinstance(imp, (ast.Import, ast.ImportFrom)):
            continue
        if _has_netcdf4_import(imp):
            violations.append(
                f"{path}:{getattr(imp, 'lineno', 0)}: direct netCDF4 import — "
                "use iris.fileformats.netcdf._thread_safe_nc instead."
            )

    return violations


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    paths = [Path(p) for p in argv]
    all_violations: list[str] = []

    for path in paths:
        all_violations.extend(check_file(path))

    if all_violations:
        print(
            "netCDF4 import check failed.\n"
            "All netCDF4 imports must be via iris.fileformats.netcdf._thread_safe_nc.\n"
        )
        for v in all_violations:
            print(v)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
