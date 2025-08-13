# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

import importlib
import pathlib

import pytest

BASE = pathlib.Path("..")
IGNORE = ["__init__.py", "__pycache__", "pandas.py", "raster.py", "tests"]


class TestImports:
    @staticmethod
    def itree(path, prefix=None, debug=False):
        if prefix is None:
            prefix = "iris"

        if debug:
            print(f"<{path}> prefix=<{prefix}>")

        emsg = '\n*** Failed to import "{}" ***'

        if path.is_dir() and path.name not in IGNORE:
            children = sorted(path.iterdir())
            if debug:
                print(f"\nchildren {children}")
            dunder_init = path / "__init__.py"
            if dunder_init in children:
                package = prefix
                if debug:
                    print(f"import {package}")
                try:
                    importlib.import_module(package)
                except (ImportError, ModuleNotFoundError):
                    pytest.fail(emsg.format(package))
                for child in children:
                    parent = f"{prefix}.{child.stem}" if child.is_dir() else prefix
                    TestImports.itree(child, prefix=parent, debug=debug)
        elif path.is_file() and path.name not in IGNORE and path.suffix == ".py":
            package = f"{prefix}.{path.stem}"
            if debug:
                print(f"import {package}")
            try:
                importlib.import_module(package)
            except (ImportError, ModuleNotFoundError):
                pytest.fail(emsg.format(package))

    def test_imports(self):
        self.itree(BASE, debug=False)
