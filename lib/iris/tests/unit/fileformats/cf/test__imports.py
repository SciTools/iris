# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Every module in the CF and netCDF packages can be imported first.

The two packages import each other: the CF reader builds a netCDF dataset, and
the netCDF saver builds on the CF saver. Nothing in the source states the order
the module bodies must run in, so these cases pin that every module survives
being the one a process asks for first. Each runs in a subprocess for that
reason.

They exercise one order, not nine. Importing anything under
``iris.fileformats`` runs that package's ``__init__`` first, and that imports
the netCDF package, so every case here takes the same path.
"""

import subprocess
import sys

import pytest

MODULES = [
    "iris.fileformats.cf",
    "iris.fileformats.cf.dataset",
    "iris.fileformats.cf.loader",
    "iris.fileformats.cf.saver",
    "iris.fileformats.cf._dask_locks",
    "iris.fileformats.netcdf",
    "iris.fileformats.netcdf._dataset",
    "iris.fileformats.netcdf.loader",
    "iris.fileformats.netcdf.saver",
]


@pytest.mark.parametrize("module", MODULES)
def test_importing_it_first_works(module):
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
