# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Every module in the CF and netCDF packages can be imported first.

The two packages import each other: the CF reader builds a netCDF dataset, and
the netCDF saver builds on the CF saver. Python resolves that as long as no
module needs a name from a module that is still part-way through its own
imports - which depends on where the interpreter starts. A test suite only
ever starts in one place, so each case gets a subprocess of its own.
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
