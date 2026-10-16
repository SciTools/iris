# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""The CF and netCDF packages import from cold.

The two packages import each other: the CF reader builds a netCDF dataset, and
the netCDF saver builds on the CF saver. Nothing in the source states the order
the module bodies must run in.

One import covers every module in both packages. Python runs
``iris.fileformats.__init__`` before any submodule body, and that imports the
netCDF package, which draws in the whole cascade.

The subprocess is what makes it a cold start. Inside the test process the
modules are already in ``sys.modules``, so an import there is a dictionary
lookup.
"""

import subprocess
import sys


def test_importing_from_cold_works():
    completed = subprocess.run(
        [sys.executable, "-c", "import iris.fileformats"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
