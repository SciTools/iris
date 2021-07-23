# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Common code for benchmarks."""

import os
from pathlib import Path

# Environment variable names
_ASVDIR_VARNAME = "ASV_DIR"  # As set in nightly script "asv_nightly/asv.sh"
_DATADIR_VARNAME = "BENCHMARK_DATA"  # For local runs

ARTIFICIAL_DIM_SIZE = int(10e3)  # For all artificial cubes, coords etc.

# Work out where the benchmark data dir is.
asv_dir = os.environ.get("ASV_DIR", None)
if asv_dir:
    # For an overnight run, this comes from the 'ASV_DIR' setting.
    benchmark_data_dir = Path(asv_dir) / "data"
else:
    # For a local run, you set 'BENCHMARK_DATA'.
    benchmark_data_dir = os.environ.get(_DATADIR_VARNAME, None)
    if benchmark_data_dir is not None:
        benchmark_data_dir = Path(benchmark_data_dir)


def testdata_path(*path_names):
    """
    Return the path of a benchmark test data file.

    These are based from a test-data location dir, which is either
    ${}/data (for overnight tests), or ${} for local testing.

    If neither of these were set, an error is raised.

    """.format(
        _ASVDIR_VARNAME, _DATADIR_VARNAME
    )
    if benchmark_data_dir is None:
        msg = (
            "Benchmark data dir is not defined : "
            'Either "${}" or "${}" must be set.'
        )
        raise (ValueError(msg.format(_ASVDIR_VARNAME, _DATADIR_VARNAME)))
    path = benchmark_data_dir.joinpath(*path_names)
    path = str(path)  # Because Iris doesn't understand Path objects yet.
    return path
