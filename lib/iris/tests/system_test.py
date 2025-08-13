# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.


"""System test module.

System test module is useful to identify if some of the key components required for
Iris are available.

The system tests can be run with ``pytest lib/iris/tests/system_test.py``.

"""

import cf_units
import numpy as np

import iris
from iris.tests import _shared_utils


class TestSystemInitial:
    def test_supported_filetypes(self, request, tmp_path):
        nx, ny = 60, 60
        data = np.arange(nx * ny, dtype=">f4").reshape(nx, ny)

        laty = np.linspace(0, 59, ny).astype("f8")
        lonx = np.linspace(30, 89, nx).astype("f8")

        def horiz_cs():
            return iris.coord_systems.GeogCS(6371229)

        cm = iris.cube.Cube(data, "wind_speed", units="m s-1")
        cm.add_dim_coord(
            iris.coords.DimCoord(
                laty, "latitude", units="degrees", coord_system=horiz_cs()
            ),
            0,
        )
        cm.add_dim_coord(
            iris.coords.DimCoord(
                lonx, "longitude", units="degrees", coord_system=horiz_cs()
            ),
            1,
        )
        cm.add_aux_coord(
            iris.coords.AuxCoord(np.array([9], "i8"), "forecast_period", units="hours")
        )
        hours_since_epoch = cf_units.Unit(
            "hours since epoch", cf_units.CALENDAR_STANDARD
        )
        cm.add_aux_coord(
            iris.coords.AuxCoord(np.array([3], "i8"), "time", units=hours_since_epoch)
        )
        cm.add_aux_coord(
            iris.coords.AuxCoord(np.array([99], "i8"), long_name="pressure", units="Pa")
        )

        filetypes = (".nc", ".pp")
        for filetype in filetypes:
            saved_tmpfile = tmp_path / f"tmp{filetype}"
            iris.save(cm, saved_tmpfile)

            new_cube = iris.load_cube(saved_tmpfile)
            _shared_utils.assert_CML(
                request, new_cube, ("system", "supported_filetype_%s.cml" % filetype)
            )

    def test_imports_general(self):
        if _shared_utils.MPL_AVAILABLE:
            import matplotlib  # noqa
        import netCDF4  # noqa
