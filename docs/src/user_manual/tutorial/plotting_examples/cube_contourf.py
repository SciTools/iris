# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Simple filled contour plot of a cube.

Can use iris.plot.contour() or iris.quickplot.contour().

"""

import matplotlib.pyplot as plt

import iris
import iris.quickplot as qplt

fname = iris.sample_data_path("air_temp.pp")
temperature_cube = iris.load_cube(fname)

# Draw the contour with 25 levels.
qplt.contourf(temperature_cube, 25)

# Add coastlines to the map created by contourf.
plt.gca().coastlines()

plt.show()
