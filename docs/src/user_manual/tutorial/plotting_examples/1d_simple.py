# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Simple 1D plot using iris.plot.plot()."""

import matplotlib.pyplot as plt

import iris
import iris.plot as iplt

fname = iris.sample_data_path("air_temp.pp")
temperature = iris.load_cube(fname)

# Take a 1d slice using array style indexing.
temperature_1d = temperature[5, :]

iplt.plot(temperature_1d)

plt.show()
