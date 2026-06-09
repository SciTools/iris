"""Cube block plot using using iris.plot.pcolormesh()."""

import matplotlib.pyplot as plt

import iris
import iris.quickplot as qplt

# Load the data for a single value of model level number.
fname = iris.sample_data_path("hybrid_height.nc")
temperature_cube = iris.load_cube(fname, iris.Constraint(model_level_number=1))

# Draw the block plot.
qplt.pcolormesh(temperature_cube)

plt.show()
