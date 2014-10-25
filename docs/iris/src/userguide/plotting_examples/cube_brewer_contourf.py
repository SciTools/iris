
from __future__ import (absolute_import, division, print_function)

import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt

import iris
import iris.quickplot as qplt

fname = iris.sample_data_path('air_temp.pp')
temperature_cube = iris.load_cube(fname)

# Load a Cynthia Brewer palette.
brewer_cmap = mpl_cm.get_cmap('brewer_OrRd_09')

# Draw the contours, with n-levels set for the map colours (9).
# NOTE: needed as the map is non-interpolated, but matplotlib does not provide
# any special behaviour for these.
qplt.contourf(temperature_cube, brewer_cmap.N, cmap=brewer_cmap)

# Add coastlines to the map created by contourf.
plt.gca().coastlines()

plt.show()
