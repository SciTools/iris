import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt

import iris
import iris.quickplot as qplt
import iris.plot as iplt

fname = iris.sample_data_path('air_temp.pp')
temperature_cube = iris.load_cube(fname)

# Load a Cynthia Brewer palette.
brewer_cmap = mpl_cm.get_cmap('brewer_OrRd_09')

# Draw the contours, with n-levels set for the map colours (9).
# NOTE: needed for non-interpolated colormaps, as matplotlib does not check for them.
qplt.contourf(temperature_cube, brewer_cmap.N, cmap=brewer_cmap)

# Add coastlines to the map created by contourf
plt.gca().coastlines()

plt.show()
