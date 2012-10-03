import matplotlib.pyplot as plt

import iris
import iris.quickplot as qplt
import iris.plot as iplt

fname = iris.sample_data_path('air_temp.pp')
temperature_cube = iris.load_cube(fname)

# put bounds on the latitude and longitude coordinates
temperature_cube.coord('latitude').guess_bounds()
temperature_cube.coord('longitude').guess_bounds()

# Draw the contour with 25 levels
qplt.pcolormesh(temperature_cube)

# Add coastlines to the map created by pcolormesh
plt.gca().coastlines()

plt.show()
