import matplotlib.pyplot as plt

import iris
import iris.quickplot as qplt
import iris.plot as iplt

fname = iris.sample_data_path('air_temp.pp')
temperature_cube = iris.load_cube(fname)

# extract the left-hand half only, to avoid a pcolormesh problem
temperature_cube = temperature_cube[:,:48]

# put bounds on the latitude and longitude coordinates
temperature_cube.coord('latitude').guess_bounds()
temperature_cube.coord('longitude').guess_bounds()

# draw block plot
qplt.pcolormesh(temperature_cube)

# add coastlines on the map created by qplt
plt.gca().coastlines()

plt.show()
