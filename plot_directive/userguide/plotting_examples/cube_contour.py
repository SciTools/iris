import matplotlib.pyplot as plt

import iris
import iris.plot as iplt
import iris.quickplot as qplt

fname = iris.sample_data_path('air_temp.pp')

temperature_cube = iris.load_strict(fname)

# Add a contour, and put the result in a variable called contour.
contour = qplt.contour(temperature_cube)

# Get the map created by contourf
current_map = iplt.gcm()

# Add coastlines to the map
current_map.drawcoastlines()

# Add contour labels based on the contour we have just created
plt.clabel(contour)

plt.show()
