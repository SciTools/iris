import matplotlib.pyplot as plt

import iris
import iris.plot as iplt

fname = iris.sample_data_path("air_temp.pp")

# Load exactly one cube from the given file
temperature = iris.load_cube(fname)

# We are only interested in a small number of longitudes (the 4 after and
# including the 5th element), so index them out
temperature = temperature[5:9, :]

for cube in temperature.slices("longitude"):
    # Create a string label to identify this cube (i.e. latitude: value)
    cube_label = "latitude: %s" % cube.coord("latitude").points[0]

    # Plot the cube, and associate it with a label
    iplt.plot(cube, label=cube_label)

# Match the longitude range to global
max_lon = temperature.coord("longitude").points.max()
min_lon = temperature.coord("longitude").points.min()
plt.xlim(min_lon, max_lon)

# Add the legend with 2 columns
plt.legend(ncol=2)

# Put a grid on the plot
plt.grid(True)

# Provide some axis labels
plt.ylabel("Temperature / kelvin")
plt.xlabel("Longitude / degrees")

# And a sensible title
plt.suptitle("Air Temperature", fontsize=20, y=0.9)

# Finally, show it.
plt.show()
