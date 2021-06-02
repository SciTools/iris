import matplotlib.pyplot as plt
import numpy as np

import iris.analysis
import iris.quickplot as qplt

fname = iris.sample_data_path("hybrid_height.nc")
column = iris.load_cube(fname, "air_potential_temperature")[:, 0, 0]

alt_coord = column.coord("altitude")

# Interpolate the "perfect" linear interpolation. Really this is just
# a high number of interpolation points, in this case 1000 of them.
altitude_points = [
    (
        "altitude",
        np.linspace(min(alt_coord.points), max(alt_coord.points), 1000),
    )
]
scheme = iris.analysis.Linear()
linear_column = column.interpolate(altitude_points, scheme)

# Now interpolate the data onto 10 evenly spaced altitude levels,
# as we did in the example.
altitude_points = [("altitude", np.linspace(400, 1250, 10))]
scheme = iris.analysis.Linear()
new_column = column.interpolate(altitude_points, scheme)

plt.figure(figsize=(5, 4), dpi=100)

# Plot the black markers for the original data.
qplt.plot(
    column,
    marker="o",
    color="black",
    linestyle="",
    markersize=3,
    label="Original values",
    zorder=2,
)

# Plot the gray line to display the linear interpolation.
qplt.plot(
    linear_column,
    color="gray",
    label="Linear interpolation",
    zorder=0,
)

# Plot the red markers for the new data.
qplt.plot(
    new_column,
    marker="D",
    color="red",
    linestyle="",
    label="Interpolated values",
    zorder=1,
)

ax = plt.gca()
# Space the plot such that the labels appear correctly.
plt.subplots_adjust(left=0.17, bottom=0.14)

# Limit the plot to a maximum of 5 ticks.
ax.xaxis.get_major_locator().set_params(nbins=5)

# Prevent matplotlib from using "offset" notation on the xaxis.
ax.xaxis.get_major_formatter().set_useOffset(False)

# Put some space between the line and the axes.
ax.margins(0.05)

# Place gridlines and a legend.
ax.grid()
plt.legend(loc="lower right")

plt.show()
