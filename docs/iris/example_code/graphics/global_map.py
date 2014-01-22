"""
Quickplot of a 2d cube on a map
===============================

This example demonstrates a contour plot of global air temperature.
The plot title and the labels for the axes are automatically derived from the metadata.

"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import iris
import iris.quickplot as qplt


def main():
    fname = iris.sample_data_path('air_temp.pp')
    temperature = iris.load_cube(fname)

    # Plot #1: contourf with axes longitude from -180 to 180
    plt.figure()
    qplt.contourf(temperature, 15)
    plt.gca().coastlines()
    plt.show()

    # Plot #2: contourf with axes longitude from 0 to 360
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=-180.0))
    qplt.contourf(temperature, 15)
    plt.gca().coastlines()
    plt.show()

if __name__ == '__main__':
    main()
