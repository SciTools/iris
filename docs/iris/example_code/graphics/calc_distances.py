"""
Calculating the grace circle distance from a point
==================================================

This example demonstrates how we might use an external library (pyproj) to
calculate the distance from our points in a cube to a specific location,
using our cubes grid mapping.  We then associate this information
onto our original cube as an auxiliary coordinate.


"""
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Geod

import cartopy.crs as ccrs
import iris


def main():
    # Load the data
    fnme = iris.sample_data_path('air_temp.pp')
    cube = iris.load_cube(fnme)
    lats = cube.coord('latitude')
    lons = cube.coord('longitude')

    # Create grid of lat-lon
    xv, yv = np.meshgrid(lons.points, lats.points)
    origshp = xv.shape

    # Define the points that were calculating the distance to
    reflong = 50.
    reflat = 50.
    arr_reflong = np.ones_like(xv) * reflong
    arr_reflat = np.ones_like(yv) * reflat

    # Calculate great circle distance between points (using our ellipse)
    elipse = lons.coord_system.as_cartopy_crs().proj4_params['ellps']
    g = Geod(ellps=elipse)
    _, _, dist = g.inv(xv, yv, arr_reflong, arr_reflat)

    # Associate with the cube
    dist_coord = iris.coords.AuxCoord(
        dist, long_name='great circle distance',
        units='m', attributes={'origin':
                               'great circle distance calculated using pyproj',
                               'distance_to':
                               '({},{})'.format(reflong, reflat)})
    cube.add_aux_coord(dist_coord,
                       data_dims=[cube.coord_dims(lats)[0],
                                  cube.coord_dims(lons)[0]])

    # Plot the distance between one of these points
    plt.subplot(211, projection=ccrs.PlateCarree())
    plt.title('Distance between points')
    plt1_ax = plt.gca()

    plt1_ax.stock_img()

    ilpnt_x = xv[50, 50]
    ilpnt_y = yv[50, 50]
    plt.plot([reflong, ilpnt_x], [reflat, ilpnt_y],
             color='blue', linewidth=2, marker='o',
             transform=ccrs.Geodetic())

    plt.plot([reflong, ilpnt_x], [reflat, ilpnt_y],
             color='gray', linestyle='--',
             transform=ccrs.PlateCarree())

    plt.text(reflong + 25, reflat - 12, 'Reference lat/lon',
             horizontalalignment='right',
             transform=ccrs.Geodetic())

    plt.text(ilpnt_x + 25, ilpnt_y - 12, 'Data lat/lon',
             horizontalalignment='right',
             transform=ccrs.Geodetic())

    # Histogram plot
    plt.subplot(212)
    plt.title('Histogram of phenomena vs distance')
    plt2_ax = plt.gca()
    n, bins, patches = plt2_ax.hist(dist.flatten(), bins=50, normed=1,
                                    facecolor='green', alpha=0.75,
                                    weights=cube.data.flatten())
    plt2_ax.grid(True)

    plt.show()


if __name__ == '__main__':
    main()
