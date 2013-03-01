"""
Rotated pole mapping 
=====================

This example uses several visualisation methods to achieve an array of differing images, including:

 * Visualisation of point based data
 * Contouring of point based data
 * Block plot of contiguous bounded data
 * Non native projection and a Natural Earth shaded relief image underlay

"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import iris
import iris.plot as iplt
import iris.quickplot as qplt
import iris.analysis.cartography


def main():
    fname = iris.sample_data_path('rotated_pole.nc')
    temperature = iris.load_cube(fname)
    
    # Plot #1: Point plot showing data values & a colorbar
    plt.figure()
    points = qplt.points(temperature, c=temperature.data)
    cb = plt.colorbar(points, orientation='horizontal')
    cb.set_label(temperature.units)
    plt.gca().coastlines()
    plt.show()
    
    
    # Plot #2: Contourf of the point based data
    plt.figure()
    qplt.contourf(temperature, 15)
    plt.gca().coastlines()
    plt.show()
    
    
    # Plot #3: Contourf overlayed by coloured point data
    plt.figure()
    qplt.contourf(temperature)
    iplt.points(temperature, c=temperature.data)
    plt.gca().coastlines()
    plt.show()
    
    
    
    # For the purposes of this example, add some bounds to the latitude and longitude
    temperature.coord('grid_latitude').guess_bounds()
    temperature.coord('grid_longitude').guess_bounds()
    
    
    # Plot #4: Block plot
    plt.figure()
    plt.axes(projection=ccrs.PlateCarree())
    iplt.pcolormesh(temperature)
    plt.gca().stock_img()
    plt.gca().coastlines()
    plt.show()


if __name__ == '__main__':
    main()
