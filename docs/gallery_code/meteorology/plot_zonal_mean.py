"""
Zonal Mean Diagram of Air Temperature
=====================================

This example demonstrates aligning a linear plot and a cartographic plot using Matplotlib.

"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import iris
from iris.analysis import MEAN
import iris.plot as iplt


def main():
    fname = iris.sample_data_path("air_temp.pp")
    temperature = iris.load_cube(fname)
    collapsed_temp = temperature.collapsed("longitude", iris.analysis.MEAN)

    # Set y axes with -90 and 90 limits and spacing of 15 per tick.
    yticks = np.arange(-90, 105, 15)
    ylim = [-90, 90]
    fig = plt.figure(figsize=[12, 4])
    ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree())
    plt.sca(ax1)
    im = iplt.contourf(temperature, cmap="RdYlBu_r")
    ax1.coastlines()
    ax1.gridlines()
    ax1.set_xticks([-180, -90, 0, 90, 180], crs=ccrs.PlateCarree())
    ax1.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax1.set_title('Air Temperature')
    ax1.set_ylabel('latitude')
    ax1.set_xlabel('longitude')
    ax1.set_ylim(*ylim)
    divider = make_axes_locatable(ax1)

    # Gives the air temperature bar size, colour and a title.
    ax2 = divider.new_vertical(size="5%", pad=0.5, axes_class=plt.Axes, pack_start=True)
    fig.add_axes(ax2)
    plt.sca(ax2)
    cbar = plt.colorbar(im, cax=ax2, orientation='horizontal')
    cbar.ax.set_xlabel('Air Temperature [k]')

    # Round each tick for the third ax to the nearest 20 (ready for use).
    data_max = collapsed_temp.data.max()
    x_max = data_max - data_max % -20
    data_min = collapsed_temp.data.min()
    x_min = data_min - data_min % 20

    # Plot "collapsed_temp" on the mean graph and set the ticks and titles on the axes.
    ax3 = divider.new_horizontal(size="30%", pad=0.4, axes_class=plt.Axes)
    fig.add_axes(ax3)
    plt.sca(ax3)
    iplt.plot(collapsed_temp, collapsed_temp.coord(latitude))
    ax3.axvline(0, color='k', linewidth=0.5)
    ax3.set_ylim(*ylim)
    ax3.set_title('Zonal mean')
    ax3.set_ylabel('latitude')
    ax3.set_xlabel('Air Temperature [k]')
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    ax3.set_yticks(yticks)
    ax3.set_xlim(x_min, x_max)

    plt.show()


if __name__ == "__main__":
    main()
