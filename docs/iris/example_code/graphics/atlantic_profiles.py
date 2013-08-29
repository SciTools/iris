"""
Oceanographic profiles and T-S diagrams
=======================================

This example demonstrates how to plot vertical profiles of different
variables in the same axes, and how to make a scatter plot of two
variables. There is an oceanographic theme but the same techniques are
equally applicable to atmospheric or other kinds of data.

The data used are profiles of potential temperature and salinity in the
Equatorial and South Atlantic, output from an ocean model.

The y-axis of the first plot produced will be automatically inverted due to the
presence of the attribute positive=down on the depth coordinate. This means
depth values intuitively increase downward on the y-axis.

"""
import iris
import iris.iterate
import iris.plot as iplt
import matplotlib.pyplot as plt


def main():

    # Load the gridded temperature and salinity data.
    fname = iris.sample_data_path('atlantic_profiles.nc')
    cubes = iris.load(fname)
    theta, = cubes.extract('sea_water_potential_temperature')
    salinity, = cubes.extract('sea_water_practical_salinity')

    # Extract profiles of temperature and salinity from a particular point in
    # the southern portion of the domain, and limit the depth of the profile
    # to 1000m.
    lon_cons = iris.Constraint(longitude=330.5)
    lat_cons = iris.Constraint(latitude=lambda l: -10 < l < -9)
    depth_cons = iris.Constraint(depth=lambda d: d <= 1000)
    theta_1000m = theta.extract(depth_cons & lon_cons & lat_cons)
    salinity_1000m = salinity.extract(depth_cons & lon_cons & lat_cons)

    # Plot these profiles on the same set of axes. In each case we call plot
    # with two arguments, the cube followed by the depth coordinate. Putting
    # them in this order places the depth coordinate on the y-axis.
    # The first plot is in the default axes. We'll use the same color for the
    # curve and its axes/tick labels.
    fig = plt.figure(figsize=(5, 6))
    temperature_color = (.3, .4, .5)
    ax1 = plt.gca()
    iplt.plot(theta_1000m, theta_1000m.coord('depth'), linewidth=2,
              color=temperature_color, alpha=.75)
    ax1.set_xlabel('Potential Temperature / K', color=temperature_color)
    ax1.set_ylabel('Depth / m')
    for ticklabel in ax1.get_xticklabels():
        ticklabel.set_color(temperature_color)
    # To plot salinity in the same axes we use twiny(). We'll use a different
    # color to identify salinity.
    salinity_color = (.6, .1, .15)
    ax2 = plt.gca().twiny()
    iplt.plot(salinity_1000m, salinity_1000m.coord('depth'), linewidth=2,
              color=salinity_color, alpha=.75)
    ax2.set_xlabel('Salinity / PSU', color=salinity_color)
    for ticklabel in ax2.get_xticklabels():
        ticklabel.set_color(salinity_color)
    plt.tight_layout()
    plt.show()

    # Now plot a T-S diagram using scatter. We'll use all the profiles here,
    # and each point will be coloured according to its depth.
    plt.figure(figsize=(6, 6))
    depth_values = theta.coord('depth').points
    for s, t in iris.iterate.izip(salinity, theta, coords='depth'):
        iplt.scatter(s, t, c=depth_values, marker='+', cmap='RdYlBu_r')
    ax = plt.gca()
    ax.set_xlabel('Salinity / PSU')
    ax.set_ylabel('Potential Temperature / K')
    cb = plt.colorbar(orientation='horizontal')
    cb.set_label('Depth / m')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
