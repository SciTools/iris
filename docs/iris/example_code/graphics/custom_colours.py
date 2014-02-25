"""
Controlling plot colours
========================

This example shows how to use custom colour schemes for anomaly plotting,
demonstrating some key matplotlib techniques including:
    * defining custom colour schemes
    * non-linear mapping of data values to colours
    * continuous and discrete colour ranges
    * colour scales for pseudocolour and contour-filled plots

NOTE: "pseudocolour" means that each data point is drawn as a "cell" region on
the plot, coloured according to its data value.  In matplotlib, this is done
with the methods `matplotlib.pyplot.pcolor
<http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.pcolor>`_
and `matplotlib.pyplot.pcolormesh
<http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.pcolormesh>`_ .
See also : http://en.wikipedia.org/wiki/False_color#Pseudocolor.

In this case, we want to colour signed data "logarithmically" -- i.e. we have
values both above and below zero, and we want to have an equal range of colour
between data values of, say, 1 and 10 as between 10 and 100.
This involves making custom colour maps to suit the requirements of the plot
type and the data range.

To do this, we construct a custom colour scheme and value mapping function
(normalization), and use these to produce a cell-filled pseudocolour plot with
continuously varying colours.

We then display the data as a filled-contour plot with the same colour scheme.

Finally, we produce a cell-filled plot with a "stepped" (discontinuous) colour
scale, which matches the levels and colours in the contoured example.

"""
import iris
import iris.plot as iplt
import matplotlib.pyplot as plt
import matplotlib.colors as mcols
import numpy as np


# Define a function to create a colour map and data normalisation.
def log_color_controls(min_log_scale, max_scale, colour_zero='white',
                       colour_min='blue', colour_max='red',
                       colour_zero_minus=None, colour_zero_plus=None):
    # Returns: (matplotlib.colors.Colormap, matplotlib.colors.Normalize).
    #
    # In terms of the arguments, the colour is a piecewise function:
    #  * Values of magnitude less than 'min_log_scale' have 'colour_zero'.
    #  * Values just larger than +/-'min_log_scale' have 'colour_zero_plus' and
    #    'colour_zero_minus' (N.B. these both default to 'colour_zero').
    #  * Values of +/-'max_scale' have 'colour_max' and 'colour_min'.

    # Transform all the colour arguments into the form of RGBA tuples.
    rgba_tuples = [mcols.colorConverter.to_rgba(colour)
                   for colour in (colour_min,
                                  colour_zero_minus or colour_zero,
                                  colour_zero,
                                  colour_zero_plus or colour_zero,
                                  colour_max)]
    (colour_min, colour_zero_minus, colour_zero, colour_zero_plus,
     colour_max) = rgba_tuples

    # Calculate the range of colour-values allocated to the "zero band": make
    # this equal to one decade of the logarithmic sections, as specified by the
    # 'linscale' argument to SymLogNorm (see in code below).
    log_range_decades = np.log10(max_scale / min_log_scale)
    half_lin_range = 0.5 / (1.0 + 2*log_range_decades)
    # Construct a dictionary argument to make a LinearSegmentedColormap.
    cmap_segs = {}
    for i_rgba, name_rgba in enumerate(('red', 'green', 'blue', 'alpha')):
        cmap_segs[name_rgba] = [
            # At 0.0, 'Negatives' segment starts : colour --> colour_min.
            (0.0, colour_min[i_rgba], colour_min[i_rgba]),
            # At 0.5 - "delta", end of 'Negatives' --> colour_zero_minus,
            # then becoming start of 'zero band' --> colour_zero.
            (0.5 - half_lin_range,
                colour_zero_minus[i_rgba], colour_zero[i_rgba]),
            # At 0.5 + "delta", end of 'zero band' --> colour_zero,
            # then becoming start of 'Positives' --> colour_zero_plus.
            (0.5 + half_lin_range,
                colour_zero[i_rgba], colour_zero_plus[i_rgba]),
            # At 1.0, 'Positives' segment ends : colour --> colour_max.
            (1.0, colour_max[i_rgba], colour_max[i_rgba])]

    # NOTE: there is a matplotlib example with much more detail on this.
    # See: http://matplotlib.org/examples/pylab_examples/custom_cmap.html.

    # Create the colormap.
    anom_cmap = mcols.LinearSegmentedColormap('anom', cmap_segs)

    # Create a suitable matching Norm operator (a special "SymLogNorm" type).
    # NOTE: there seems to be a bug in the use of the 'linscale' argument,
    # which the following adjustment factor fixes (for now).
    linscale_bug_factor = np.log(10) * (1.0 - 1.0/np.e)
    anom_norm = mcols.SymLogNorm(linthresh=min_log_scale,
                                 linscale=0.5 * linscale_bug_factor,
                                 vmin=-max_scale, vmax=max_scale)
    # NOTE: 'linthresh' sets the non-logarithmic region to our 'zero band'.
    # NOTE: 'linscale' makes the non-log region one decade wide (as above).

    return anom_cmap, anom_norm


# Define a convenience function for making a pseudocolour plot.
def plot_pseudocolour(data, tick_levels, cmap, norm, title):
    mesh = iplt.pcolormesh(data, cmap=cmap, norm=norm)
    bar = plt.colorbar(mesh, orientation='horizontal', extend='both')
    bar.set_ticks(tick_levels)
    plt.gca().coastlines()
    plt.title(title)


# Define a convenience function for making a filled-contour plot.
def plot_filled_contours(data, tick_levels, cmap, norm, title):
    mesh = iplt.contourf(data, tick_levels, cmap=cmap, norm=norm,
                         extend='both')
    bar = plt.colorbar(mesh, orientation='horizontal')
    plt.gca().coastlines()
    plt.title(title)


def main():
    # Load a sample air temperatures sequence.
    file_path = iris.sample_data_path('E1_north_america.nc')
    cube = iris.load_cube(file_path)

    # Create a sample anomaly field for one year, by subtracting a time mean.
    i_year = 122
    time_mean = cube.collapsed('time', iris.analysis.MEAN)
    anomaly = cube[i_year] - time_mean

    # Construct a plot title string to explain which years are used.
    cube_time = cube.coord('time')
    cube_years = [time.year
                  for time in cube_time.units.num2date(cube_time.points)]
    title = '{} difference from {}-{} average.'.format(
        cube_years[i_year], cube_years[0], cube_years[-1])

    # Setup scaling levels and thresholds for the anomaly data plots.
    minimum_log_level, maximum_scale_level = 0.1, 3.0
    threshold_levels = np.array([-3, -1, -0.3, -0.1, 0.1, 0.3, 1, 3])

    # Calculate suitable color controls for these data levels.
    anom_cmap, anom_norm = log_color_controls(
        minimum_log_level, maximum_scale_level,
        colour_min='#0000d0', colour_max='red',
        colour_zero_minus='#c0ffff', colour_zero_plus='#ffffc0',
    )

    # Make a pseudocolour plot using this continuous colour scheme.
    plt.figure(figsize=(14, 5))
    plt.subplots_adjust(left=0.025, right=0.975, wspace=0.05)
    plt.subplot(131)
    plot_pseudocolour(anomaly, threshold_levels, anom_cmap, anom_norm,
                      title + '\nCell-filled plot with continuous colours.')

    # Make a filled contour plot of the same data, with our chosen levels.
    plt.subplot(132)
    plot_filled_contours(anomaly, threshold_levels, anom_cmap, anom_norm,
                         title + '\nFilled-contour plot.')

    # To make a pcolor plot with similarly quantised color-levels...

    # First, get a colour value (0-1) for each threshold level, using our
    # existing (continuous) norm.
    tick_colour_values = anom_norm(threshold_levels)

    # Average adjacent colour values, to get one for each contour level.
    level_colour_values = 0.5 * (tick_colour_values[:-1] +
                                 tick_colour_values[1:])
    # NOTE: this calculation reproduces the colours chosen by 'contour', but
    # these do not extend to the minimum and maximum of the colour scale.

    # Convert back into actual colours with the existing (continuous) map.
    colour_values = anom_cmap(level_colour_values)

    # Make a _new_ Colormap and Normalize, programmed to produce these exact
    # colours between the appropriate threshold levels (discontinuously).
    discrete_cmap = mcols.ListedColormap(colour_values)
    discrete_norm = mcols.BoundaryNorm(threshold_levels, len(colour_values))
    # NOTE: a LinearSegmentedColormap could be used, but this way is simpler.
    # See BoundayNorm documentation, at:
    # http://matplotlib.org/api/colors_api.html#matplotlib.colors.BoundaryNorm

    # Make another cell-filled plot with the discontinous colour scheme.
    plt.subplot(133)
    plot_pseudocolour(anomaly, threshold_levels, discrete_cmap, discrete_norm,
                      title + '\nCell-filled plot coloured by level.')

    # Display results.
    plt.show()


if __name__ == '__main__':
    main()
