"""
Controlling plot colours
========================

This example shows how to use custom colour schemes for anomaly plotting.
This demonstrates key techniques for using colour in matplotlib including:
    * defining custom colour schemes
    * non-linear mapping of data values to colours
    * continuous and discrete colour ranges
    * colour scales for pseudocolour and contour-filled plots

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
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import matplotlib.colors as mcols
import numpy as np


# Define a function to create a colour map and a data normalisation, for
# continuous logarithmic colouring in a given value range.
def log_color_controls(min_log_scale, max_scale, colour_zero='white',
                       colour_min='blue', colour_max='red',
                       colour_zero_minus=None, colour_zero_plus=None):
    """
    Create a matplotlib.colors.Colormap and a matplotlib.colors.Normalize to
    colour data values logarithmically.  All values of less than a certain
    absolute magnitude map into a 'zero band' of a constant colour.

    Colour mappings, as defined by the arguments:
    * Values of magnitude less than 'min_log_scale' have 'colour_zero'.
    * Values just larger than +/-'min_log_scale' have 'colour_zero_plus' or
      'colour_zero_minus' (N.B. these both default to 'zero colour').
    * Values of +/-'max_scale' have 'colour_max'/'colour_min'.

    Returns:
        a pair of (matplotlib.colors.Colormap, matplotlib.colors.Normalize)

    """
    # Transform all the colour arguments into the form of RGBA tuples.
    colour_min, colour_zero_minus, colour_zero, colour_zero_plus, colour_max \
        = [mcols.colorConverter.to_rgba(colour)
           for colour in (colour_min,
                          colour_zero_minus or colour_zero,
                          colour_zero,
                          colour_zero_plus or colour_zero,
                          colour_max)]

    # Construct the argument dictionary for a LinearSegmentedColormap, which
    # specifies ranges of colour value, and the colour at each end.
    # Note: the "zero band" occupies a range of colour numbers equivalent to
    # one decade of the logarithmic sections (see SymLogNorm code, below).
    log_range_decades = np.log10(max_scale / min_log_scale)
    half_lin_range = 0.5 / (1.0 + 2*log_range_decades)
    cmap_segs = {}
    for i_rgba, name_rgba in enumerate(('red', 'green', 'blue', 'alpha')):
        cmap_segs[name_rgba] = [
            (0.0, colour_min[i_rgba], colour_min[i_rgba]),
            (0.5 - half_lin_range,
                colour_zero_minus[i_rgba], colour_zero[i_rgba]),
            (0.5 + half_lin_range,
                colour_zero[i_rgba], colour_zero_plus[i_rgba]),
            (1.0, colour_max[i_rgba], colour_max[i_rgba])]

    # Make the colormap.
    anom_cmap = mcols.LinearSegmentedColormap('anom', cmap_segs)

    # Make a suitable Norm operator.
    # The 'linthresh' argument is a minimum magnitude below which logs are not
    # taken:  Here, this set to our 'zero band' range.
    # NOTE: there seems to be a bug in the use of the 'linscale' argument,
    # which for now this adjustment factor fixes.
    linscale_bug_factor = np.log(10) * (1.0 - 1.0/np.e)
    anom_norm = mcols.SymLogNorm(linthresh=min_log_scale,
                                 linscale=0.5 * linscale_bug_factor,
                                 vmin=-max_scale, vmax=max_scale)

    return anom_cmap, anom_norm


def main():
    # Load a sample air temperatures sequence.
    file_path = iris.sample_data_path('E1_north_america.nc')
    cube = iris.load_cube(file_path)

    # Create a sample anomaly field for one year, by subtracting a time mean.
    i_year = 122
    time_mean = cube.collapsed('time', iris.analysis.MEAN)
    anomaly = cube[i_year] - time_mean

    # Setup scaling levels and thresholds for the anomaly data plots.
    minimum_log_level, maximum_scale_level = 0.1, 3.0
    threshold_levels = np.array([-3, -1, -0.3, -0.1, 0.1, 0.3, 1, 3])

    # Calculate color controls suitable for these data levels.
    anom_cmap, anom_norm = log_color_controls(
        minimum_log_level, maximum_scale_level,
        colour_min='#0000d0', colour_max='red',
        #colour_zero_minus='paleturquoise', colour_zero_plus='lightyellow',
    )

    # Make a pseudocolor plot using this continuous colour scheme.
    mesh = iplt.pcolormesh(anomaly, cmap=anom_cmap, norm=anom_norm)
    bar = plt.colorbar(mesh, orientation='horizontal', extend='both')
    bar.set_ticks(threshold_levels)
    plt.gca().coastlines()

    # Construct a plot title explaining which years are used.
    cube_time = cube.coord('time')
    cube_years = [time.year
                  for time in cube_time.units.num2date(cube_time.points)]
    title = 'Temperature anomalies : {} against {}-{} average.'.format(
        cube_years[i_year], cube_years[0], cube_years[-1])

    plt.title(title + '\n-- A cell-filled plot with continuous colours.')
    plt.show()

    # Make a filled contour plot of the same data, with our chosen levels.
    plt.figure()
    mesh = iplt.contourf(anomaly, threshold_levels,
                         cmap=anom_cmap, norm=anom_norm,
                         extend='both'
                         )
    bar = plt.colorbar(mesh, orientation='horizontal', extend='both')
    bar.set_ticks(threshold_levels)
    plt.gca().coastlines()
    plt.title(title + '\n-- A filled-contour plot.')
    plt.show()

    # Make a pcolor plot with similarly quantised color-levels...
    plt.figure()
    # Convert the threshold levels into colour values (floats) with our
    # existing continuous Normalize.
    tick_colour_values = anom_norm(threshold_levels)
    # Average adjacent colour values to get a colour value for each level, and
    # convert these back from 0-1 values into actual colours.
    # NOTE: this calculation reproduces the colours generated by 'contour', but
    # they do not extend to the minimum and maximum of the colour scale.
    level_colour_values = 0.5 * (tick_colour_values[:-1] +
                                 tick_colour_values[1:])
    colour_values = anom_cmap(level_colour_values)

    # Make a new Colormap and Normalize to give discrete fixed colours between
    # selected threshold levels (discontinuous).
    # NOTE: a LinearSegmentedColormap could be used, but this way is simpler.
    discrete_cmap = mcols.ListedColormap(colour_values)
    discrete_norm = mcols.BoundaryNorm(threshold_levels, len(colour_values))

    # Redo the cell-filled plot with the "stepped" colour scheme.
    mesh = iplt.pcolormesh(anomaly, cmap=discrete_cmap, norm=discrete_norm)
    bar = plt.colorbar(mesh, orientation='horizontal', extend='both')
    bar.set_ticks(threshold_levels)
    plt.gca().coastlines()
    plt.title(title + '\n-- A cell-filled plot coloured by level.')
    plt.show()


if __name__ == '__main__':
    main()
