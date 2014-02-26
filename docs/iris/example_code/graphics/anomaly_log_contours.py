"""
Contouring anomaly data with logarithmic levels
===============================================

In this example, we need to plot anomaly data where the values have a
"logarithmic" significance  -- i.e. we want to give approximately equal ranges
of colour between data values of, say, 1 and 10 as between 10 and 100.

In many similar cases, as long as the required data range is symmetrical about
zero, it is perfectly practical to simply select a suitable colormap and allow
'contourf' to pick the level colours automatically from that.
(The colormap needs to be of the "diverging" style:
for suitable options, see the 'Diverging colormaps' section in:
`<http://matplotlib.org/examples/color/colormaps_reference.html>`_).

In this case, however, that approach would not allow for our log-scaling
requirement.  Whilst this can be overcome with an alternative type of
`matplotlib.colors.Normalize
<http://matplotlib.org/api/colors_api.html#matplotlib.colors.Normalize>`_, the
resulting code can become rather complex.
Also, and perhaps more importantly, it turns out that the resulting colours are
not precisely what one might expect (or want).

Therefore in this example we have chosen instead to define the contour layer
colours *explicitly*, using a helper function to calculate the shading tones.
This is an approach with a very much more general applicability:  In this case,
for instance, it is much more easily adapted to slightly altered requirements
such as unequal positive and negative scales.
Using this method, the "logarithmic" shading requirement is provided simply by
the appropriate choice of contouring levels.

"""
import iris
import iris.plot as iplt
import matplotlib.pyplot as plt
import matplotlib.colors as mcols
import numpy as np


# Define a function to construct suitable graduated colours in two shades.
def make_anomaly_colours(n_layers, colour_minus='blue', colour_plus='red',
                         colour_zero='white'):
    # Calculate colours for anomaly plot contours, interpolated from a central
    # 'colour_zero' to extremes of 'colour_minus' and 'colour_plus'.
    #
    # Returns an iterable of 'n_layers' colours, consisting of equal numbers of
    # negative and positive tones, plus a central value of 'colour_zero'.
    # That is, both upper and lower portions contain (n_layers - 1)/2 colours.
    # Thus, 'n_layers' should always be *odd*.

    # Convert the three keypoint colour specifications into RGBA value arrays.
    key_colours = (colour_minus, colour_zero, colour_plus)
    key_colours = [mcols.colorConverter.to_rgba(colour)
                   for colour in key_colours]
    key_colours = [np.array(colour) for colour in key_colours]

    # Calculate the blending step fractions.
    n_steps_oneside = (n_layers - 1)/2
    layer_fractions = np.linspace(0.0, 1.0, n_steps_oneside, endpoint=False)

    # Add extra *1 dimensions so we can multiply the colours by the fractions.
    colour_minus, colour_zero, colour_plus = [colour.reshape((1, 4))
                                              for colour in key_colours]
    layer_fractions = layer_fractions.reshape((n_steps_oneside, 1))

    # Calculate the upper and lower colour value sequences.
    colours_low = colour_minus + layer_fractions*(colour_zero - colour_minus)
    # NOTE: low colours from exactly colour_minus to "nearly" colour_zero.
    colours_high = colour_plus + layer_fractions*(colour_zero - colour_plus)
    # NOTE: high colours from exactly colour_plus to "nearly" colour_zero -- so
    # these ones are 'upside down' relative to the intended result order.

    # Join the two ends with the middle, and return this as the result.
    layer_colours = np.concatenate((colours_low,
                                    colour_zero,
                                    colours_high[::-1]))
    return layer_colours


def main():
    # Load a sample air temperatures sequence.
    file_path = iris.sample_data_path('E1_north_america.nc')
    temperatures = iris.load_cube(file_path)

    # Create a sample anomaly field for one year, by subtracting a time mean.
    i_year = 122
    time_mean = temperatures.collapsed('time', iris.analysis.MEAN)
    anomaly = temperatures[i_year] - time_mean

    # Construct a plot title string explaining which years are involved.
    times = temperatures.coord('time')
    cube_years = [time.year for time in times.units.num2date(times.points)]
    title = 'Temperature anomaly [{}, log scale]'.format(anomaly.units)
    title += '\n{} differences from {}-{} average.'.format(
        cube_years[i_year], cube_years[0], cube_years[-1])

    # Define the levels we want to contour with.
    # NOTE: these will also appear as the colorbar ticks.
    contour_levels = [-3.0, -1, -0.3, -0.1, 0.1, 0.3, 1, 3]

    # calculate a suitable set of graduated colour values.
    layer_colours = make_anomaly_colours(n_layers=len(contour_levels) - 1,
                                         colour_minus='#0040c0',
                                         colour_plus='darkred')

    # Make a contour plot with these levels and colours.
    contours = iplt.contourf(anomaly, contour_levels,
                             colors=layer_colours,
                             extend='both')
    # NOTE: Setting "extend=both" means that out-of-range values are coloured
    # with the min and max colours.

    # Add a colourbar.
    plt.colorbar(contours, orientation='horizontal')
    # NOTE: This picks up the 'extend=both' from the plot, automatically
    # showing how out-of-range values are handled.

    # Add coastlines and a title.
    plt.gca().coastlines()
    plt.title(title)

    # Display the result.
    plt.show()


if __name__ == '__main__':
    main()
