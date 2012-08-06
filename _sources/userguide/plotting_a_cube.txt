.. _introduction_to_plotting_in_iris:

==================================
Introduction to plotting in Iris
==================================

Iris utilises the power of Python's `Matplotlib <http://matplotlib.sourceforge.net/>`_ package in order to generate high 
quality, production ready 1D and 2D plots. 
The functionality of the Matplotlib `pyplot <http://matplotlib.sourceforge.net/api/pyplot_api.html>`_ module has been 
extended within Iris to facilitate easy visualisation of a cube's data.

Matplotlib's pyplot has been modelled on the MATLAB framework, therefore users of MATLAB may find a degree of 
familiarity with the interface.


***************************
Matplotlib's pyplot basics
***************************

A simple line plot can created using the :py:func:`matplotlib.pyplot.plot` function::

	import matplotlib.pyplot as plt
	plt.plot([1, 2, 2.5])
	plt.show()

This code will automatically create a figure with appropriate axes for the plot and show it on screen. 
The call to **plt.plot([1, 2, 2.5])** will create a line plot with appropriate axes for the data (x=0, y=1; x=1, y=2; x=2, y=2.5). 
The call to **plt.show()** tells Matplotlib that you have finished with this plot and that you would like to visualise it in a window. 
This is an example of using matplotlib in *non-interactive* mode.

There are two modes of rendering within Matplotlib; **interactive** and **non-interactive**.


Interactive plot rendering
==========================
The previous example was *non-interactive* as the figure is only rendered *after* the call to :py:func:`plt.show() <matplotlib.pyplot.show>`. 
Rendering plots *interactively* can be achieved by changing the interactive mode::

	import matplotlib.pyplot as plt
	plt.interactive(True)
	plt.plot([1, 2, 2.5])

In this case the plot is rendered automatically with no need to explicitly call :py:func:`matplotlib.pyplot.show` after **plt.plot**. 
Subsequent changes to your figure will be automatically rendered in the window. 

The current rendering mode can be determined as follows::

	import matplotlib.pyplot as plt
	print plt.isinteractive()

.. note:: 
	For clarity, each example includes all of the imports required to run on its own; when combining 
	examples such as the two above, it would not be necessary to repeat the import statement more than once::

        	import matplotlib.pyplot as plt
	        plt.interactive(True)
	        plt.plot([1, 2, 2.5])
	        print plt.isinteractive()

Interactive mode does not clear out the figure buffer, so figures have to be explicitly closed when they are finished with::

        plt.close()

Interactive mode sometimes requires an extra draw command to update all changes, which can be done with::

        plt.draw()

For the remainder of this tutorial we will work in non-interactive mode, so ensure that interactive mode is turned off with::

        plt.interactive(False)

Saving a plot
=============

The :py:func:`matplotlib.pyplot.savefig` function is similar to **plt.show()** in that they are both *non-interactive* visualisation modes. 
As you might expect, **plt.savefig** saves your figure as an image::

	import matplotlib.pyplot as plt
	plt.plot([1, 2, 2.5])
	plt.savefig('plot123.png')

The filename extension passed to the :py:func:`matplotlib.pyplot.savefig` function can be used to control 
the output file format of the plot (keywords can also be used to control this and other aspects, 
see :py:func:`matplotlib.pyplot.savefig`). 

Some of the formats which are supported by **plt.savefig**:

	======  ======  ======================================================================
	Format  Type    Description
	======  ======  ======================================================================
	EPS     Vector  Encapsulated PostScript
	PDF     Vector  Portable Document Format
	PNG     Raster  Portable Network Graphics, a format with a lossless compression method
	PS      Vector  Post Script, ideal for printer output
	SVG     Vector  Scalable Vector Graphics, XML based
	======  ======  ======================================================================

******************
Iris cube plotting
******************

The Iris modules :py:mod:`iris.quickplot` and :py:mod:`iris.plot` extend the Matplotlib pyplot interface 
by implementing thin *wrapper* functions. 
These wrapper functions simply bridge the gap between an Iris cube and the data expected by standard 
Matplotlib pyplot functions. 
This means that *all* Matplotlib pyplot functionality, including keyword options, are still available 
through the Iris plotting wrapper functions.

As a rule of thumb:

 * if you wish to do a visualisation with a cube, use ``iris.plot`` or ``iris.quickplot``.
 * if you wish to show, save or manipulate **any** visualisation, including ones created with Iris, use ``matplotlib.pyplot``.
 * if you wish to create a non cube visualisation, also use ``matplotlib.pyplot``.

The ``iris.quickplot`` module is exactly the same as the ``iris.plot`` module, except that ``quickplot`` will add a 
title, x and y labels and a colorbar where appropriate.

.. note::
   In all subsequent examples the ``matplotlib.pyplot``, ``iris.plot`` and ``iris.quickplot`` modules are 
   imported as ``plt``, ``iplt`` and ``qplt`` respectively in order to make the code more readable.
   This is equivalent to::

       import matplotlib.pyplot as plt
       import iris.plot as iplt
       import iris.quickplot as qplt


Plotting 1-dimensional cubes 
============================

The simplest 1D plot is achieved with the :py:func:`iris.plot.plot` function. The syntax is very similar to that 
which you would provide to Matplotlib's equivalent :py:func:`matplotlib.pyplot.plot` and indeed all of the 
keyword arguments are equivalent:

.. literalinclude:: plotting_examples/1d_simple.py

.. plot:: userguide/plotting_examples/1d_simple.py

For more information on how this example reduced the 2D cube to 1 dimension see the previous section 
entitled :doc:`reducing_a_cube`.

.. note:: 
    Axis labels and a plot title can be added using the :func:`plt.title() <matplotlib.pyplot.title>`, 
	:func:`plt.xlabel() <matplotlib.pyplot.xlabel>` and :func:`plt.ylabel <matplotlib.pyplot.ylabel>` functions.

As well as providing simple Matplotlib wrappers, Iris also has a :py:mod:`iris.quickplot` module, which adds 
extra cube based meta-data to a plot. 
For example, the previous plot can be improved quickly by replacing **iris.plot** with **iris.quickplot**:

.. literalinclude:: plotting_examples/1d_quickplot_simple.py

.. plot:: userguide/plotting_examples/1d_quickplot_simple.py



Multi-line plot
---------------

A multi-lined (or over-plotted) plot, with a legend, can be achieved easily by calling :func:`iris.plot.plot` 
or :func:`iris.quickplot.plot` consecutively and providing the label keyword to identify it.
Once all of the lines have been added the :func:`matplotlib.pyplot.legend` function can be called 
to indicate that a legend is desired: 

.. literalinclude:: ../../example_code/graphics/lineplot_with_legend.py

.. plot:: ../example_code/graphics/lineplot_with_legend.py

This example of consecutive ``qplt.plot`` calls coupled with the :func:`Cube.slices() <iris.cube.Cube.slices>` 
method on a cube shows the temperature at some latitude cross-sections. 

.. note::
    The previous example uses the ``if __name__ == "__main__"`` style to run the desired code if and only if the 
    script is run from the command line.

    This is a good habit to get into when writing scripts in Python as it means that any useful functions or 
    variables defined within the script can be imported into other scripts without running all of the code and 
    thus creating an unwanted plot. This is discussed in more detail at `<http://effbot.org/pyfaq/tutor-what-is-if-name-main-for.htm>`_.

    In order to run this example, you will need to copy the code into a file and run it using ``python2.7 my_file.py``.


Plotting 2-dimensional cubes
============================

Creating maps
-------------
Whenever a 2D plot is created and the x and y coordinates are longitude and latitude a 
:class:`mpl_toolkits.basemap.Basemap` instance is created which can be accessed with the :func:`iris.plot.gcm` function. 

Given the current map, you can draw meridians, parallels and coastlines amongst other things. 

.. seealso:: 
	:meth:`Basemap.drawmeridians() <mpl_toolkits.basemap.Basemap.drawmeridians>`, 
	:meth:`Basemap.drawparallels() <mpl_toolkits.basemap.Basemap.drawparallels>` and 
	:meth:`Basemap.drawcoastlines() <mpl_toolkits.basemap.Basemap.drawcoastlines>`.


Cube contour
------------
A simple contour plot of a cube can be created with either the :func:`iris.plot.contour` or :func:`iris.quickplot.contour` functions:

.. literalinclude:: plotting_examples/cube_contour.py

.. plot:: userguide/plotting_examples/cube_contour.py


Cube filled contour
-------------------
Similarly a filled contour plot of a cube can be created with the :func:`iris.plot.contourf` or :func:`iris.quickplot.contourf` functions:

.. literalinclude:: plotting_examples/cube_contourf.py

.. plot:: userguide/plotting_examples/cube_contourf.py


Cube block plot
---------------
Both ``contour`` and ``contourf`` are point based visualisations in that for both the x and y plot axes the 
coordinates must have :attr:`Coord.points <iris.coords.Coord.points>`. 
In some situations the underlying coordinates are not point based and instead are better represented with a 
continuous bounded coordinate, in which case a "block" plot may be more appropriate.
Continuous block plots can be achieved with either :func:`iris.plot.pcolormesh` or :py:func:`iris.quickplot.pcolormesh`:

.. literalinclude:: plotting_examples/cube_blockplot.py

.. plot:: userguide/plotting_examples/cube_blockplot.py
