"""
Adding an annotation to a plot
==============================

This example demonstrates adding annotations (such as date and time) to a plot.

This is achieved through adding an axes to the figure that displays useful
information such as the date and time of plot creation.

"""
import datetime

import matplotlib.pyplot as plt
import numpy as np

import iris
import iris.quickplot as qplt


def annotate():
    """
    Annotate the current figure by adding an axes displaying the current date
    and time.

    """
    # Create a new axes for the annotation.
    fig = plt.gcf()
    kwargs = {'navigate': False, 'frameon': False}
    # Axes location is defined as (left, bottom, width, height).
    location = (0., 0., 0.5, 0.1)
    annotation_axes = fig.add_axes(location, **kwargs)
    annotation_axes.set_xticks([])
    annotation_axes.set_yticks([])

    # Get annotation information.
    now = datetime.datetime.utcnow()
    annotation = '{:%Y-%m-%d %H:%M}'.format(now)

    # Add annotation to axes.
    annotation_axes.text(0.01, 0.1, annotation, fontsize=8)


def main():
    """Load and plot a cube to create a figure to annotate."""
    cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
    qplt.pcolormesh(cube)
    annotate()
    qplt.show()


if __name__ == '__main__':
    main()
