"""
Cross section plots
===================

This example demonstrates contour plots of a cross-sectioned multi-dimensional cube which features 
a hybrid height vertical coordinate system.

.. note:: 
   The data is limited to the first fifteen model levels to emphasise the terrain-following 
   nature of the data.

"""

import matplotlib.pyplot as plt

import iris
import iris.quickplot as qplt


def main():
    fname = iris.sample_data_path('PP', 'COLPEX', 'theta_and_orog_subset.pp')
    theta = iris.load_strict(fname, 'air_potential_temperature')
    
    # Extract a height vs longitude cross-section. N.B. This could easily changed to
    # extract a specific slice, or even to loop over *all* cross section slices.
    cross_section = theta[0, :15, 0, :]
    
    qplt.contourf(cross_section)
    plt.show()
    
    # Now do the equivalent plot, only against model level
    plt.figure()
    
    qplt.contourf(cross_section, coords=['grid_longitude', 'model_level_number'])
    plt.show()


if __name__ == '__main__':
    main()
