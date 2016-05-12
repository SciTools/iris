"""
Example of a polar stereographic plot
=====================================

Demonstrates plotting data that are defined on a polar stereographic
projection.

"""

import matplotlib.pyplot as plt

import iris
import iris.plot as iplt
import iris.quickplot as qplt


def main():
    # Enable a future option, to ensure that the grib load uses the mechanism
    # intended for future Iris versions.
    iris.FUTURE.external_grib_support = True

    # What if anything is needed here for the first version of iris_grib ?
    # - we don't actually know yet ...
    # What we might need in future...
    #    import iris_grib
    #    iris_grib.FUTURE.strict_grib_load = True
    # FOR NOW: a nasty kluge, as setting this is now *itself* deprecated.
    iris.FUTURE.__dict__['strict_grib_load'] = True

    file_path = iris.sample_data_path('polar_stereo.grib2')
    cube = iris.load_cube(file_path)
    qplt.contourf(cube)
    ax = plt.gca()
    ax.coastlines()
    ax.gridlines()
    iplt.show()


if __name__ == '__main__':
    main()
