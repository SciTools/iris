"""
Example of a polar stereographic plot
=====================================

Demonstrates plotting data that are defined on a polar stereographic
projection.

"""

import iris
import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.pyplot as plt


def main():
    file_path = iris.sample_data_path("toa_brightness_stereographic.nc")
    cube = iris.load_cube(file_path)
    qplt.contourf(cube)
    ax = plt.gca()
    ax.coastlines()
    ax.gridlines()
    iplt.show()


if __name__ == "__main__":
    main()
