"""
Example of a polar stereographic plot
=====================================

Demonstrates plotting some data that is defined on a polar stereographic projection.

"""

import matplotlib.pyplot as plt

import iris
import iris.quickplot as qplt


def main():
    file_path = iris.sample_data_path('CMC_glb_TMP_ISBL_1015_ps30km_2013052000_P006.grib2')
    cube = iris.load_cube(file_path)
    qplt.contourf(cube)
    ax = plt.gca() 
    ax.coastlines()
    ax.gridlines()
    plt.show()

if __name__ == '__main__':
    main()
