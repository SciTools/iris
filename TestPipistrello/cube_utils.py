#!/usr/bin/env python


######################################################
#                                                    #
#  For this program to run, you need to acrivate     #
#  the pipistrello environment. You can do this by   #
#  entering the following command in the shell:      #
#                                                    #
#        $ source activate pipistrello               #
#                                                    #
######################################################

#Import modules:
import sys
import os, errno
import user_interface as ui
import matplotlib.pyplot as plt
import iris
import iris.quickplot as iplt

import fileinput
import math
import subprocess
from subprocess import call

def get_bounds(c0,coordinate):
    #We get the bounds:
    print(coordinate)
    if not c0.coord(coordinate).has_bounds():
        c0.coord(coordinate).guess_bounds()
        
    mincoord = c0.coord(coordinate).bounds.min()
    maxcoord = c0.coord(coordinate).bounds.max()
    return str(mincoord), str(maxcoord), str(c0.coord(coordinate).units)

#example use:
#    get_bounds(some_iris_cube,'latitude')
#    get_bounds(some_iris_cube,'longitude')
#    get_bounds(some_iris_cube,'TIME')
