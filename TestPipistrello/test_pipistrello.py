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

#load cubes from command line:
cubes = ui.command_line_load()

#print each cube:
for i in range(len(cubes)):
    print("------ Cube {}: ------".format(i))
    print(cubes[i])
    print("\n")



#This cleans up leftover files (compiled python files)
#and prints a good-bye message.
ui.cleanup_and_finish()
