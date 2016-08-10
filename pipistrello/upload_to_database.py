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
import cube_utils

import fileinput
import math
import subprocess
from subprocess import call

ui.cleanup_start()

#Filesystem, database filename and list of files to upload:
files_to_upload = ui.command_line_start_upload()
filesystem_location = "/home/juan/SomeFilesystem/"
database_filename = "database_file.txt"

#Where the database info is going to be written:
database_file = open(filesystem_location+database_filename,mode="w")

#File containing the filenames to upload:

for each_line in fileinput.input(files_to_upload):
########THIS IS FOR HADOOP###############
#    command="ls -l "+each_line
#    size=float(os.popen(command).read().split(" ")[4])
#    block_size=math.ceil(size/512)*512
#    upload_command="time hdfs dfs -D dfs.blocksize="+str(int(block_size))+" -put "+each_line[:-1]+" ./NetCDF_files"
#########################################

    #read the cubes contained in each filename:
    print("reading "+each_line)
    cubes = ui.read_cubes_from_file(each_line[:-1])

    #Write info to database: location of the binary (NetCDF) files:
    database_file.write("FILE:\n")
    database_file.write(each_line)
    #Write cubes it contains:  
    for each_cube in cubes:
        database_file.write("CUBE:\n")
        database_file.write(str(each_cube.standard_name)+"\n")
        #Write the coordinates of each cube; its name, max, min, and units:
        for each_coord in each_cube.coords():
            coordmin, coordmax, coordunits = cube_utils.get_bounds(each_cube,each_coord.standard_name)
            database_file.write("COORDINATE:\n")
            database_file.write(each_coord.standard_name+" "+  str(coordmin)+" "+  str(coordmax)+" "+ "<<"+ str(coordunits)+">>"+"\n")
		#Write the metadata of the cube:
        database_file.write("METADATA:\n")
        database_file.write( str(each_cube.metadata) + "\n" )

    #Upload the binary (NetCDF) file to the database:
#    upload_command="scp "+each_line[:-1]+" "+filesystem_location
#    remove_command="rm "+filesystem_location+each_line[:-1]
#    print(upload_command)
#    os.system(upload_command)
#    print(remove_command)
#    os.system(remove_command)
database_file.close()
