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

#
print("Hello...")

#Filesystem:
filesystem_location = "/home/jcarmona/SomeFilesystem/"

#Where the database info is going to be written:
database_filename = "database_file.txt"
database_file = open(filesystem_location+database_filename,mode="w")

#File containing the filenames to upload:
files_to_upload = "File_list.txt"

for each_line in fileinput.input(files_to_upload):
    print("\n===============================================================")

########THIS IS FOR HADOOP###############
#    command="ls -l "+each_line
#    size=float(os.popen(command).read().split(" ")[4])
#    block_size=math.ceil(size/512)*512
#    upload_command="time hdfs dfs -D dfs.blocksize="+str(int(block_size))+" -put "+each_line[:-1]+" ./NetCDF_files"
#########################################

    #read the cubes contained in each filename:
    print(each_line)
    cubes = ui.read_cubes_from_file(each_line[:-1])

    #Write info to database: name of the binary (NetCDF) 
    #file and cubes it contains:
    database_file.write(each_line)
    database_file.write(cubes)

    #Upload the binary (NetCDF) file to the database:
    upload_command="scp "+each_line[:-1]+" "+filesystem_location
    remove_command="rm "+filesystem_location+each_line[:-1]
    print(upload_command)
    os.system(upload_command)
    print(remove_command)
    os.system(remove_command)

    print("===============================================================\n")

database_file.close()
