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

import fileinput
import math
import subprocess
from subprocess import call

#File containing the filenames to upload:
files_to_upload = "walk.stdout"

#temporary directory to store scped files:
local_temp_dir = "./NetCDF_temp/"
os.system("mkdir "+local_temp_dir)

#Hadoop directory to store the files:
hadoop_dir = "NetCDF_Files"
hadoop_mkdir = "hdfs dfs -mkdir" 
os.system(hadoop_mkdir+" "+hadoop_dir)

for each_line in fileinput.input(files_to_upload):
    #scp from remote server to local filesystem in local server:
    scp_command = "scp jcarmona@argo:/" + each_line[:-1] + " "+local_temp_dir
    print(scp_command)
    os.system(scp_command)

    #put file in hadoop
    command="ls -l "+local_temp_dir
    size=float(os.popen(command).read().split(" ")[5])
    filename=os.popen(command).read().split(" ")[-1]
    block_size=math.ceil(size/512)*512
    upload_command="time hdfs dfs -D dfs.blocksize="+str(int(block_size))+" -put "+local_temp_dir+filename[:-1]+" "+hadoop_dir
    print(upload_command)
    os.system(upload_command)
    
    #remove file in local filesystem
    remove_command = "rm "+local_temp_dir+filename[:-1]
    print(upload_command)
    os.system(remove_command)

########THIS IS FOR HADOOP###############
#    command="ls -l "+each_line
#    size=float(os.popen(command).read().split(" ")[4])
#    block_size=math.ceil(size/512)*512
#    upload_command="time hdfs dfs -D dfs.blocksize="+str(int(block_size))+" -put "+each_line[:-1]+" ./NetCDF_files"
#########################################

    #Upload the binary (NetCDF) file to the database:
#    upload_command="scp "+each_line[:-1]+" "+filesystem_location
#    remove_command="rm "+filesystem_location+each_line[:-1]
#    print(upload_command)
#    os.system(upload_command)
#    print(remove_command)
#    os.system(remove_command)
