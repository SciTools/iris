#!/usr/bin/env python

import fileinput
import math
import subprocess
from subprocess import call
import os

for each_line in fileinput.input("File_list.txt"):
	print("\n===============================================================")
	command="ls -l "+each_line
	size=float(os.popen(command).read().split(" ")[4])
	block_size=math.ceil(size/512)*512
	upload_command="time hdfs dfs -D dfs.blocksize="+str(int(block_size))+" -put "+each_line[:-1]+" ./NetCDF_files"
	print(upload_command)
	os.system(upload_command)
	print("===============================================================\n")
