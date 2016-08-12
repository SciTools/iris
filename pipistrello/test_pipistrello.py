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
import pipistrello
import utils
import time
import os

#time_a = time.clock()
#database_dir = ('/home/esp-shared-b/RegCM_Data/CAM2')
#database_dir = ('/home/esp-shared-b/RegCM_Data/regcm3')
#all_cubes = pipistrello.database(database_dir,new_catalogue=True)

#cubesA = all_cubes.load_cubes('')
#for cube in cubesA:
#	print(cube)

#time_b = time.clock()

#print("total time taken: {} seconds.\n".format(time_b - time_a))

files_to_catalogue = os.listdir(self.location)


#This cleans up leftover files (compiled python files)
#and prints a good-bye message.
#utils.cleanup_and_finish()

