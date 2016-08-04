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

class database():

    def __init__(self,location):
        #What we need to know about our database:
        self.catalogue_filename = "CATALOGUE.txt"
        self.location = location
        self.catalogue_filepath = self.location+"/"+self.catalogue_filename
        self.datafiles = []
        self.cubes = []
        self.coordinates = []
        self.metadatas = []

        try:
            self.load()
        except FileNotFoundError:
            print("{} not found in {}.".format(self.catalogue_filename,location))
            print("Trying to create it now...")
            self.create_catalogue()

    def load(self):

        #Where the database info is going to be written:
        try:
            catalogue_file = open(self.catalogue_filepath,mode="r")
        except FileNotFoundError:
            raise
        print("loading database...")
        #Create the list of available cubes
        index = -1
        flag_file = False
        flag_cube = False
        flag_coordinate = False
        flag_metadata = False
        for line in fileinput.input(self.catalogue_filepath):
            #Put the line in the corresponding list
            if flag_file:
                self.datafiles.append(line[:-1])
                print(self.datafiles[-1])
                flag_file = False
            if flag_cube:
                print("index = ",index)
                self.cubes[index].append(line[:-1])
                flag_cube = False
            if flag_coordinate:
                self.coordinates[index].append(line[:-1])
                flag_coordinate = False
            if flag_metadata:
                self.metadatas.append(line[:-1])
                flag_metadata = False

            if line == 'FILE:\n':
                flag_file = True
                self.coordinates.append([])
                self.cubes.append([])
                index += 1
            elif line == 'CUBE:\n':
                flag_cube = True
            elif line == 'COORDINATE:\n':
                flag_coordinate = True
            elif line == 'METADATA:\n':
                flag_metadata = True

    def create_catalogue(self):

        #Get the list of files inside the specified path:
        files_to_catalogue = os.listdir(self.location)
        print(files_to_catalogue)

        #The catalogue will be written here:
        catalogue_file = open(self.catalogue_filepath,'w')
    
        #files that were not read are stored in this list and printed at the end:
        not_read_files = []

        #Go through each file, find what is inside and ad it 
        #to the catalogue:
        for each_file in files_to_catalogue:
            #read the cubes contained in each filename:
            print("reading "+each_file)
            try:    
                cubes = ui.read_cubes_from_file(self.location+"/"+each_file)
            except:
                not_read_files.append(self.location+"/"+each_file)
                continue

    
            #Write location of the binary (NetCDF) files:
            catalogue_file.write("FILE:\n")
            catalogue_file.write(each_file+"\n")
            #Write cubes it contains:  
            for each_cube in cubes:
                catalogue_file.write("CUBE:\n")
                catalogue_file.write(str(each_cube.name())+"\n")
                #Write the coordinates of each cube; its name, max, min, and units:
                for each_coord in each_cube.coords():
                    coordmin, coordmax, coordunits = cube_utils.get_bounds(each_cube,each_coord.name())
                    catalogue_file.write("COORDINATE:\n")
                    catalogue_file.write(each_coord.name()+" "+  str(coordmin)+" "+  str(coordmax)+" "+ "<<"+ str(coordunits)+">>"+"\n")
                #Write the metadata of the cube:
                catalogue_file.write("METADATA:\n")
                catalogue_file.write( str(each_cube.metadata) + "\n" )
    
        catalogue_file.close()

        if( len(not_read_files) > 0 ):
            print("The following files were not added to the catalogue:")
            for item in not_read_files:
                print(item)

        self.load()
