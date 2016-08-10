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
import os, errno
import utils
import iris

import fileinput

#cleans compiled files that may be present:
utils.cleanup_start()

class database():

    #a database object is initialized by 
    #specifying a location in the filesystem:
    def __init__(self,location):
        #File which contains all the datafiles, cubes, 
        #coordinates and metadata of the database:
        self.catalogue_filename = "CATALOGUE.txt"

        #Location of the database inside the filesystem:
        self.location = location+"/"

        #Location of the catalogue file inside the filesystem:
        self.catalogue_filepath = self.location+self.catalogue_filename

        #List of all filenames in the database:
        self.datafiles = [] 
        
        #List of all filenames in the database:
        self.datafiles = [] 
        
        #List of lists. 
        #The [i][j] entry will contain the cube j in file i.
        self.cubes = [] 
        
        #Three nested lists. 
        #The [i][j][k] entry will contain the coordinate k of cube j in file i.
        self.coordinates = []

        #The [i][j][k] entry will contain the metadata of cube j in file i.
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
        ifile = -1
        flag_file = False
        flag_cube = False
        flag_coordinate = False
        flag_metadata = False
        for line in fileinput.input(self.catalogue_filepath):
            #Put the line in the corresponding list
            if flag_file:
                utils.debug("{}, {}, {}".format(ifile,icube,line))
                self.datafiles.append(line[:-1])
                flag_file = False
            if flag_cube:
                utils.debug("{}, {}, {}".format(ifile,icube,line))
                self.cubes[ifile].append(line[:-1])
                flag_cube = False
            if flag_coordinate:
                utils.debug("{}, {}, {}".format(ifile,icube,line))
                self.coordinates[ifile][icube].append(line[:-1])
                flag_coordinate = False
            if flag_metadata:
                utils.debug("{}, {}, {}".format(ifile,icube,line))
                self.metadatas[ifile].append(line[:-1])
                flag_metadata = False

            if line == 'FILE:\n':
                utils.debug(line)
                flag_file = True
                self.cubes.append([])
                self.coordinates.append([])
                self.metadatas.append([])
                ifile += 1
                icube = -1
            elif line == 'CUBE:\n':
                utils.debug(line)
                self.coordinates[ifile].append([])
                #self.metadatas[ifile].append([])
                flag_cube = True
                icube += 1
            elif line == 'COORDINATE:\n':
                utils.debug(line)
                flag_coordinate = True
            elif line == 'METADATA:\n':
                utils.debug(line)
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
                cubes = utils.read_cubes_from_file(self.location+"/"+each_file)
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
                    coordmin, coordmax, coordunits = utils.get_bounds(each_cube,each_coord.name())
                    catalogue_file.write("COORDINATE:\n")
                    catalogue_file.write(each_coord.name()+", "+  str(coordmin)+", "+  str(coordmax)+", "+ ", "+ str(coordunits)+"\n")
                #Write the metadata of the cube:
                catalogue_file.write("METADATA:\n")
                catalogue_file.write( str(each_cube.metadata) + "\n" )
    
        catalogue_file.close()

        if( len(not_read_files) > 0 ):
            print("The following files were not added to the catalogue:")
            for item in not_read_files:
                print(item)

        self.load()

    #This function adds a file to the catalogue:
    def add_to_catalogue(filename):
        print("To be implemented. Use instead create_catalogue")
        return


    def load_cubes(self,cube_name=''):
       
        #To avoid confusion, we map the input of the user
        #to lower case letters.  
        cube_name = cube_name.lower()

        #Scan the catalogue and obtain the indices
        #corresponding to the filenames and cubes
        #requested by the user.
        indices_to_load = []
        for i in range(len(self.datafiles)):
            for j in range(len(self.cubes[i])):
                if cube_name in self.cubes[i][j].lower():
                    indices_to_load.append([i,j])

        #Create a list with the desired Iris cubes.
        #It is only at this point that the cubes are
        #actually loaded 
        loaded_cubes = []
        for indices in indices_to_load:
            loaded_cubes.append(
                iris.load(self.location+self.datafiles[indices[0]],
                          self.cubes[indices[0]][indices[1]])
                         )
    
        #If the list of cubes is empty, print a message including
        #possible cube names that the user may have intended.
        if len(loaded_cubes) < 1:
            print('No cubes found with "{}" in its name'.format(cube_name))
            print("Did you mean...?")
    
            for i in range(len(self.datafiles)):
                for j in range(len(self.cubes[i])):
                    for k in range(0,len(cube_name)-3):
                        if cube_name[k:k+3] in self.cubes[i][j].lower():
                            print(self.cubes[i][j])
                            break
    
        return loaded_cubes
