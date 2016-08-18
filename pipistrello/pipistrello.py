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
import sys
import utils
import iris
import time

import fileinput

#cleans compiled files that may be present:
utils.cleanup_start()

class database():

    #a database object is initialized by 
    #specifying a location in the filesystem:
    def __init__(self,location,new_catalogue=False):
        #File which contains all the datafiles, cubes, 
        #coordinates and metadata of the database:
        self.catalogue_filename = "CATALOGUE.txt"

        #Location of the database inside the filesystem:
        self.location = location+"/"

        #Location of the catalogue file inside the filesystem:
        self.catalogue_filepath = "./"+self.catalogue_filename#self.location+self.catalogue_filename
        print("Catalogue location is here!!!")
        print("Catalogue location is here!!!")
        print("Catalogue location is here!!!")
        print("Catalogue location is here!!!")
        print("Catalogue location is here!!!")

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

        #The [i][j] entry will contain the metadata of cube j in file i.
        self.metadatas = []

        #Database object is initialized by loading the catalogue file above.
        #If the file is not present, the catalogue file is created, only if
        #explicitly asked by the user.
        try:
            self.load()
        except FileNotFoundError:
            print("{} not found in {}.".format(self.catalogue_filename,self.location))
            if new_catalogue:
                print("Creating it now...")
                self.create_catalogue()
            else:
                print(
                        '\tIf you want to use "{}"\n '
                        '\tas a pipistrello database,\n '
                        '\tadd the argument "new_catalogue=True" when creating\n '
                        '\tthe pipistrello database object, e.g.\n\n'
                        'my_database = ' 
                        'pipistrello.database("{}",new_catalogue=True)'
                        .format(self.location,self.location)
                     )



    #This is the core function for creating a database object:
    #It scans, from the catalogue file, all the available cubes.
    #It creates a directory of the cubes contained in each file,
    #its coordinates and its metadata. No cubes is ever loaded.
    def load(self):
        #Where the database info is going to be written:
        try:
            catalogue_file = open(self.catalogue_filepath,mode="r")
        except FileNotFoundError:
            raise

        print("Loading database...")

        #Here is an extract of a sample catalogue file.
        #The next line to the label "FILE:" is the filename of a database file.
        #The next line to the label "CUBE:" 
        #                           gives an iris_cube.name() present on the above file.
        #The next line to the label "COORDINATE:" 
        #                        is an iris_coordinate.name() present on the above cube.
        #The next line to the label "METADATA:" 
        #                            is an iris_cube.metadata present on the above cube.
        ################################################################################
        ################################################################################
                #FILE:
                #coads_climatology.nc
                #CUBE:
                #SEA SURFACE TEMPERATURE
                #COORDINATE:
                #TIME, 0.7575, 8766.5775, , hour since 0000-01-01 00:00:00
                #COORDINATE:
                #latitude, -90.0, 90.0, , degrees
                #COORDINATE:
                #longitude, 20.0, 380.0, , degrees
                #METADATA:
                #CubeMetadata(...)
                #CUBE:
                #MERIDIONAL WIND
                #COORDINATE:
                #TIME, 0.7575, 8766.5775, , hour since 0000-01-01 00:00:00
                #COORDINATE:
                #latitude, -90.0, 90.0, , degrees
                #COORDINATE:
                #longitude, 20.0, 380.0, , degrees
                #METADATA:
                #CubeMetadata(...)
                #CUBE:
                #ZONAL WIND
                #COORDINATE:
                #TIME, 0.7575, 8766.5775, , hour since 0000-01-01 00:00:00
                #COORDINATE:
                #latitude, -90.0, 90.0, , degrees
                #COORDINATE:
                #longitude, 20.0, 380.0, , degrees
                #METADATA:
                #CubeMetadata(...)
                #CUBE:
                #AIR TEMPERATURE
                #COORDINATE:
                #TIME, 0.7575, 8766.5775, , hour since 0000-01-01 00:00:00
                #COORDINATE:
                #latitude, -90.0, 90.0, , degrees
                #COORDINATE:
                #longitude, 20.0, 380.0, , degrees
                #METADATA:
                #CubeMetadata(...)
                #FILE:
                #...
                #CUBE:
                #...
                #COORDINATE:
                #...
                #METADATA:
                #...
        ################################################################################
        ################################################################################

        #The following loop fills the lists of 
        #filenames, cubes, coordinates and metadata
        #available from the catalogue file.
        ifile = -1
        for line in fileinput.input(self.catalogue_filepath):

            #These labels say what next line after them is EXPECTED to be...
            if line in [ 'FILE:\n','CUBE:\n','COORDINATE:\n','METADATA:\n' ]:
                #If the label "FILE:" is found, we will need to
                #add cubes, coordinates and metadata.
                #We alseo increase the counter of filenames read
                #and reset the cube counter.
                if line == 'FILE:\n':
                    utils.debug(line)
                    self.cubes.append([])
                    self.coordinates.append([])
                    self.metadatas.append([])
                    ifile += 1
                    icube = -1
    
                #If the label "CUBE:" is found, we will need
                #to add coordinates and increase the cube counter.
                elif line == 'CUBE:\n':
                    utils.debug(line)
                    self.coordinates[ifile].append([])
                    icube += 1
    
                #For the next two labels we only get preared
                #for what to read next.
                elif line == 'COORDINATE:\n':
                    utils.debug(line)
                elif line == 'METADATA:\n':
                    utils.debug(line)

                #Next line should contain information related to the flag:
                flag = line[:-2]

            else:
                #Put the line after the label in the corresponding list
                if flag == 'FILE':
                    utils.debug("{}, {}, {}".format(ifile,icube,line))
                    self.datafiles.append(line[:-1])
                if flag == 'CUBE':
                    utils.debug("{}, {}, {}".format(ifile,icube,line))
                    self.cubes[ifile].append(line[:-1])
                if flag == 'COORDINATE':
                    utils.debug("{}, {}, {}".format(ifile,icube,line))
                    self.coordinates[ifile][icube].append(line[:-1])
                if flag == 'METADATA':
                    utils.debug("{}, {}, {}".format(ifile,icube,line))
                    self.metadatas[ifile].append(line[:-1])

        print("...database loaded.")
        return

    #When initializing a database object, if a catalogue file is not
    #found inside the database location, this function is automatically
    #called to create one.
    def create_catalogue(self):

        #Get the list of files inside the location of the database.
        #files_to_catalogue = os.listdir(self.location)
        files_to_catalogue = [ os.path.join(p,filename) 
                               for (p,d,f) in os.walk(self.location,onerror=utils.catch_walk_error) 
                               for filename in f ]
        utils.debug(files_to_catalogue)

        #The catalogue will be written here:
        catalogue_file = open(self.catalogue_filepath,'w')
    
        #files that were not read are stored in this list and printed at the end:
        not_read_files = []

        #Go through each file, find what is inside and add it 
        #to the catalogue. If an error occurs with a file, add it
        #to the not_read_files list.
        i = 0
        print("Creating catalogue for {} files...".format(len(files_to_catalogue)))
        t_0 = time.clock()
        t_a = time.clock()
        for each_file in files_to_catalogue:

            #Print timing information.
            i += 1
            if(i%100 == 0):
                files_remaining = len(files_to_catalogue) - i
                minutes_remaining = files_remaining * ( ( time.clock() - t_a ) / 100.0 ) / 60.0
                print("Time taken for adding {} files to catalogue: {} seconds".format(i,time.clock() - t_0))
                print("{} files to go. Estimated time remaining: {} minutes".format( files_remaining, minutes_remaining ) )
                t_a = time.clock()

            #We want the catalogue to contain full path filenames,
            #Even if only printing the filename inside the database
            #directory.
            each_file_path = each_file
            print("reading "+each_file)
            try:    
                cubes = utils.read_cubes_from_file(each_file_path)
            except:
                not_read_files.append(each_file_path)
                continue

    
            #Write location of the binary (NetCDF) files:
            catalogue_file.write("FILE:\n")
            catalogue_file.write(each_file_path+"\n")

            #Write cubes it contains:  
            for each_cube in cubes:
                catalogue_file.write("CUBE:\n")
                catalogue_file.write(str(each_cube.name())+"\n")

                #Write the coordinates of each cube; its name, max, min, and units:
                for each_coord in each_cube.coords():
                    try:
                        coordmin, coordmax, coordunits = utils.get_bounds(each_cube,each_coord.name())
                    except:
                        sys.stderr.write("\n{}: In cube '{}': Unable to set bounds for coordinate '{}'. Coordinate not addded to catalogue\n".format(each_file_path, each_cube.name(),each_coord.name()))
                        continue
                    catalogue_file.write("COORDINATE:\n")
                    catalogue_file.write(each_coord.name()+", "+  str(coordmin)+", "+  str(coordmax)+", "+ str(coordunits)+"\n")

                #Write the metadata of the cube:
                catalogue_file.write("METADATA:\n")
                catalogue_file.write( str(each_cube.metadata) + "\n" )
   
        #Close the catalogue file just created 
        catalogue_file.close()

        #
        print("...catalogue created.")

        #If there were files inside the databae location that were not
        #loaded, let the user know by printing a message.
        if( len(not_read_files) > 0 ):
            print("The following files were not added to the catalogue:")
            for item in not_read_files:
                print(item)

        #Load the databae with the catalogue just created.
        self.load()

    #This function adds a file to the catalogue:
    def add_to_catalogue(filename):
        print("To be implemented. Use instead create_catalogue")
        return


    def load_cubes(self,cube_requested=''):
       
        #To avoid confusion, we map the input of the user
        #to lower case letters.  
        cube_requested = cube_requested.lower()

        #Scan the catalogue and load requested cubes:
        loaded_cubes = []
        t_0 = time.clock()
        t_a = time.clock()
        for i in range(len(self.datafiles)):
            I = i+1
            cubes_to_load = []
            for j in range(len(self.cubes[i])):
                cube_name = self.cubes[i][j]
                if cube_requested in cube_name.lower():
                   cubes_to_load.append(cube_name)
            if len(cubes_to_load) > 0:
                try:
                    loaded_cubes += iris.load(self.datafiles[i],cubes_to_load)
                except:
                    raise

            if(I%100 == 0):
                files_remaining = len(self.datafiles) - I
                minutes_remaining = files_remaining * ( ( time.clock() - t_a ) / 100.0 ) / 60.0
                print("Time taken for exploring {} files in catalogue: {} seconds".format(i,time.clock() - t_0))
                print("{} files to go. Estimated time remaining: {} minutes".format( files_remaining, minutes_remaining ) )
                t_a = time.clock()


        loaded_cubes = iris.cube.CubeList(loaded_cubes)
    
        #If the list of cubes is empty, print a message including
        #possible cube names that the user may have intended.
        if len(loaded_cubes) < 1:
            print('No cubes found with "{}" in its name'.format(cube_requested))
            print("Did you mean...?")
    
            for i in range(len(self.datafiles)):
                for j in range(len(self.cubes[i])):
                    for k in range(0,len(cube_requested)-3):
                        if cube_requested[k:k+3] in self.cubes[i][j].lower():
                            print(self.cubes[i][j])
                            break
        else:
            print("{} cubes found".format(len(loaded_cubes)))
    
        return loaded_cubes


    def load_no_catalogue(self, name_requested=''):

        #Get the list of files inside the location of the database.
        files_to_load = [ os.path.join(p,filename) 
                               for (p,d,f) in os.walk(self.location,onerror=utils.catch_walk_error) 
                               for filename in f ]
        utils.debug(files_to_load)

        #files that were not read will be stored in this list and printed at the end:
        not_read_files = []

        #all the cubes that loaded will be stored here:
        loaded_cubes = []

        #Go through each file, find what is inside and add it 
        #to the list if necessary. If an error occurs with a file, add it
        #to the not_read_files list.
        i = 0
        print("Loading directly from {} files...".format(len(files_to_load)))
        t_0 = time.clock()
        t_a = time.clock()
        for each_file in files_to_load:

            #Print timing information.
            i += 1
            if(i%100 == 0):
                files_remaining = len(files_to_load) - i
                minutes_remaining = files_remaining * ( ( time.clock() - t_a ) / 100.0 ) / 60.0
                print("Time taken for reading {} files: {} seconds".format(i,time.clock() - t_0))
                print("{} files to go. Estimated time remaining: {} minutes".format( files_remaining, minutes_remaining ) )
                t_a = time.clock()

            #Read all the cubes in each file:
            try:    
                cubes = utils.read_cubes_from_file(each_file)
            except:
                not_read_files.append(each_file)
                continue            

            #Chose the relevant cubes:
            cubes_to_save = []
            for j in range(len(cubes)):
                cube_name = cubes[j].name()
                if name_requested in cube_name.lower():
                   cubes_to_save.append(cube_name)

            #Save the cubes just chosen:
            if len(cubes_to_save) > 0:
                try:
                    loaded_cubes += iris.load(each_file,cubes_to_save)
                except:
                    raise

        #Make the list of cubes an iris CubeList:
        loaded_cubes = iris.cube.CubeList(loaded_cubes)
 
        #If no cubes were found, say it...
        if len(loaded_cubes) < 1:
            print('No cubes found with "{}" in its name'.format(name_requested))
        else:
            print("{} cubes found".format(len(loaded_cubes)))

        #If there were problematic files, warn the user:
        if( len(not_read_files) > 0 ):
            print("The following files were not read:")
            for item in not_read_files:
                print(item)


        return loaded_cubes
