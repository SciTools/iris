#!/usr/bin/env python

##################################################
#                                                #
#  For this program to run, you need to acrivate #
#  the pipistrello environment. You do this by   #
#  entering the following command in the shell:  #
#                                                #
#        $ source activate pipistrello           #
#                                                #
##################################################
#Import modules:
#import fileinput
#import math
#import subprocess
#from subprocess import call

#Import modules:
import sys
import argparse
import os, errno

DEBUG_FLAG=False

#To use the installed version of iris:
#import iris

#In python 3:
from importlib.machinery import SourceFileLoader
#iris = SourceFileLoader("iris", "/home/juan/MHPC-Thesis/pipistrello/lib/iris/__init__.py").load_module()
iris = SourceFileLoader("iris", "../lib/iris/__init__.py").load_module()

#In python 2: I cannot manage to make it work, stick to python 3.
#import importlib
#iris = importlib.importmodule('iris')
#sys.path.append('/home/juan/MHPC-Thesis/pipistrello/lib/iris')
#import imp
#f, pathname, description = imp.find_module('iris')
#iris = imp.load_source('iris','/home/juan/MHPC-Thesis/pipistrello/lib/iris')
#iris = imp.load_module('iris', f, pathname, description)

#Put this to supress some warnings:
iris.FUTURE.netcdf_promote = True

#Silently removes a directory tree: use with caution.
def silentremove(filename):
    try:
        os.remove(filename)
        debug("\t {} removed".format(filename))
    except OSError as e:
        if e.errno == errno.EISDIR: 
            for f in os.listdir(filename):
                silentremove(filename+"/"+f)
            os.rmdir(filename)
            debug("\t {} removed".format(filename))
        elif e.errno == errno.ENOENT: # errno.ENOENT = no such file or directory
            print("{}: No such file or directory.".format(filename))
        else:
            raise

#Reads cubes from a file. 
#In case the file is not a valid Iris file, the execution
#does not stop. It only prints a message saying that the
#file does not exist or is not compatible.
def read_cubes_from_file(filenames,constraints=None):
    #Load cubes from file:
    try:
        cubes = iris.load(filenames,constraints)
    except IOError:
        print('Cannot open one or more files in {}.'.format(filenames)+
          ' Please make sure that file exists.')
    except ValueError:
        print('Cannot open one or more files in {}.'.format(filenames)+
          ' Please make sure that the file is compatible with Iris.')
    except:
        raise

    return cubes

#cleans up compiled files:
def cleanup_start():
    #Clean compiled files:
    print("Cleaning up compiled files...")
    silentremove('compiled_krb')
    silentremove('__pycache__')

#cleans up compiled files
def cleanup_and_finish():
    #Clean compiled files:
    print("Cleaning up compiled files...")
    silentremove('compiled_krb')
    silentremove('__pycache__')

    #If everything ran smoothly, one should see this:
    print("------------------")
    print("Finished execution")


#returns the max, min and units of an iris cube coordinate.
#example use:
#get_bounds(some_iris_cube,'latitude')
def get_bounds(some_cube,coordinate):
    #We get the bounds:
    debug("coordinate = {}".format( coordinate))
    if not some_cube.coord(coordinate).has_bounds():
        try:
            some_cube.coord(coordinate).guess_bounds()
        except:
            raise
        
    mincoord = some_cube.coord(coordinate).bounds.min()
    maxcoord = some_cube.coord(coordinate).bounds.max()
    return str(mincoord), str(maxcoord), str(some_cube.coord(coordinate).units)

#prints a message if debug switch is on
def debug(message,on=False):
    if on:
        print(message)
    return
