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

import utils

#Prints usage message:
def print_usage():
    print("\n\tUsage: {} {}\n".format(sys.argv[0],' <filename1> <filename2> <filename3> ... [--restriction <restriction 1> <restriction 2> ...]'))
    return

def command_line_start_upload():    
    #parse for command line arguments
    parser = argparse.ArgumentParser(description='Uploads datafiles to Filesystem.')
    parser.add_argument('file_with_filepaths', metavar='<Text file containing the list of files to upload, one per line.>',type=str)

    args = parser.parse_args()
    file_with_filepaths=args.file_with_filepaths

    return str(file_with_filepaths)

#Load
def command_line_load_cubes():
    #parse for command line arguments
    parser = argparse.ArgumentParser(description='Loads cubes from files')
    parser.add_argument('filenames', metavar='path/to/NEtCDF_file',type=str,  nargs='+')
    parser.add_argument('--restriction',required=False,type=str, nargs='+')

    args = parser.parse_args()
    filenames=args.filenames
    restriction = args.restriction

    #Load cubes from file:
    try:
        cubes = iris.load(filenames,constraints=restriction)
    except:
        print_usage()
        raise

    return cubes
