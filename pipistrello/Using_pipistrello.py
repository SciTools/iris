#!/usr/bin/env python

import pipistrello



database_dir = ('/home/juan/MHPC-Thesis/NetCDF_Files')
all_cubes = pipistrello.database(database_dir,new_catalogue=True)
cubesA = all_cubes.load_cubes('')


#Function to make all cubes compatible:
from iris.util import describe_diff
from iris.util import unify_time_units
from iris.experimental.equalise_cubes import equalise_attributes

def make_cubes_compatible(list_of_cubes):
    equalise_attributes(list_of_cubes)
    unify_time_units(list_of_cubes)

    for cube_i in list_of_cubes:
        cube_i.cell_methods = ()
    
    c = 0
    for i in range(len(list_of_cubes)):
        for j in range(i+1,len(list_of_cubes)):
            if( not list_of_cubes[i].is_compatible(list_of_cubes[j])):
                print('cubes {} and {}:\n'.format(i,j))
                describe_diff(list_of_cubes[i],list_of_cubes[j])
                c+=1
    if c == 0:
        print("All cubes are now compatible.")
    else:
        print("{} incompatible cubes".format(c))

#This function prints the span of a given cube:
def print_span(some_cube):
    for i in some_cube.coords():
        coord_name = i.name()
        if coord_name.lower() == 'time':
            print("{} < {}, {} calendar < {}".format
                  (
                  i.units.num2date(i.points[0]),
                  coord_name, i.units.calendar,
                  i.units.num2date(i.points[-1])
                  )
                 )
        else:
            print("{} < {}, {} < {}".format
                  (
                  i.points[0],
                  coord_name, i.units,
                  i.points[-1]
                  )
                 )

#The next two functions are designed to pick only times
#contained in lists of years, months, days and hours.
#The first one only complements the second.
from iris.time import PartialDateTime
from iris import Constraint
from iris import FUTURE

def _get_truth_value(a,years,months,days,hours):

    year_bool = (a.point.year in years) or (years == [])            
    month_bool = (a.point.month in months) or (months == [])
    day_bool = (a.point.day in days) or (days == [])
    hour_bool = (a.point.hour in hours) or (hours == [])
    
    the_constraint = year_bool and month_bool and day_bool and hour_bool
    
    return the_constraint

def pick_times(some_cube,years,months,days,hours):
    tconstraint = Constraint(time = lambda a: _get_truth_value(a, years, months, days,hours))

    with FUTURE.context(cell_datetime_objects=True):
        extracted = some_cube.extract(tconstraint)
        if extracted is None:
            t_coord = some_cube.coord('time')
            print("No cube extracted, returning 'None'.")
            print("Is your selection within the time bounds of the original cube?")
            print(t_coord.units.num2date(t_coord.points[0]))
            print(t_coord.units.num2date(t_coord.points[-1]))
            print(t_coord.units.calendar)
            
        return some_cube.extract(tconstraint)


       
                
make_cubes_compatible(cubesA)

A_concat = cubesA.concatenate()

for cube in A_concat:
    print_span(cube)
    print("")

#We chose one cube to work with:
test_cube = A_concat[0]

#Now that we have a cube with which to work, we 
#proceed to the analysis:
from iris.analysis.cartography import area_weights
from iris.analysis import MEAN

month_name = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',
              7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}

#This loop devide the test cube into 12 subsets.
#Each sub-cube will contain only one month of
#every year on the dataset.
print("Picking months...")
month_cubes = []
for month in range(1,13):
    print("Picking {}...".format(month_name[month]))
    #We pick the desired cubes:
    month_cubes.append(pick_times(test_cube,[],[month],[],[]))
print("...done picking months.")

#This loop calculates the mean temperature over all space,
#it "collapses" the cube.
print("Calculating spatial means...")
collapsed_cubes = []
for i in range(len(month_cubes)):
    print("Calculating spatial mean for {}...".format(month_name[i+1]))
    #We get the area weights of the cells composing the region:
    grid_areas = area_weights(month_cubes[i])

    #We "collapse" our 2D+Time cube into a 0D+Time by averaging using MEAN aggregator:
    collapsed_cubes.append(month_cubes[i].collapsed(['longitude', 'latitude'], MEAN, weights=grid_areas))
print("...done calculating spatial means.")


#Finally, we cast our analysis into a plot
import matplotlib.pyplot as plt
import iris.quickplot as iplt

print("Plotting...".format(month_name[i+1]))
for i in range(len(collapsed_cubes)):
    #Plot...
    iplt.plot(collapsed_cubes[i],linewidth='10',label=month_name[i+1])
    plt.legend(loc=4)
    #iplt.plot(c_interp,'b.')
print("...done with the plot.")

figname = 'asdf.png'
plt.savefig(figname)
print('Figure saved to {}'.format(figname))
