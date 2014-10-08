
from __future__ import (absolute_import, division, print_function)

import iris
import iris.analysis
import iris.plot as iplt
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

global_air_temp = iris.load_cube(iris.sample_data_path('air_temp.pp'))

regional_ash = iris.load_cube(iris.sample_data_path('NAME_output.txt'))
regional_ash = regional_ash.collapsed('flight_level', iris.analysis.SUM)

# Mask values so low that they are anomalous.
regional_ash.data = np.ma.masked_less(regional_ash.data, 5e-6)

norm = matplotlib.colors.LogNorm(5e-6, 0.0175)

global_air_temp.coord('longitude').guess_bounds()
global_air_temp.coord('latitude').guess_bounds()

fig = plt.figure(figsize=(8, 4.5))

plt.subplot(2, 2, 1)
iplt.pcolormesh(regional_ash, norm=norm)
plt.title('Volcanic ash total\nconcentration not regridded',
          size='medium')

for subplot_num, mdtol in zip([2, 3, 4], [0, 0.5, 1]):
    plt.subplot(2, 2, subplot_num)
    scheme = iris.analysis.AreaWeighted(mdtol=mdtol)
    global_ash = regional_ash.regrid(global_air_temp, scheme)
    iplt.pcolormesh(global_ash, norm=norm)
    plt.title('Volcanic ash total concentration\n'
              'regridded with AreaWeighted(mdtol={})'.format(mdtol),
              size='medium')

plt.subplots_adjust(hspace=0, wspace=0.05,
                    left=0.001, right=0.999, bottom=0, top=0.955)

# Iterate over each of the figure's axes, adding coastlines, gridlines
# and setting the extent.
for ax in fig.axes:
    ax.coastlines('50m')
    ax.gridlines()
    ax.set_extent([-80, 40, 31, 75])

plt.show()
