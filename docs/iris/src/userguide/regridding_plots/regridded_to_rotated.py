
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import iris
import iris.analysis
import iris.plot as iplt
import matplotlib.pyplot as plt

global_air_temp = iris.load_cube(iris.sample_data_path('air_temp.pp'))
rotated_psl = iris.load_cube(iris.sample_data_path('rotated_pole.nc'))

rotated_air_temp = global_air_temp.regrid(rotated_psl, iris.analysis.Linear())


plt.figure(figsize=(4, 3))

iplt.pcolormesh(rotated_air_temp, norm=plt.Normalize(260, 300))
plt.title('Air temperature\n'
          'on a limited area rotated pole grid')
ax = plt.gca()
ax.coastlines(resolution='50m')
ax.gridlines()

plt.show()
