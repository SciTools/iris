
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import matplotlib.pyplot as plt

import iris
import iris.quickplot as qplt


fname = iris.sample_data_path('air_temp.pp')
temperature = iris.load_cube(fname)

# Take a 1d slice using array style indexing.
temperature_1d = temperature[5, :]

qplt.plot(temperature_1d)
plt.show()
