import matplotlib.pyplot as plt

import iris
import iris.quickplot as qplt

fname = iris.sample_data_path('PP', 'aPPglob1', 'global.pp')
temperature = iris.load_strict(fname)

# Take a 1d slice using array style indexing.
temperature_1d = temperature[5, :]

qplt.plot(temperature_1d)
plt.show()
