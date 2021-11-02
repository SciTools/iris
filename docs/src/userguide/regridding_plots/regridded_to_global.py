import matplotlib.pyplot as plt

import iris
import iris.analysis
import iris.plot as iplt

global_air_temp = iris.load_cube(iris.sample_data_path("air_temp.pp"))
rotated_psl = iris.load_cube(iris.sample_data_path("rotated_pole.nc"))

scheme = iris.analysis.Linear(extrapolation_mode="mask")
global_psl = rotated_psl.regrid(global_air_temp, scheme)

plt.figure(figsize=(4, 3))
iplt.pcolormesh(global_psl)
plt.title("Air pressure\n" "on a global longitude latitude grid")
ax = plt.gca()
ax.coastlines()
ax.gridlines()
ax.set_extent([-90, 70, 10, 80])

plt.show()
