import matplotlib.pyplot as plt

import iris
import iris.plot as iplt

# Load the data.
global_air_temp = iris.load_cube(iris.sample_data_path("air_temp.pp"))
rotated_psl = iris.load_cube(iris.sample_data_path("rotated_pole.nc"))

plt.figure(figsize=(9, 3.5))

plt.subplot(1, 2, 1)
iplt.pcolormesh(global_air_temp, norm=plt.Normalize(260, 300))
plt.title("Air temperature\n" "on a global longitude latitude grid")
ax = plt.gca()
ax.coastlines()
ax.gridlines()

plt.subplot(1, 2, 2)
iplt.pcolormesh(rotated_psl)
plt.title("Air pressure\n" "on a limited area rotated pole grid")
ax = plt.gca()
ax.coastlines(resolution="50m")
ax.gridlines()

plt.show()
