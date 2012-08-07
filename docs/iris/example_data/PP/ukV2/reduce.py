
import iris

cubes = iris.load("THOxayrk.pp")

cube0 = (cubes[0])[::2,::10,::4,::4]
cube1 = (cubes[1])[::2,::10,::4,::4]
cube2 = (cubes[2])[::2,::4,::4]

iris.save([cube0, cube1, cube2], "THOxayrk_subset.pp")
