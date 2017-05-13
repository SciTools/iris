# (C) British Crown Copyright 2013, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Vector functions.

"""

import numpy as np


class Vector(object):
    """ A class which defines a vector e.g. wind """
    def __init__(self, u, v, w=None):
        """
        Constructs a vector object from
         iris.cube.Cube objects representing
         orthogonal components.

        Args:
        * u (iris.cube.Cube):
            ith component
        * v (iris.cube.Cube):
            jth component
        * w (iris.cube.Cube):
            optional kth components

        """

        self.u = u
        self.v = v
        if w:
            self.w = w
        else:
            self.w = None

    def __add__(self, other):
        if self.w is None and other.w is None:
            return Vector(self.u+other.u, self.v+other.v)
        elif self.w is not None and other.w is not None:
            return Vector(self.u+other.u, self.v+other.v, self.w+other.w)
        else:
            raise AttributeError("Both Vector must have the same "
                                 "number of component dimensions.")

    def __sub__(self, other):
        if self.w is None and other.w is None:
            return Vector(self.u-other.u, self.v-other.v)
        elif self.w is not None and other.w is not None:
            return Vector(self.u-other.u, self.v-other.v, self.w-other.w)
        else:
            raise AttributeError("Both Vector must have the same "
                                 "number of component dimensions.")

    def __getitem__(self, key):
        if self.w is None:
            return Vector(self.u[key], self.v[key])
        else:
            return Vector(self.u[key], self.v[key], self.w[key])

    def magnitude(self):
        """
        Returns a cube of the magnitude for 2 or 3 dimensional Vector.

        """
        if self.w is None:
            mag = self.u**2 + self.v**2
            mag.attributes["history"] = "Magnitude of %s and %s components."\
                % (self.u.name(),
                   self.v.name())
        else:
            mag = self.u**2 + self.v**2 + self.w**2
            mag.attributes["history"] = "Magnitude of %s, %s and %s"\
                                        " components."\
                                        % (self.u.name(),
                                           self.v.name(),
                                           self.w.name())
        mag.data = np.sqrt(mag.data)
        mag.rename("Magnitude")
        mag.units = self.u.units

        return mag
