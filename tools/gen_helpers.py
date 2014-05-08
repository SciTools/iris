# (C) British Crown Copyright 2013 - 2014, Met Office
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

from datetime import datetime
import os
import os.path

HEADER = \
    '''# (C) British Crown Copyright 2013 - {}, Met Office
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
#
# DO NOT EDIT: AUTO-GENERATED'''


def absolute_path(path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def prep_module_file(module_path):
    """
    prepare a module file, creating directory if needed and writing the
    header into that file

    """
    module_path = absolute_path(module_path)
    module_dir = os.path.dirname(module_path)
    if not os.path.isdir(module_dir):
        os.makedirs(module_dir)
    with open(module_path, 'w') as module_file:
        module_file.write(HEADER.format(datetime.utcnow().year))
