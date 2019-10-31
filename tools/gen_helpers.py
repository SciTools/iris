# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

from datetime import datetime
import os
import os.path

HEADER = \
    '''# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
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
