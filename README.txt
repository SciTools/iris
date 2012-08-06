Custom Builds/Installs of Iris
==============================

Optional library: libmo_unpack

This library is property of the MetOffice and licensed separately.
It is used for decoding/unpacking PP files or Fields files that use an lbpack of 1 or 4.

If this library is not available, then Iris can be installed without it by using the following command:

  python setup.py --without-unpack install

If this library is available but installed in a custom location
then additional compiler arguments can be passed in to ensure that the Python extension
module linking against it builds correctly:

  python setup.py build_ext -I <custom include path> -L <custom static libdir> -R <custom runtime libdir> install
