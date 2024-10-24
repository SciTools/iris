# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for loading with varying references.

Practically, this mostly means loading from fields-based file formats such as PP and
GRIB,  and when hybrid vertical coordinates which have time-varying reference fields.
E.G. hybrid height with time-varying orography, or hybrid-pressure with time-varying
surface pressure.

"""
