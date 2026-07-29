This document explains the changes made to Iris for this release
(:doc:`View all changes </whatsnew/index>`.)


.. dropdown:: 3.16 Release Highlights
   :color: primary
   :icon: info
   :animate: fade-in
   :open:

   The highlights for this minor release of Iris include:

   * Introduced the function :func:`~iris.analysis.cartography.guess_2D_bounds()` for guessing the bounds of pairs of 2D coordinates.
   * Improved NetCDF character-array handling so Iris now supports string data in cube data arrays.
   * Added support for loading and saving of Zarr files formatted in Zarr Storage Spec Version 2
   * Improved the speed of field iteration when reading PP files. Up to 3x speed up has been seen, depending on the circumstances.

   * Special shoutout to :user:`pt331` for their work on the release while deployed into the core development team.
   * We also want to thank :user:`gaoflow`, :user:`2319bli`, :user:`GitRylee`, :user:`king56180468-droid`, :user:`SgtVarmint` for their contributions that have gone into this release.

   And finally, get in touch with us on :issue:`GitHub<new/choose>` if you have
   any issues or feature requests for improving Iris. Enjoy!
