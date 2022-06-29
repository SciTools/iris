# Mergenate

## Philosophy

Mergenate is not magic.
Mergenate will tell you why it failed.
Mergenate will make one cube, the way the user asked.

## What's mergenate doing under the hood?

* Choose the axis to work on. For no coord provided, that's an anonymous leading
  axis. For one coord that's the axis that coord lies on (if it's consistent
  across all cubes), or a new anonymous leading one if it's scalar everywhere.
  Inconsistent axes causes a failure. Multiple coords are mergenated on one at a
  time in order.

* Validate that that cubes are all the right shape to merge.

* Reorder the axes of each cube to make the merge axis the leading one

* Establish the order that the cubes go in to make the given coordinate
  monotonic. If all cubes have the coordinate scalar, assume ascending. Sort the
  cubes into this order. 

* Concatenate the cube data.

* Prepare a table of every coordinate in every cube.

* Use this table to ascertain the correct treatment for each coordinate
  (anything dervied from CoordMetadata).

  * All copies of a coordinate lying along the merge axis are concatenated
    together in the same order as their source cubes.

  * A coordinate that's not along the merge axis and has the same values in all
    cubes is just kept as it is.

  * A coordinate that's not along the merge axis and doesn't have the same
    values in all cubes is extended along the merge axis **if
    extend_coords=True**.

* Handle AuxFactories separately.

* Build a new cube.

* Restore the order of the output cube's axes to the expected order.

## What next for mergenate?

* User testing - putting it in front of users and getting feedback.

* Additional test coverage - some errors aren't yet tested because spoofing the
  scenarios that cause them to fire is hard.

* Verbose mode - when a merge or concatenate does something odd, it's hard to
  trace the reasoning it followed. Mergenate should be easier to follow the
  reasoning of, but ideally it would take a kwarg of ``verbose=True`` that made
  it describe what it was doing at any decision point.

* Merge wrapper - in some cases (e.g. pp load) Iris shouldn't need the user to
  explain which coordinates they want merging on. In that case there should be a
  merge-like wrapper that tries each variable and keeps the merges that work,
  discarding the rest.