Examples generated from CDL example sections in UGRID conventions v1.0 
  ( see webpage: https://ugrid-conventions.github.io/ugrid-conventions/ )

CHANGES:
    * added a data-var to all examples, for ease of iris-roundtripping
    * EX4 :
        - had a couple of missing ";"s at lineends
        - the formula terms (depth+surface) should map to 'Mesh2_layers', and not to the mesh at all.
            - use Mesh2d_layers dim, and have no 'mesh' or 'location'
    * "EX4a" -- possibly (future) closer mix of hybrid-vertical and mesh dimensions
        - *don't* think we can have a hybrid coord ON the mesh dimension
            - mesh being a vertical location (only) seems to make no sense
            - .. and implies that the mesh is 1d and ordered, which is not really unstructured at all
        - *could* have hybrid-height with the _orography_ mapping to the mesh
            - doesn't match the UGRID examples, but see : iris.tests.unit.fileformats.netcdf.test_Saver__ugrid.TestSaveUgrid__cube.test_nonmesh_hybrid_dim

