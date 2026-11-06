"""Data generation pipeline: Sobol design -> Gmsh mesh -> PyFR iLES -> HDF5.

Stages (one case directory per sample, see :mod:`turtles.generation.pipeline`)::

    [sample]  Sobol design of (Re, alpha, a, n) with train/val/test assignment
    [setup]   case directory, Gmsh geometry + mesh, PyFR import + partition, input files
    [transient] PyFR run until the wake statistics are stationary (chi criterion)
    [extend]  optional continuation runs while chi is still large
    [trajectory] 400 snapshots every 0.2 time units in the sampling box
    [convert] .pyfrs -> .vtu -> deduplicated HDF5 chunks (+ coordinates)

``turtles-gen run <stage> <case_dir>`` runs the last four stages of one case.
Solver, meshing and conversion need PyFR, Gmsh and PyVista (``turtles[generation]``);
the design and the convergence statistic only need NumPy/SciPy/pandas.
"""
