"""TurtLES: turbulent 3D cylinder wakes for neural PDE surrogates.

The package is organised around the lifecycle of the benchmark:

* ``turtles.generation`` reproduces the CFD pipeline (Gmsh + PyFR + HDF5 conversion).
* ``turtles.data`` loads trajectories from a local mirror or straight from Hugging Face.
* ``turtles.models`` implements the baselines behind one forward contract.
* ``turtles.training`` and ``turtles.evaluation`` train surrogates and score full-mesh
  autoregressive rollouts with the metrics of the paper.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
