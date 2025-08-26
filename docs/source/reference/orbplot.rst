
.. module:: pymgp.orbplot

Satellite orbit and visibility plots (:mod:`pymgp.orbplot`)
============================================================

Plot satellite orbits in Earth Centered Inertial (ECI) and as ground tracks in
Earth Centered Earth Fixed (ECEF) reference frame, plot information on visibility
from an observer, and look angles, range and range-rate.

High level plot functions
-------------------------

.. autosummary::
   :toctree: generated/

   pltposvel
   pltsattrack
   pltgroundtrack
   pltsatview
   skyplot
   plt3dorbit


Low level plot functions
------------------------

.. autosummary::
   :toctree: generated/

   plot_orbit_3D


Plot support functions
----------------------


.. autosummary::
   :toctree: generated/

   rewrap
   rewrap2d
   xlabels


Other functions
---------------


.. autosummary::
   :toctree: generated/

   plh2ecef
   plh2eci
   ecef2plh
   obj2sat
   xyz2radec
   xyz2zassp
   ra2lon


The functions are optimized for Python's numpy library, it accepts mostly numpy
alike ndarrays with the coordinates (usually three) in the last axis.
