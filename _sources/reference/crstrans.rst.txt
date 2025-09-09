
.. module:: pymgp.crstrans

Coordinate conversions and transformations (:mod:`pymgp.crstrans`)
===================================================================

This module provides basic functions for coordinate conversions and transformations
for Earth Centered Earth Fixed Cartesian coordinates, local topocentric 
coordinates, and ellipsoidal geodetic latitude, longitude and height. 

.. currentmodule:: pymgp.crstrans


Coordinate conversions in ECEF reference frame
----------------------------------------------

.. autosummary::
   :toctree: generated/

   xyz2plh
   plh2xyz
   inqell
   setell


Curvilinear coordinate conversions
----------------------------------

.. autosummary::
   :toctree: generated/

   ellcurvature


Coordinate transformations between ECEF and local (topocentric) reference frames
--------------------------------------------------------------------------------

.. autosummary::
   :toctree: generated/

   xyz2neu
   neu2xyz
   xyz2zas
   zas2xyz
   ellnormal
   ellrotmatrix
   covtransform


Print functions
---------------

.. autosummary::
   :toctree: generated/

   printcrd
   printxyz
   printplh
   deg2dms


The functions are optimized for Python's numpy library, it accepts mostly numpy
alike ndarrays with the coordinates (usually three) in the last axis.
