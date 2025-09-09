
.. module:: pymgp.satorb

Satellite orbits (:mod:`pymgp.satorb`)
=======================================

This module provides basic functions for orbital (Keplerian) elements,
conversion to/from intertial state vectors with position and velocity,
computation of the Earth rotation angle (GMST) and conversion between
intertial (ECI) and Earth Fixed (ECEF) reference frame.

The module also provides function to compute satellite lookangles, range 
and range rate for an object on Earth or in orbit.


.. currentmodule:: pymgp.satorb


Keplerian elements
------------------

.. autosummary::
   :toctree: generated/

   vec2orb
   orb2vec
   orbtype
   kepler
   keplerm
   keplernu


UT1 to GMST, and ECI/ECEF, conversions
--------------------------------------

.. autosummary::
   :toctree: generated/

   ut2gmst
   ecef2eci
   eci2ecef

Satellite look angles, range and range-rate
-------------------------------------------

.. autosummary::
   :toctree: generated/

   satlookangle
   prtlookangle

Sequential date numbers
-----------------------

.. autosummary::
   :toctree: generated/

   datetime2num
   num2datetime
   num2datetime64


The functions are optimized for Python's numpy library, it accepts mostly numpy
alike ndarrays with the coordinates (usually three) in the last axis.


