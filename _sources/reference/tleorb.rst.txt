
.. module:: pymgp.tleorb

Two-Line Element (TLE) orbits(:mod:`pymgp.tleorb`)
===================================================

Compute satellite position and velocity using two-line elements available from
Celestrak (https://celestrak.org).

.. currentmodule:: pymgp.tleorb


Get and read NORAD Two Line Elements in TLE structure array
-----------------------------------------------------------

.. autosummary::
   :toctree: generated/

   tleget
   tleread
   tleprint


Find satellites and select dates
--------------------------------


.. autosummary::
   :toctree: generated/

   tlefind
   tledatenum


Compute satellite positions and orbit propagation
-------------------------------------------------

.. autosummary::
   :toctree: generated/

   tle2orb
   tle2vec


The functions are optimized for Python's numpy library, it accepts mostly numpy
alike ndarrays with the coordinates (usually three) in the last axis.

Notes
-----
The mean orbital elements in the two-line elements are Earth-centered-inertial 
(ECI) coordinates with respect to the true equator of date and the mean equinox
of date. They do not include the effect of nutation. The tle's are compatible 
with the SGP4 and SDP4 orbit propagators of NORAD.

The default orbit propagator used by this module is a simplified method that
only includes the secular effects of J2. This gives acceptable results for 
plotting purposes and planning computations. The SGP4 and SDP4 methods are
not implemented, anyway, for precise orbits the user should resort to other
sources for satellite orbits  

See Also
--------
pymgp.satorb:
    Satellite orbit module (used by this module)
pymgp.orbplot:
    Plotting of satellite orbits (relies on this module)

Examples
--------
Download tle data for Earth Resource Satellites
    
>>> tlefile = tleget('resource')
Saving TLE set to default name
TLEGET: Downloaded resource.txt from http://celestrak.com/NORAD/elements/
Saved TLE to resource.txt  

Read tle data for Earth Resource Satellites from file (file downloaded on Aug 2, 2024) and
compute the position and velocity of RADARSAT-2

>>> tleERS = tleread('resource-20240802.tle', verbose=0)
>>> xsat, vsat, *_ = tle2vec(tleERS,'2024-08-02 05:58:10','RADARSAT-2') 
>>> print(xsat, vsat)
[[5458404.60763595 4591181.8357549  -732126.12944127]] [[  144.29836955 -1339.50959494 -7333.2677589 ]]

Compute RADARSAT-2 position and velocities for Aug 2, 2024, with one minute intervals

>>> xsat, vsat, t, *_ = tle2vec(tleERS,['2024-08-02 0:00', 24*60, 1],'RADARSAT-2')
>>> print(xsat.shape, vsat.shape, t.shape)
(1441, 3) (1441, 3) (1441,)
   
Position and velocity, on Aug 2, 2024, for every SENTINEL satellite, with one minute interval
 
>>> xsat, vsat, t, satids = tle2vec(tleERS,['2024-08-02 0:00', 24*60, 1],'SENTINEL')
>>> print(xsat.shape, vsat.shape, t.shape, satids.shape)
>>> print(satids)
(7, 1441, 3) (7, 1441, 3) (1441,) (7,)
['SENTINEL-1A' 'SENTINEL-2A' 'SENTINEL-3A' 'SENTINEL-2B' 'SENTINEL-5P'
 'SENTINEL-3B' 'SENTINEL-6']

Find satellites and set date range and interval (used for instance by tle2vec)

>>> isat, satids = tlefind(tleERS, 'SENTINEL', verbose=0)
print(isat, satids)
[ 65  87  99 116 122 133 148] ['SENTINEL-1A' 'SENTINEL-2A' 'SENTINEL-3A' 'SENTINEL-2B' 'SENTINEL-5P'
 'SENTINEL-3B' 'SENTINEL-6']

>>> t = tledatenum(['2024-08-02 0:00', 24*60, 1])
>>> print(t.shape)
(1441,)

References
----------
- Celestrak web-site (https://celestrak.org)
- NORAD GP Element Sets Current Data (https://celestrak.org/NORAD/elements/)

Copyright Hans van der Marel, Delft University of Technology, 2012-2024

