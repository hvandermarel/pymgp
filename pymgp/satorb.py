"""
Satellite orbits.

Functions:

Keplerian elements

- vec2orb        Convert inertial state vector into Keplerian elements
- orb2vec        Convert Keplerian elements into inertial state vector
- orbtype        Classify circular and equatorial orbit types    
- kepler         Compute mean anomaly from eccentric anomaly (Kepler's equation)
- keplernu       Compute mean anomaly from true anomaly (Kepler's equation)
- keplerm        Compute eccentric/true from mean anomaly solving Kepler's eqn

UT1 to GMST, and ECI/ECEF, conversions

- ut2gmst        Compute Greenwich Mean Siderial Time from UT1
- ecef2eci       Convert position and velocity from ECEF to ECI reference frame
- eci2ecef       Convert position and velocity from ECI to ECEF reference frame

Look angles
    
- satlookanglesp Compute table with satellite look angles
- prtlookangle   Print a table with satellite look angles

Sequential date numbers

- datetime2num   Convert datestring, datetime or np.datetime64 to serial datenumber
- num2datetime64 Convert serial datenumber to np.datetime64
- num2datetime   Convert serial datenumber to datetime

"""

__author__ = "Hans van der Marel"
__copyright__ = "Copyright 2010-2024, Hans van der Marel, Delft University of Technology."
__credits__ = ["Hans van der Marel", "Simon van Diepen", "Ullas Rajvanshi"]
__license__ = "License Name and Info"
__version__ = "0.9.0"
__maintainer__ = "Hans van der Marel"
__email__ = "h.vandermarel@tudelft.nl"
__status__ = "development"

"""
Created:    16 Nov 2010 by Hans van der Marel for Matlab
Modified:   26 Nov 2016 by Hans van der Marel
             - added eci2ecef and ecef2eci to the Matlab version
            26 Jul 2017 by Hans van der Marel
             - created satlookangle function for matlab
             1 Nov 2017 by Hans van der Marel
             - satlookangle version for sperical angles for matlab
             - created prtlookangle (split off from satlookangle) for matlab
            18 Nov 2020 by Simon van Diepen, Ullas Rajvanshi and Hans van der Marel
             - port to Python (part of `crsutil` module)
            25 Nov 2021 by Hans van der Marel
             - added satlookanglesp, prtlookangle, datetime2num and num2datetime
            10 July 2024 by Hans van der Marel
               - added option for ellipsoidal or spherical angles to satlookangle,
               - satlookanglesp is now obsolete
            28 Jul 2024 by Hans van der Marel
             - functions vec2orb, orb2vec, kepler, keplernu, keplerm, ut2gmst, ecef2eci,
               eci2ecef, datetime2num, num2datetime, satlookangle and prtlookangle
               moved into new (this) module `satorb`
             - added new function `orbtype`
             - major rewrite with support for multidimension arrays
             - docstrings in numpy style
             - moved track changes sections from the individual functions to a
               single track changes section at the head of the module
            30 Jul 2024 by Hans van der Marel
             - rewrite of ut2gmst, removed dependency on datetime2num, using
               numpy datetime64 instead of datetime and parsedate
             - functions datetime2num and num2datetime rewritten using numpy 
               datetime64 instead of Python datetime. 
             - New function num2datetime64
             - import of dateutil.parser removed
            31 Jul 2024 by Hans van der Marel
             - major changes to satlookangle and prtlookangle, including
               modified docstring, and support for more dimensions, removed
               dependency on num2datetime
            22 Aug 2024 by Hans van der Marel
             - Major edits to the docstrings to facilitate sphynx
                                  
Based on code originally developed for Matlab(TM) in 2010-2016 by the author.

Copyright Hans van der Marel, Delft University of Technology, 2010-2024
"""

# Import modules

import numpy as np
from crstrans import xyz2plh
from datetime import datetime

# ----------------------------------------------------------------------------
#                            KEPLERIAN ELEMENTS
# ----------------------------------------------------------------------------
#
#   vec2orb     - Convert inertial state vector into Keplerian elements
#   orb2vec     - Convert Keplerian elements into inertial state vector
#   orbtype     - Classify circular and equatorial orbit types    
#   kepler      - Compute mean anomaly from eccentric anomaly (Kepler's equation)
#   keplernu    - Compute mean anomaly from true anomaly (Kepler's equation)
#   keplerm     - Compute eccentric/true from mean anomaly solving Kepler's eqn


def vec2orb(svec, GM=3986004418e5):
    """
    Convert inertial state vector into Keplerian elements.
    
    This function converts an array with 6-element inertial state vector(s), 
    with cartesian position and velocity `[X, Y, Z, Xdot, Ydot, Zdot]`, into an 
    array with the 6 Keplerian elements.

    Parameters
    ----------
    svec : array_like with shape (...,6) 
        Array with  6-element inertial state vector `svec` with Cartesian  
        position and velocity `[X, Y, Z, Xdot, Ydot, Zdot]` in an ECI frame.
        Units are meter and meter/sec
    GM : float, optional
        Value of GM. Default for GM [meter**3/sec**2] is the IERS 1996 standard
        value for the Earth (GM=3986004418e5)
        
    Returns
    -------
    orb : ndarray with shape (...,6) similar to `svec`
        Array with the 6 Keplerian elements `[ Semi-major axis (meters), Eccentricity (unity),  
        Inclination (radians), Right ascension of the ascending node (radians),  
        Argument of the pericenter (radians),  True anomaly (radians) ]`.
        Units are meters or radians.

    Notes
    -----
    The computations do not do special hanling of circular or equatorial
    orbits. This is possible because atan2(0,0)=0 is defined in numpy, however,
    some pairs of angles will actually be singular.

    See Also
    --------
    orb2vec, kepler, keplernu, keplerm
    
    Examples
    --------
    >>> vec2orb([-5767701.7786, -3097382.0132, 2737195.4374,  3001.4240, -6762.1059, -1280.2560])
    array([7.12138217e+06, 4.31759805e-03, 4.35789999e-01, 1.60777001e+00,
           1.39024880e+00, 5.98791190e-01])

    >>> #              X              Y             Z            Vx          Vy          Vz
    >>> svec = [[-5767701.7786, -3097382.0132, 2737195.4374,  3001.4240, -6762.1059, -1280.2560],
    ...         [ -625224.3608, -7162713.9833,   16233.9739, -1131.8951,   116.3879,  7358.7436],
    ...         [-2301220.9378,  6666917.3737,  968353.5303, -6610.7556, -1833.8971, -2992.6557]]
    >>> vec2orb(svec)
    array([[7.12138217e+06, 4.31759805e-03, 4.35789999e-01, 1.60777001e+00,
            1.39024880e+00, 5.98791190e-01],
           [7.19093100e+06, 1.80794915e-04, 1.72419000e+00, 4.62567000e+00,
            7.39124272e-01, 5.54634573e+00],
           [7.12297382e+06, 1.74770022e-03, 4.36259992e-01, 5.34368001e+00,
            1.56062623e+00, 1.25322376e+00]])
    >>> vec2orb(svec).shape
    (3, 6)

    >>> vec2orb([svec, svec]).shape
    (2, 3, 6)
    """ 
    
    # Force input array to ndarray and check the shape

    svec = np.asarray(svec)

    assert svec.shape[-1] == 6 , "State vector svec must have six elements in the last dimension." 

    # Inner products (rrdot = R.V , r=sqrt(R.R) , vsq = V.V )

    rrdot = svec[..., 0] * svec[..., 3] + svec[..., 1] * svec[..., 4] + svec[..., 2] * svec[..., 5]
    r = np.sqrt(svec[..., 0] * svec[..., 0] + svec[..., 1] * svec[..., 1] + svec[..., 2] * svec[..., 2])
    vsq = svec[..., 3] * svec[..., 3] + svec[..., 4] * svec[..., 4] + svec[..., 5] * svec[..., 5]

    # Angular momentum vector (H = R x V)

    hx = svec[..., 1] * svec[..., 5] - svec[..., 2] * svec[..., 4]
    hy = svec[..., 2] * svec[..., 3] - svec[..., 0] * svec[..., 5]
    hz = svec[..., 0] * svec[..., 4] - svec[..., 1] * svec[..., 3]

    hsini2 = hx * hx + hy * hy
    hsq = hsini2 + hz * hz
    h = np.sqrt(hsq)

    # Semi-major axis

    ainv = 2 / r - vsq / GM
    a = 1. / ainv

    # Eccentricity

    ome2 = np.asarray(hsq * ainv / GM)
    #ecc = np.sqrt(1.0 - ome2)
    #ecc[ome2 > 1] = 0  # special handling of negative values
    ecc = np.zeros_like(ome2)
    ecc[ome2 <= 1] = np.sqrt(1.0 - ome2[ome2 <= 1])

    # Inclination (0...pi)

    incl = np.arccos(hz / h)

    # Determine orbit type (for handling of special cases)
    #    
    #   'ei'   elliptical inclined     (all Kepler elements defined)
    #   'ci'   circular inclined       (w =0, nu=arglat)
    #   'ee'   elliptical equatorial   (w=lonper, omega=0)
    #   'ce'   circular equatorial     (w=0, omega=0, nu=truelon)
    
    # corbtyp = orbtype(ecc, incl, tol=1e-8)

    # Standard handling of elliptical inclined orbits...

    # The computations below do not do special hanling of circular or equatorial
    # orbits. This is possible because atan2(0,0)=0 is defined in Matlab, however,
    # the some pairs of angles will actually be singular

    # Longitude of ascending node (0...2*pi)

    omega = np.asarray(np.arctan2(hx, -hy))
    omega[omega < 0] += 2 * np.pi

    # True anomaly (0...2*pi)

    resinf = a * ome2 * rrdot / h
    recosf = a * ome2 - r
    nu = np.asarray(np.arctan2(resinf, recosf))
    nu[nu < 0] += 2 * np.pi

    # Argument of perigee (0...2*pi)

    suprod = -hz * (svec[..., 0] * hx + svec[..., 1] * hy) + svec[..., 2] * hsini2
    cuprod = h * (-svec[..., 0] * hy + svec[..., 1] * hx)
    w = np.asarray(np.arctan2(suprod * recosf - cuprod * resinf, cuprod * recosf + suprod * resinf))
    w[w < 0] += 2 * np.pi

    # stack the output array

    #orb = np.array([a, ecc, incl, omega, w, nu])
    #orb = orb.transpose()
    
    orb = np.stack((a, ecc, incl, omega, w, nu), axis=-1)

    return orb


def orb2vec(orb, GM=3986004418e5):
    """
    Convert Keplerian elements into intertial state vector.
    
    This function converts an array with 6-element Kepler elements into an inertial state 
    vector(s), with cartesian position and velocity `[X, Y, Z, Xdot, Ydot, Zdot]`.

    Parameters
    ----------
    orb : array_like with shape (...,6)
        Array with the 6 Keplerian elements `[ Semi-major axis (meters), Eccentricity (unity),  
        Inclination (radians), Right ascension of the ascending node (radians),  
        Argument of the pericenter (radians),  True anomaly (radians) ]`.
        Units are meters or radians.
    GM : float, optional
        Value of GM. Default for GM [meter**3/sec**2] is the IERS 1996 standard
        value for the Earth (GM=3986004418e5)
        
    Returns
    -------
    svec : ndarray with shape (...,6) similar to `orb`
        Array with  6-element inertial state vector `svec` with Cartesian  
        position and velocity `[X, Y, Z, Xdot, Ydot, Zdot]` in an ECI frame.
        Units are meter and meter/sec

    See Also
    --------
    vec2orb, kepler, keplernu, keplerm
    
    Examples
    --------
    >>> orb2vec([7121382.201, 0.0043176, 0.43579, 1.60777, 1.39025, 0.59879])
    array([-5.76770178e+06, -3.09738201e+06,  2.73719544e+06,  3.00142402e+03,
           -6.76210591e+03, -1.28025603e+03])

    >>> #             a          ecc      inc      raan      argp     nu 
    >>> orb = [ [7121382.201, 0.0043176, 0.43579, 1.60777, 1.39025, 0.59879],
    ...         [7190931.091, 0.0001808, 1.72419, 4.62567, 0.73905, 5.54642],
    ...         [7122973.745, 0.0017477, 0.43626, 5.34368, 1.56062, 1.25323] ]
    >>> orb2vec(orb)
    array([[-5.76770178e+06, -3.09738201e+06,  2.73719544e+06,
             3.00142402e+03, -6.76210591e+03, -1.28025603e+03],
           [-6.25224361e+05, -7.16271398e+06,  1.62339739e+04,
            -1.13189511e+03,  1.16387852e+02,  7.35874365e+03],
           [-2.30122094e+06,  6.66691737e+06,  9.68353530e+05,
            -6.61075555e+03, -1.83389705e+03, -2.99265574e+03]])
    >>> orb2vec(orb).shape
    (3, 6)
    
    >>> orb2vec([orb, orb]).shape
    (2, 3, 6)
    
    """ 
    
    # Force input array to ndarray and check the shape

    orb = np.asarray(orb)

    assert orb.shape[-1] == 6 , "Array with Keplerian elements must have six elements in the last dimension." 

    # Compute position (rx,ry) and velocity (vx,vy) in orbital plane (perifocal system)

    ecc = orb[..., 1]                   # Eccentricity
    cosnu = np.cos(orb[..., 5])         # Cosine and sine of true anomaly (nu)
    sinnu = np.sin(orb[..., 5])

    p = orb[..., 0] * (1.0 - ecc ** 2)  # Parameter of the ellipse p=a*(1-e^2)

    r = p / (1.0 + ecc * cosnu)         # Length of position vector

    rx = r * cosnu                      # Position (rx,ry) in orbital plane
    ry = r * sinnu

    p = np.asarray(p)
    p[abs(p) < 0.0001] = 0.0001         # Protect against division by zero
    tmp = np.sqrt(GM / p)

    vx = -tmp * sinnu                   # Velocity (vx,vy) in orbital plane
    vy = tmp * (ecc + cosnu)

    # Convert into inertial frame (3-1-3 Euler rotations)

    cosincl = np.cos(orb[..., 2])       # Cosine and sine of inclination (incl)
    sinincl = np.sin(orb[..., 2])
    cosomega = np.cos(orb[..., 3])      # Cosine and sine of longitude of ascending node (omega)
    sinomega = np.sin(orb[..., 3])
    cosw = np.cos(orb[..., 4])          # Cosine and sine of argument of perigee (w)
    sinw = np.sin(orb[..., 4])

    rx0 = cosw * rx - sinw * ry         # Cosine and sine of argument of latitude u=w+nu
    ry0 = cosw * ry + sinw * rx

    vx0 = cosw * vx - sinw * vy
    vy0 = cosw * vy + sinw * vx

    # stack the output array

    svec = np.stack((rx0 * cosomega - ry0 * cosincl * sinomega,
                     rx0 * sinomega + ry0 * cosincl * cosomega,
                     ry0 * sinincl,
                     vx0 * cosomega - vy0 * cosincl * sinomega,
                     vx0 * sinomega + vy0 * cosincl * cosomega,
                     vy0 * sinincl), axis=-1)

    return svec

def orbtype(ecc, incl, tol=1e-8):
    """Determine circular and equatorial orbit types (for handling of special cases).
    
    Parameters
    ----------
    ecc, incl : array_like 
        Eccentricity (unity) and inclination (radians)
    tol : float, default 1e-8
        Tolerance for zero eccentricity and zero inclination

    Returns
    -------
    corbtype : ndarray of {'ei', 'ci', 'ee', 'ce'}
        Array with two character codes for the orbit type:    
        
        | 'ei' ->  elliptical inclined     (all Kepler elements defined)
        | 'ci' ->  circular inclined       (w =0, nu=arglat)
        | 'ee' ->  elliptical equatorial   (w=lonper, omega=0)
        | 'ce' ->  circular equatorial     (w=0, omega=0, nu=truelon)
        
    Notes
    -----
    Nearly circular and/or equatiorial orbits are often in need of special handling 
    because of singularities between pairs of orbital elements. 

    See Also
    --------
    vec2orb, orb2vec
    
    Examples
    --------
    >>> #             a          ecc      inc      raan      argp     nu 
    >>> orb = [ [7121382.201, 0.0043176, 0.43579, 1.60777, 1.39025, 0.59879],
    ...         [7190931.091, 0.1e-9,    1.72419, 4.62567, 0.73905, 5.54642],
    ...         [7122973.745, 0.0017477, 0.1e-9,  5.34368, 1.56062, 1.25323] ]
    >>> orb = np.asarray(orb)
    >>> orbtype(orb[...,1],orb[...,2])
    array([b'ei', b'ci', b'ee'], dtype='|S2')
    
    """    
    
    ecc = np.asarray(ecc)
    incl = np.asarray(incl)

    # Determine orbit type (for handling of special cases)
  
    idxecc = ecc < tol
    idxincl = np.logical_or((incl < tol), (np.abs(incl - np.pi) < tol))

    # Return array with character codes

    corbtype = np.zeros(ecc.shape, dtype='S2')
    corbtype[ idxecc & idxincl ] = 'ce'      # circular equatorial => w=0, omega=0, nu=truelon
    corbtype[ idxecc & ~idxincl ] = 'ci'     # circular inclined => w =0, nu=arglat
    corbtype[ ~idxecc & idxincl ] = 'ee';    # elliptical equatorial => w=lonper, omega=0
    corbtype[ ~idxecc & ~idxincl ] = 'ei'    # elliptical inclined

    return corbtype

def kepler(E, ecc):
    """Compute mean anomaly from eccentric anomaly (Kepler's equation).
        
    Parameters
    ----------
    E : array_like
        Eccentric anomaly (radians)
    ecc : array_like 
        Eccentricity (unity)

    Returns
    -------
    M : array_like
        Mean anomaly (radians)

    Notes
    -----
    This routine should only be used for elliptical orbits. Parabolic and
    hyperbolic orbits are not supported and give false results (this is
    nowhere checked for).
    
    See Also
    --------
    keplernu, keplerm
    """    

    M = E - ecc * np.sin(E)

    return M


def keplernu(nu, ecc):
    """Compute mean and eccentric anomaly from true anomaly (Kepler's equation).
        
    Parameters
    ----------
    nu : array_like
        True anomaly (radians)
    ecc : array_like 
        Eccentricity (unity)

    Returns
    -------
    M : array_like
        Mean anomaly (radians)
    E : array_like
        Eccentric anomaly (radians) 

    Notes
    -----
    This routine should only be used for elliptical orbits. Parabolic and
    hyperbolic orbits are not supported and give false results (this is
    nowhere checked for).
    
    See Also
    --------
    kepler, keplerm
    """

    denom = 1.0 + ecc * np.cos(nu)
    sine = (np.sqrt(1.0 - ecc * ecc) * np.sin(nu)) / denom
    cose = (ecc + np.cos(nu)) / denom
    E = np.arctan2(sine, cose)

    # Compute mean anomaly

    M = E - ecc * np.sin(E)

    return M, E


def keplerm(M, ecc, TOL=1e-10):
    """Compute eccentric and true anomaly from mean anomaly solving Kepler's eqn.
    
    Parameters
    ----------
    M : array_like
        Mean anomaly (radians)
    ecc : array_like 
        Eccentricity (unity)
    tol : float, default 1e-10
        Stop criterion for iterations

    Returns
    -------
    E : array_like
        Eccentric anomaly (radians) 
    nu : array_like
        True anomaly (radians)

    Notes
    -----
    Kepler's equation ``M=E-ecc*sin(E)`` is solved iteratively using Newton's
    method. 
    
    This routine should only be used for elliptical orbits. Parabolic and
    hyperbolic orbits are not supported and give false results (this is
    nowhere checked for).
    
    See Also
    --------
    kepler, keplernu
    """
    
    E = M                            # Use M for the first value of E
    # [m, n] = E.shape
    f = np.ones(E.shape)              # Newton's method for root finding
    while max(abs(f)) > TOL:
        f = M - E + ecc * np.sin(E)  # Kepler's Equation
        fdot = -1 + ecc * np.cos(E)  # Derivative of Kepler's equation
        E = E - f / fdot

    sinnu = -1 * np.sqrt(1 - ecc ** 2) * np.sin(E) / fdot
    cosnu = (ecc - np.cos(E)) / fdot

    nu = np.arctan2(sinnu, cosnu)    # True anomaly

    return E, nu


# ----------------------------------------------------------------------------
#                 UT1 to GMST, and ECI/ECEF, conversions
# ----------------------------------------------------------------------------
#
#   ut2gmst    - Compute Greenwich Mean Siderial Time from UT1
#   ecef2eci   - Convert position and velocity from ECEF to ECI reference frame
#   eci2ecef   - Convert position and velocity from ECI to ECEF reference frame

def ut2gmst(ut1, model="IAU-82"):
    """Compute Greenwich Mean Siderial Time from UT1.
    
    Parameters
    ----------
    ut1 : array_like, float64 or datetime64 or str
        Universal time datetime64 object, numpy parsable datetime string, or
        matplotlib sequential datenumber (days since '01-Jan-1970')
    model : {'IAU-82', 'APPROXIMATE'}, optional
        Model for `ut` to `gmst` conversion. Default is 'IAU-82' model.

    Returns
    -------
    gmst : array_like
        Greenwich Mean Siderial Time GMST [0-2pi rad] for UT1 (radians)
    omegae : float
        Rotation rate of the Earth (rev/day)
    
    See Also
    --------
    ecef2eci, eci2ecef
    
    Examples
    --------
    >>> ut2gmst('2012-01-04 00:00')
    (1.7979884328663978, 1.0027379093)

    >>> ut2gmst(['2012-01-04 15:00:03.13', '2012-01-04 16:00:03'])
    (array([5.73595924, 5.99846593]), 1.0027379093)

    Use matplotlib datenumbers as input (days since 1-jan-1970)

    >>> datenum = np.datetime64('2012-01-04 15:00:03.13','ns').astype(np.int64)*1e-9/86400
    >>> print(datenum)
    15343.625036226851
    >>> ut2gmst(datenum)
    (5.7359592379770525, 1.0027379093)

    >>> ut2gmst(np.arange(datenum,datenum+1,.1))
    (array([5.73595924, 0.08281274, 0.71285155, 1.34289036, 1.97292917,
           2.60296798, 3.23300679, 3.8630456 , 4.49308441, 5.12312322]), 1.0027379093)
    >>> gmst0, omegae = ut2gmst(datenum)
    >>> gmst = (gmst0 + 2*np.pi*omegae*np.arange(0,1,.1) ) % ( 2*np.pi ) 
    >>> gmst
    array([5.73595924, 0.08281274, 0.71285155, 1.34289036, 1.97292917,
           2.60296798, 3.23300679, 3.8630456 , 4.49308441, 5.12312322])

    """
    
    assert  model.upper() in ['IAU-82', 'APPROXIMATE'], f"Unsupported model {model}."

    # force input to be numpy array

    ut1 = np.asarray(ut1)

    # convert `ut1` to `dut` (np.float64) with days since 2000-01-01 12:00

    if ut1.dtype.type == np.float64:
        # ut1 is matplotlib sequential datenumber (days since 01-Jan-1970)
        # ( np.datetime64('2000-01-01T12:00:00') - np.datetime64('1970-01-01') ) / np.timedelta64(1, 'D') -> 10957.5
        dut1 = ut1 - 10957.5
    else:
        # ut1 is datetime64 object or a parsable ISO string by datetime64
        t0 = np.datetime64('2000-01-01T12:00:00','ns')
        dut1 = ( np.array( ut1, dtype='datetime64[ns]') - t0 ) / np.timedelta64(1, 'D')

    # gmst in seconds

    if model.upper() == "APPROXIMATE":
        gmst = (18.697374558 + 24.06570982441908 * dut1 ) % 24
        gmst *= 3600
    elif model.upper() == "IAU-82":
        j2000 = dut1 / 36525.0
        gmst = ((- 6.2e-6 * j2000 + 0.093104) * j2000 + (876600.0 * 3600.0 + 8640184.812866)) * j2000 + 67310.54841
        gmst %= 86400

    # output gmst in radians and omegae in revs/day    

    gmst *= np.pi / 43200
    omegae = 1.0027379093

    return gmst, omegae

def ecef2eci(t, xsate, vsate=[]):
    """Convert position and velocity from ECEF to ECI reference frame.
    
    Convert position and velocity given in an Earth Centered Earth Fixed (ECEF) 
    reference frame into a non-rotating pseudo-inertial Earth Centered Inertial (ECI) 
    reference frame.
    
    Parameters
    ----------
    t : array_like with shape (n,) or scalar, of type datetime64, str or float
        Universal time as `datetime64` object, ISO date string or sequential date 
        number (days since 1970-01-01).
    xsate : array_like with shape (...,n,3) or (...,n,6), or, shape (3,) or (6,) 
        Array with Cartesian coordinates (m) or state vector with positions (m) 
        and velocities (m/s) in an ECEF reference frame .
    vsate : array_like with shape (...,n,3), optional
        Array with velocities in ECEF reference frame (m/s). If empty, velocities
        are taken from 'xsate[...,3:6]', or if `xsate` has shape (...,n,3) or (3,)
        velocities are assumed to be zero (e.g. non-moving points on the Earth surface)  

    Returns
    -------
    xsat : ndarray with shape (...,n,3) or shape (...,n,6) 
        Array with Cartesian coordinates (m) or state vector with positions (m) 
        and velocities (m/s) in ECI reference frame. 
    vsat : ndarray with shape (...,n,3), optional
        Array with velocities in ECI reference frame (m/s), only if `xsat` is not
        a state vector.

    Notes
    -----    
    The function returns a single ndarray `xsat` with the ECI state vector when the 
    parameter `xsate` is a state vector (having 6 elements). Otherwise, it returns 
    two ndarrays.
    
    If the parameter `xsate` has shape (3,) or (...,1,3), and parameter `t` has shape
    (n,) with 'n > 1', then `xsate` is extended to match the length of `t`.
    
    The function always returns ndarrays with at least two dimensions.
    
    See Also
    --------
    ut2gmst, eci2ecef

    Examples
    --------
    Example with single statevector (note the two dimensional result(s)) 
    
    >>> ecef2eci('2012-01-04 15:00:00',[ -3312531.1007, -5646883.8176, 2737195.4374,  5670.7751, -3969.9994, -1280.2560 ])
    array([[-5.76770178e+06, -3.09738201e+06,  2.73719544e+06,
             3.00142403e+03, -6.76210591e+03, -1.28025600e+03]])
    >>> ecef2eci('2012-01-04 15:00:00',[ -3312531.1007, -5646883.8176, 2737195.4374], [ 5670.7751, -3969.9994, -1280.2560 ])
    (array([[-5767701.77861271, -3097382.01317908,  2737195.4374    ]]), array([[ 3001.4240322 , -6762.10590869, -1280.256     ]]))

    Two dimensional examples 

    >>> t = np.array(['2012-01-04 15:00:00', '2012-01-04 16:00:00', '2012-01-04 17:00:00', '2012-01-04 18:00:00'], dtype='datetime64[ns]')
    >>> svece = [[-3312531.1007, -5646883.8176, 2737195.4374,  5670.7751, -3969.9994, -1280.2560 ],
    ...          [ 1413410.4872, -7049655.8712,   16233.9739, -1633.0415,  -309.5460,  7358.7436 ],
    ...          [-2450115.4006,  6613647.9795,  968353.5303, -6085.7029, -1802.9846, -2992.6557 ],
    ...          [ -649857.9621,  7022897.5289,  968353.5303, -6345.1020,  -161.9056, -2992.6557 ]]
    >>> ecef2eci(t, svece)
    array([[-5.76770178e+06, -3.09738201e+06,  2.73719544e+06,
             3.00142403e+03, -6.76210591e+03, -1.28025600e+03],
           [-6.25224361e+05, -7.16271398e+06,  1.62339739e+04,
            -1.13189509e+03,  1.16387949e+02,  7.35874360e+03],
           [-2.30122094e+06,  6.66691737e+06,  9.68353530e+05,
            -6.61075561e+03, -1.83389708e+03, -2.99265570e+03],
           [-2.30122094e+06,  6.66691737e+06,  9.68353530e+05,
            -6.61075556e+03, -1.83389711e+03, -2.99265570e+03]])

    >>> svece_array = np.asarray(svece)
    >>> xsate = svece_array[:,0:3]
    >>> vsate = svece_array[:,3:]
    >>> ecef2eci(t, xsate, vsate)
    (array([[-5767701.77861271, -3097382.01317908,  2737195.4374    ],
           [ -625224.36062227, -7162713.98329963,    16233.9739    ],
           [-2301220.9380175 ,  6666917.37367665,   968353.5303    ],
           [-2301220.93814086,  6666917.37358403,   968353.5303    ]]), array([[ 3001.4240322 , -6762.10590869, -1280.256     ],
           [-1131.89509013,   116.38794901,  7358.7436    ],
           [-6610.7556061 , -1833.89707562, -2992.6557    ],
           [-6610.75556101, -1833.89710895, -2992.6557    ]]))

    Check that inverse operation returns the original

    >>> svec = ecef2eci(t, svece)
    >>> svece2 = eci2ecef(t,svec)
    >>> print(np.max(np.abs(svece2-svece)) < 1e-8)
    True

    Example with more than two dimensions

    >>> svec = ecef2eci(t, svece)
    >>> svec23 = ecef2eci(t,[[svece, svece, svece],[svece, svece, svece]])
    >>> svec23.shape
    (2, 3, 4, 6)
    >>> np.max(np.abs(svec23 - svec), axis=(-1,-2))
    array([[0., 0., 0.],
           [0., 0., 0.]])

    Special case, point on the Earth surface (Delft), ECEF velocity is assumed to be zero

    >>> xDelftECI, vDelftECI= ecef2eci('2012-01-04 15:00:00',[ 3924687.7018, 301132.7660, 5001910.7746])
    >>> print(xDelftECI, vDelftECI)
    [[ 3507848.04809349 -1785736.98256649  5001910.7746    ]] [[130.21800963 255.79634368   0.        ]]

    Check that the inverse transformation returns the original result

    >>> eci2ecef('2012-01-04 15:00:00', xDelftECI, vDelftECI)
    (array([[3924687.7018,  301132.766 , 5001910.7746]]), array([[0., 0., 0.]]))

    If ECI velocity is not specified in the inverse operation, 'nan' is returned for the ECEF velocity.
    This is intentional, as there is no sensible default for velocites in an ECI.

    >>> eci2ecef('2012-01-04 15:00:00', xDelftECI)
    (array([[3924687.7018,  301132.766 , 5001910.7746]]), array([[nan, nan, nan]]))

    """
    
    # Convert input parameters to all numpy arrays
    
    t = np.atleast_1d(t)
    xsate = np.atleast_2d(xsate)
    vsate = np.atleast_2d(vsate)
    
    # Check input arguments
   
    assert xsate.shape[-1] == 3 or xsate.shape[-1] == 6 , "xsate must have shape (...,3) or (...,6)."
    assert xsate.shape[-2] == 1 or xsate.shape[-2]  == t.shape[-1], "Second to last dimension of xsate must be one or match the length of array t."
    if xsate.shape[-1] == 6:
         assert vsate.size == 0, "xsate has shape (...,6) and contains also velocities, but then vsate must not be used."
         vsate = xsate[...,3:6]
    elif vsate.size == 0:
         vsate = np.zeros(xsate.shape)

    assert vsate.shape[-1] == 3, "vsate must have shape (...,3)."
    assert vsate.shape == xsate[...,0:3].shape, "xsate and vsate must have same shape."

    # extend xsate and vsate to match size of t  (HM: CHECK IF THIS WORKS WITH MORE THAN 2 DIMENSIONS)
    
    if xsate.shape[-2] == 1 and t.shape[-1] > 1:
        xsate = np.tile(xsate,[t.shape[-1],1])
        vsate = np.tile(vsate,[t.shape[-1],1])

    # Compute rotation angle (GMST) around Z-axis
   
    gst, omegae = ut2gmst(t)
    gst = -1*gst

    # Rotate satellite positions around z-axis (ECEF -> ECI)

    xsat = np.zeros(xsate.shape)
    xsat[..., 0] = np.cos(gst)*xsate[..., 0] + np.sin(gst) * xsate[..., 1]
    xsat[..., 1] = -np.sin(gst) * xsate[..., 0] + np.cos(gst) * xsate[..., 1]
    xsat[..., 2] = xsate[..., 2]

    """
    To convert the velocity is more complicated. The velocity in ECEF
    consists of two parts. We find this by differentiating the transformation
    formula for the positions
 
       xsat = R * xsate

    This gives (product rule, and some rewriting), with `_dot` the derivatives

       xsat_dot = R * xsate_dot + R_dot * xsate    <=>
       vsat = R * ( vsate + inv(R)*R_dot * xsate ) <=>
       vsat = R * ( vsate + W * xsate )
 
    with 'W = inv(R)*R_dot = [[ 0, -w[2] w[1]],[ w[2] 0 -w[0]],[-w[1] w[0] 0]]' 
    and with 'w' the angular velocity vector of the ECI frame with respect to
    the ECEF frame, expressed in the ECEF frame. 
    
    For the ECEF to ECI transformation the angular velocity vector is w = [0, 0, w0]'
    with 'w0=2*np.pi*omegae/86400', thus 'W = [[0, -w0, 0], [w0, 0, 0], [0, 0, 0]]', 
    hence 'W * xsate = [ -w0*xsate[1], w0*xsate[0], 0 ]'
    """
 
    # The velocity vector in the ECI is computed as follows

    w0 = 2*np.pi*omegae/86400
    h0 = vsate[...,0] - w0 * xsate[...,1]   
    h1 = vsate[...,1] + w0 * xsate[...,0] 
    
    vsat = np.zeros(vsate.shape)
    vsat[..., 0] = np.cos(gst) * h0 + np.sin(gst) * h1
    vsat[..., 1] = -np.sin(gst) * h0 + np.cos(gst) * h1
    vsat[..., 2] = vsate[..., 2]

    if xsat.shape[-1] == 6:
        xsat[...,3:] = vsat
        return xsat
    else:
        return xsat, vsat

def eci2ecef(t, xsat, vsat=[]):
    """Convert position and velocity from ECI to ECEF reference frame.
    
    Convert position and velocity given in a non-rotating pseudo-inertial 
    Earth Centered Inertial (ECI) reference frame into an Earth Centered Earth 
    Fixed (ECEF) reference frame.
    
    Parameters
    ----------
    t : array_like with shape (n,) or scalar, of type datetime64, str or float
        Universal time as `datetime64` object, ISO date string or sequential date 
        number (days since 1970-01-01).
    xsat : array_like with shape (...,n,3) or (...,n,6) 
        Array with Cartesian coordinates (m) or state vector with positions (m) 
        and velocities (m/s) in an ECI reference frame .
    vsat : array_like with shape (...,n,3), optional
        Array with velocities in ECI reference frame (m/s). If empty, velocities
        are assumed to be part of the state vector 'xsat[...,3:6]', or if `xsat` 
        has shape (...,3), ``xsate = eci2ecef(t, xsat)`` only returns the positions.

    Returns
    -------
    xsate : ndarray with shape (...,n,3) or shape (...,n,6)
        Array with Cartesian coordinates (m) or state vector with positions (m) 
        and velocities (m/s) in ECEF reference frame. 
    vsate : ndarray with shape (...,n,3), optional
        Array with velocities in ECEF reference frame (m/s), only if the input
        parameter `vsat` is not empty.

    Notes
    -----
    The function returns a single ndarray `xsate` with the ECEF state vector when the 
    parameters `xsat` are a state vector (having 6 elements) and `vsat` is empty. When
    `vsat` is not empty, the function returns two ndarrays.

    The function returns ndarrays with at least two dimensions.

    See Also
    --------
    ut2gmst, ecef2eci

    Examples
    --------    
    Example with single statevector (note the two dimensional result(s)) 
    
    >>> eci2ecef('2012-01-04 15:00:00',[-5767701.7786, -3097382.0132, 2737195.4374,  3001.4240, -6762.1059, -1280.2560])
    array([[-3.31253110e+06, -5.64688382e+06,  2.73719544e+06,
             5.67077507e+03, -3.96999941e+03, -1.28025600e+03]])
    
    >>> eci2ecef('2012-01-04 15:00:00',[-5767701.7786, -3097382.0132, 2737195.4374],[ 3001.4240, -6762.1059, -1280.2560])
    (array([[-3312531.10067826, -5646883.81761125,  2737195.4374    ]]), array([[ 5670.77506798, -3969.99940934, -1280.256     ]]))

    Two dimensional examples 

    >>> t = np.array(['2012-01-04 15:00:00', '2012-01-04 16:00:00', '2012-01-04 17:00:00', '2012-01-04 18:00:00'], dtype='datetime64[ns]')
    >>> svec = [[-5767701.7786, -3097382.0132, 2737195.4374,  3001.4240, -6762.1059, -1280.2560],
    ...         [ -625224.3608, -7162713.9833,   16233.9739, -1131.8951,   116.3879,  7358.7436],
    ...         [-2301220.9378,  6666917.3737,  968353.5303, -6610.7556, -1833.8971, -2992.6557],
    ...         [-2301220.9378,  6666917.3737,  968353.5303, -6610.7556, -1833.8971, -2992.6557]]
    >>> eci2ecef(t, svec)
    array([[-3.31253110e+06, -5.64688382e+06,  2.73719544e+06,
             5.67077507e+03, -3.96999941e+03, -1.28025600e+03],
           [ 1.41341049e+06, -7.04965587e+06,  1.62339739e+04,
            -1.63304150e+03, -3.09546050e+02,  7.35874360e+03],
           [-2.45011540e+06,  6.61364798e+06,  9.68353530e+05,
            -6.08570289e+03, -1.80298462e+03, -2.99265570e+03],
           [-6.49857962e+05,  7.02289753e+06,  9.68353530e+05,
            -6.34510204e+03, -1.61905582e+02, -2.99265570e+03]])

    >>> svec_array = np.asarray(svec)
    >>> xsat = svec_array[:,0:3]
    >>> vsat = svec_array[:,3:]
    >>> eci2ecef(t, xsat, vsat)
    (array([[-3312531.10067826, -5646883.81761125,  2737195.4374    ],
           [ 1413410.48702954, -7049655.87125032,    16233.9739    ],
           [-2450115.40038307,  6613647.97952822,   968353.5303    ],
           [ -649857.96174134,  7022897.52893159,   968353.5303    ]]), array([[ 5670.77506798, -3969.99940934, -1280.256     ],
           [-1633.0414957 ,  -309.54604979,  7358.7436    ],
           [-6085.70289335, -1802.98462425, -2992.6557    ],
           [-6345.10203574,  -161.90558206, -2992.6557    ]]))

    Check that inverse operation returns the original

    >>> svece = eci2ecef(t,svec)
    >>> svec2 = ecef2eci(t, svece)
    >>> print(np.max(np.abs(svec2-svec)) < 1e-8)
    True

    Example with more than two dimensions

    >>> svece = eci2ecef(t, svec)
    >>> svece23 = eci2ecef(t,[[svec, svec, svec],[svec, svec, svec]])
    >>> svece23.shape
    (2, 3, 4, 6)
    >>> np.max(np.abs(svece23 - svece), axis=(-1,-2))
    array([[0., 0., 0.],
           [0., 0., 0.]])

    """

    # Convert input parameters to all numpy arrays
    
    t = np.atleast_1d(t)
    xsat = np.atleast_2d(xsat)
    vsat = np.atleast_2d(vsat)

    # Check input arguments
   
    assert xsat.shape[-1] == 3 or xsat.shape[-1] == 6 , "xsat must have shape (...,3) or (...,6)."
    assert xsat.shape[-2]  == t.shape[-1], "Second to last dimension of xsat must match the length of the array t."
    if xsat.shape[-1] == 6:
         assert vsat.size == 0, "xsat has shape (...,6) and contains also velocities, but then vsat must not be used."
         vsat = xsat[...,3:6]
    elif vsat.size == 0:
         vsat = np.zeros(xsat.shape) * np.nan

    assert vsat.shape[-1] == 3, "vsat must have shape (...,3)."
    assert vsat.shape == xsat[...,0:3].shape, "xsat and vsat must have same shape."

    # Compute rotation angle (GMST) around Z-axis
   
    gst, omegae = ut2gmst(t)

    # Rotate satellite positions around z-axis (ECI -> ECEF)

    xsate = np.zeros(xsat.shape)
    xsate[..., 0] = np.cos(gst)*xsat[..., 0] + np.sin(gst) * xsat[..., 1]
    xsate[..., 1] = -np.sin(gst) * xsat[..., 0] + np.cos(gst) * xsat[..., 1]
    xsate[..., 2] = xsat[..., 2]

    """
    To convert the velocity is more complicated. The velocity in ECEF
    consists of two parts. We find this by differentiating the transformation
    formula for the positions

       xsate = R * xsat

    This gives (product rule, and some rewriting), with |_dot| the derivatives

       xsate_dot = R * xsat_dot + R_dot * xsat    <=>
       vsate = R * ( vsat + inv(R)*R_dot * xsat ) <=>
       vsate = R * ( vsat + W * xsat )
    
    with 'W = inv(R)*R_dot = [[ 0, -w[2] w[1]],[ w[2] 0 -w[0]],[-w[1] w[0] 0]]' 
    and with 'w' the angular velocity vector of the ECEF frame with respect to
    the ECI frame, expressed in the ECI frame. 
    
    For the ECI to ECEF transformation the angular velocity vector is w = [0, 0, -w0]'
    with 'w0=2*np.pi*omegae/86400', thus 'W = [[0, +w0, 0], [-w0, 0, 0], [0, 0, 0]]', 
    hence 'W * xsate = [ w0*xsate[1], -w0*xsate[0], 0 ]'
    """
    
    # The velocity vector in the ECEF is computed as follows
    
    w0 = 2*np.pi*omegae/86400
    h0 = vsat[...,0] + w0 * xsat[...,1]   
    h1 = vsat[...,1] - w0 * xsat[...,0]    

    vsate = np.zeros(vsat.shape)
    vsate[..., 0] = np.cos(gst) * h0 + np.sin(gst) * h1
    vsate[..., 1] = -np.sin(gst) * h0 + np.cos(gst) * h1
    vsate[..., 2] = vsat[..., 2]

    if xsat.shape[-1] == 6:
        xsate[...,3:] = vsate
        return xsate
    else:
        return xsate, vsate


# ----------------------------------------------------------------------------
#                      LOOKANGLES
# ----------------------------------------------------------------------------
#
#   satlookangle    Compute table with satellite look angles
#   prtlookangle    Print a table with satellite look angles.

def satlookangle(t, xsat, xobj, verbose=0, swathdef=['VIS', 0, 80, ''], ellips=[True, False]):
    """Compute table with satellite look angles.
    
    Compute a table with lookangles
    
    - zenith and azimuth angles to satellite for an object on the Earth or in space,
    - off-nadir and azimuth angle from satellite to object, the lookangle to the object
      with resepect to the direction of flight of the satellite, and the heading of
      angle of the satellite 
    - range and range-rate between object and satellite

    and flags for visibility, ascending/descending orbit, right/left looking, etc.  
    
    Parameters
    ----------
    t : array_like with shape (n,) or scalar, of type datetime64, str or float
        Universal time as `datetime64` object, ISO date string or sequential date 
        number (days since 1970-01-01).
    xsat : array_like with shape (...,n,6) 
        Array with satellite state vector with positions (m) and velocities (m/s) 
        in ECEF or ECI reference frame. 
    xobj : array_like with shape (3,), (...,n,3) or (...,n,6)
        Array with the position of the object for which the lookangles are to be
        computed. `xobj` is either an array of shape (3,) or (...,n,3) with position(s)
        in the ECEF reference frame, or an array with shape (...,n,6) for statevectors
        with position and velocity in an ECEF or ECI referenence frame. `xobj`  and 
        `xsat` must always be in the same reference frame.
    ellips : list of bool, optional       
        Choice between ellipsoidal or spherical angles for viewing from respectively
        object and satellite. True means using ellipsoidal angles. Default is [true, false]
    swathdef : list, optional        
        Swathdef is a list with four items defining when a satellite is visible. The
        list consists of a label, the incidence angle range, and the look direction 
        (right look'RL', left look 'LL' or both ''). 
        Example for SENTINEL-1, with swath names, minimum and maximum incidence
        angle, right looking 
         
        >>> swathdef=[['IW1', 29.16, 36.59, 'RL' ],  
        ...            'IW2', 34.77, 41.85, 'RL' ],  
        ...            'IW3', 40.04, 46.00, 'RL' ]]
                                                                               
        Default is a 10 deg elevation mask

        >>> swathdef= ['VIS', 0.00, 80.00, '']

    verbose : int, optional    
        Verbosity level, possible values are 0 or 1  (default is 0)

    Returns
    -------
    lookangles : ndarray of floats with shape (...,n,8)
        Array with the angles (rad), range (m) and range-rate (m/s)::   

            0  incidence angle at the object, which is identical to the zenith angle,
            1  azimuth angle from object to the satellite,
            2  off-nadir angle at the satellite to the direction of the object,
            3  azimuth angle at the satellite in the direction of the object,
            4  look angle in the direction of the object with respect to the flight 
               direction of the satellite, 
            5  azimuth angle of the flight direction of the satellite 
            6  range between satellite and object
            7  range rate in the line of sight

    flags : ndarray of str with shape (...,n,3)
        Array with in the first column a ascending/descending flag ``['ASC'|'DSC']``, 
        in the 2nd column the left- or right-looking flag ``['LL'|'RL']``, and in the
        third column the visibility flag ``['VIS|''|<swath>]``, whereby the swath name 
        can be set by the optional `swathdef` parameter.

    Notes
    -----
    `xsat` and `xobj` must be in the same reference frame: so if `xobj` is a 
    shape (...,3) array in ECEF, then also `xsat` must be in the ECEF reference frame.
    If `xobj` is a shape (..,6) array then both ECI and ECEF are possible, but the
    reference frame for `xsat` must be the same as for `xobj`.

    See Also
    --------
    prtlookangle, eci2ecef, ecef2eci

    Examples
    --------
    >>> t = np.array(['2012-01-04 15:00:00', '2012-01-04 16:00:00', '2012-01-04 17:00:00', '2012-01-04 18:00:00'], dtype='datetime64[ns]')
    >>> xsat = [[-3312531.1007, -5646883.8176, 2737195.4374,  5670.7751, -3969.9994, -1280.2560 ],
    ...         [ 1413410.4872, -7049655.8712,   16233.9739, -1633.0415,  -309.5460,  7358.7436 ],
    ...         [-2450115.4006,  6613647.9795,  968353.5303, -6085.7029, -1802.9846, -2992.6557 ],
    ...         [ -649857.9621,  7022897.5289,  968353.5303, -6345.1020,  -161.9056, -2992.6557 ]]
    >>> xobj = [ 3924687.7018, 301132.7660, 5001910.7746]
    
    >>> lookangles, flags = satlookangle(t, xsat, xobj, verbose=1)
    <BLANKLINE>
                            Incidence Satellite  Off-Nadir LookAngle LookAngle
             Satellite Pass     Angle   Azimuth      Angle   Azimuth FlightDir   Heading     Range Rangerate   Flags
                                (deg)     (deg)      (deg)     (deg)     (deg)     (deg)      (km)    (km/s)
    2012-01-04T15:00:00.000   132.479   310.847     41.320    30.533   289.102   101.432  9637.695    -1.507   DSC LL 
    2012-01-04T16:00:00.000   129.061   264.759     43.437    38.002    50.729   347.273  9230.254    -3.284   ASC RL 
    2012-01-04T17:00:00.000   133.693    72.462     40.230   323.433   207.920   115.513  9836.440     4.014   DSC LL 
    2012-01-04T18:00:00.000   128.662    84.296     44.263   321.590   206.077   115.513  9076.242     4.408   DSC LL 
    <BLANKLINE>

    """
 
    # Convert input parameters to all numpy arrays
    
    t = np.atleast_1d(t)
    xsat = np.atleast_2d(xsat)
    xobj = np.atleast_2d(xobj)
    
    # Check input arguments
   
    assert xsat.shape[-1] == 6 , "xsat must have shape (...,6)."
    assert xsat.shape[-2]  == t.shape[-1], "Second to last dimension of xsat must match the length of array t."
    assert xobj.shape[-1] == 3 or xobj.shape[-1]  == 6, "xobj must have shape (...,3) or (...,6)."
    assert xobj.shape[-2] == 1 or xobj.shape[-2]  == t.shape[-1], "Second to last dimension of xobj must be one or match the length of array t."

    # pad xobj with zeros if necessary
    
    if xobj.shape[-1] == 3:
        xobj_ = np.zeros(xobj.shape[:-1] + (6,))
        xobj_[..., :3] = xobj
        xobj = xobj_ 
   
    # Compute position vector from object to satellite, range and rangerate

    xobj2sat = xsat - xobj
    robj2sat = np.sqrt(np.sum(xobj2sat[..., :3]**2, axis=-1))
    rrobj2sat = np.sum(xobj2sat[..., 3:] * xobj2sat[..., :3], axis=-1) / robj2sat

    # Compute azimuth and zenith angle from object point of view
        
    robj = np.sqrt(np.sum(xobj[..., :3]**2, axis=1))     # range to object (observer)
    if ellips[0]:                                        # normal vector from object (observer) 
        plh = xyz2plh(xobj[...,:3])                      # - ellipsoidal
        n0 = np.empty(plh.shape)
        n0[...,0] = np.cos(plh[...,0]) * np.cos(plh[...,1])
        n0[...,1] = np.cos(plh[...,0]) * np.sin(plh[...,1])
        n0[...,2] = np.sin(plh[...,0])
    else:                                        
        n0 = xobj[..., :3] / robj[..., np.newaxis]       # - spherical 
    
    ers = xobj2sat[..., :3] / robj2sat[..., np.newaxis]  # init direction vector from observer to satellite

    ip = np.sum(n0 * ers, axis=-1)
    z1 = np.arccos(ip)
    a1 = np.arctan2(-n0[..., 1] * ers[..., 0] + n0[..., 0] * ers[..., 1], -ip * n0[..., 2] + ers[..., 2])
    a1 += 2*np.pi
    a1 %= 2*np.pi

    # Compute azimuth and nadir angle from satellite point of view

    rsat = np.sqrt(np.sum(xsat[..., :3]**2, axis=-1))
    if ellips[1]:                                      # normal vector at satellite (inverse nadir vector) 
        plh = xyz2plh(xsat[...,:3])                    # - ellipsoidal
        n0sat = np.empty(plh.shape)
        n0sat[...,0] = np.cos(plh[...,0]) * np.cos(plh[...,1])
        n0sat[...,1] = np.cos(plh[...,0]) * np.sin(plh[...,1])
        n0sat[...,2] = np.sin(plh[...,0])
    else:
        n0sat = xsat[..., :3]/rsat[..., np.newaxis]    # - spherical 

    ipsat = np.sum(n0sat*ers, axis=-1)
    z2 = np.arccos(ipsat)
    a2 = np.arctan2(n0sat[..., 1] * ers[..., 0] - n0sat[..., 0] * ers[..., 1], ipsat*n0sat[..., 2] - ers[..., 2])
    a2 += 2*np.pi
    a2 %= 2*np.pi

    # compute the heading and look angle from the satellite point of view

    velsat = np.sqrt(np.sum(xsat[..., 3:]**2, axis=-1))
    evsat = xsat[..., 3:] / velsat[..., np.newaxis]

    ipvel = np.sum(n0sat*evsat, axis=-1)
    heading = np.arctan2(-n0sat[..., 1] * evsat[..., 0] + n0sat[..., 0] * evsat[..., 1], -ipvel*n0sat[..., 2] + evsat[..., 2])
    heading += 2*np.pi
    heading %= 2*np.pi

    lookangle = (4*np.pi + a2 - heading) % (2*np.pi)

    """
    Set ascending/descending and look direction flags

    ASC:  Heading -90 ...  90       RL:  lookangle   0 ... 180     
    DSC:  Heading  90 ... 270       LL:  lookangle 180 ... 360 
    """
    ASCDSC = np.array(['DSC', 'ASC'], dtype=object)
    ascdsc = ASCDSC[ np.array(np.floor((heading-np.pi/2)/np.pi) % 2, int) ]

    LOOKDIR = np.array(['RL', 'LL'], dtype=object)
    lookdir = LOOKDIR[ np.array(np.floor(lookangle/np.pi) % 2, int) ]
   
    # Convert swatchdef from list to 2D numpy array
    swathdef = np.array(swathdef)
    swathdef = swathdef.reshape(-1,swathdef.shape[-1])
    
    # Set swath definition flag

    IW = np.empty(z1.shape, dtype="U12")
    for sw in swathdef:
        swbool = ( z1*180/np.pi >= float(sw[1]) ) & ( z1*180/np.pi <= float(sw[2]) ) 
        if sw[3] != '':
            swbool = swbool & ( lookdir == sw[3] ) 
        IW[swbool] = np.char.add(IW[swbool],'&' + sw[0])
    IW=np.char.strip(IW,'&')

    # Collect output
    
    lookangles = np.zeros(z1.shape + (8,))
    lookangles[..., 0] = z1
    lookangles[..., 1] = a1
    lookangles[..., 2] = z2
    lookangles[..., 3] = a2
    lookangles[..., 4] = lookangle
    lookangles[..., 5] = heading
    lookangles[..., 6] = robj2sat
    lookangles[..., 7] = rrobj2sat

    flags = np.zeros(z1.shape + (3,), dtype=object)
    flags[..., 0] = ascdsc
    flags[..., 1] = lookdir
    flags[..., 2] = IW

    if verbose:
        prtlookangle(t, lookangles, flags)

    return lookangles, flags



def prtlookangle(t, lookangles, flags, titlestr='', tableformat='default'):
    """Print a table with satellite look angles.
 
    Print a table with zenith and azimuth angles to satellite, off-nadir, azimuth and
    lookangle from satellite, satellite heading, range, range-rate, and flags for 
    visibility, ascending/descending orbit, right/left looking.  
    
    Parameters
    ----------
    t : array_like with shape (n,) or scalar, of type datetime64, str or float
        Universal time as `datetime64` object, ISO date string or sequential date 
        number (days since 1970-01-01).
    lookangles : array_like of floats with shape (...,n,8)
        Array with the angles (rad), range (m) and range-rate (m/s)::   

            0  incidence angle at the object, which is identical to the zenith angle,
            1  azimuth angle from object to the satellite,
            2  off-nadir angle at the satellite to the direction of the object,
            3  azimuth angle at the satellite in the direction of the object,
            4  look angle in the direction of the object with respect to the flight 
               direction of the satellite, 
            5  azimuth angle of the flight direction of the satellite 
            6  range between satellite and object
            7  range rate in the line of sight

    flags : array_like of str with shape (...,n,3)
        Array with in the first column a ascending/descending flag ``['ASC'|'DSC']``, 
        in the 2nd column the left- or right-looking flag ``['LL'|'RL']``, and in the
        third column the visibility flag ``['VIS|''|<swath>]``.
    titlestr : str or list of str, optional
        Title string. Use a list for 3- or more dimensional cases of `lookangles`,
        with shape matching dimensions above 2. 
    tableformat :  {'default', 'ers'}, optional
        Table format. The 'default' is 'ers'.

    See Also 
    --------
    satlookangle   

    Examples
    --------
    >>> t = np.array(['2012-01-04 15:00:00', '2012-01-04 16:00:00', '2012-01-04 17:00:00', '2012-01-04 18:00:00'], dtype='datetime64[ns]')
    >>> xsat = [[-3312531.1007, -5646883.8176, 2737195.4374,  5670.7751, -3969.9994, -1280.2560 ],
    ...         [ 1413410.4872, -7049655.8712,   16233.9739, -1633.0415,  -309.5460,  7358.7436 ],
    ...         [-2450115.4006,  6613647.9795,  968353.5303, -6085.7029, -1802.9846, -2992.6557 ],
    ...         [ -649857.9621,  7022897.5289,  968353.5303, -6345.1020,  -161.9056, -2992.6557 ]]
    >>> xobj = [ 3924687.7018, 301132.7660, 5001910.7746]
    
    >>> lookangles, flags = satlookangle(t, xsat, xobj)
    >>> prtlookangle(t, lookangles, flags)
    <BLANKLINE>
                            Incidence Satellite  Off-Nadir LookAngle LookAngle
             Satellite Pass     Angle   Azimuth      Angle   Azimuth FlightDir   Heading     Range Rangerate   Flags
                                (deg)     (deg)      (deg)     (deg)     (deg)     (deg)      (km)    (km/s)
    2012-01-04T15:00:00.000   132.479   310.847     41.320    30.533   289.102   101.432  9637.695    -1.507   DSC LL 
    2012-01-04T16:00:00.000   129.061   264.759     43.437    38.002    50.729   347.273  9230.254    -3.284   ASC RL 
    2012-01-04T17:00:00.000   133.693    72.462     40.230   323.433   207.920   115.513  9836.440     4.014   DSC LL 
    2012-01-04T18:00:00.000   128.662    84.296     44.263   321.590   206.077   115.513  9076.242     4.408   DSC LL 
    <BLANKLINE>

    """

    # Convert input parameters to all numpy arrays
    
    t = np.atleast_1d(t)
    lookangles = np.atleast_2d(lookangles)
    flags = np.atleast_2d(flags)

    # For arrays with dimension higher than 2 call this function recursively
    
    if lookangles.ndim > 2:
        titlestr = np.asarray(titlestr, dtype=object)
        for k in range(lookangles.shape[0]):
            if titlestr.size == 1:
                subtitle = titlestr
            else:
                subtitle = titlestr[k,...]
            prtlookangle(t, lookangles[k,...], flags[k,...], titlestr=subtitle,tableformat=tableformat)
        return
        
    # Check input arguments
   
    assert lookangles.shape[-1] == 8 , "lookangles must have shape (...,8)."
    assert flags.shape[-1] == 3 , "flags must have shape (...,3)."
    assert lookangles.shape[-2]  == t.shape[-1], "Second to last dimension of lookangles must match the length of array t."
    assert flags.shape[-2]  == t.shape[-1], "Second to last dimension of flags must match the length of array t."
    assert lookangles.dtype in [float, int, np.float64, np.int32, np.int64], "Lookangles must be a numeric array."

    assert tableformat in ['default', 'ers'], f"Unsupported table format {tableformat}."

    # If time not datetime64, convert from sequential date number 
   
    if t.dtype.type == np.float64:
        t = np.datetime64('1970-01-01', 'ms')  + t*np.timedelta64(86400000, 'ms') 
    
    # Print optional title

    print("{}{}".format(titlestr, "" if titlestr == "" else "\n"))
    
    # Print table
    
    if tableformat in ['default', 'ers']:
        print('                        Incidence Satellite  Off-Nadir LookAngle LookAngle')
        print('         Satellite Pass     Angle   Azimuth      Angle   Azimuth FlightDir   Heading     Range Rangerate'
              '   Flags')
        print('                            (deg)     (deg)      (deg)     (deg)     (deg)     (deg)      (km)    (km/s)'
              )
        for k in range(t.shape[0]):
            isodate = t[k].astype('datetime64[ms]')
            print("{} {:>9.3f} {:>9.3f}  {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}   {} {} {}".format(
                isodate, lookangles[k, 0] * 180/np.pi, lookangles[k, 1] * 180 / np.pi, lookangles[k, 2] * 180 / np.pi,
                lookangles[k, 3] * 180 / np.pi, lookangles[k, 4] * 180 / np.pi, lookangles[k, 5] * 180 / np.pi,
                lookangles[k, 6] / 1000, lookangles[k, 7] / 1000, flags[k, 0], flags[k, 1], flags[k, 2]))
        print("")

# ----------------------------------------------------------------------------
#                         SERIAL DATE NUMBERS
# ----------------------------------------------------------------------------
#
#  datetime2num   Convert datestring, datetime or np.datetime64 to serial datenumber
#  num2datetime64 Convert serial datenumber to np.datetime64
#  num2datetime   Convert serial datenumber to datetime

def datetime2num(t):
    """Convert array wtih np.datetime64 objects to serial datenumber (float, days since 01-01-1970 00:00).""" 
    t64 = np.array(t, dtype='datetime64')
    return ( t64 - np.datetime64('1970-01-01') ) / np.timedelta64(1, 'D')

def num2datetime64(datenum):
    """Convert array with serial datenumbers (float, days since 01-01-1970 00:00) to np.datetime64 objects.""" 
    return np.datetime64('1970-01-01', 'ms')  + datenum*np.timedelta64(86400000, 'ms')    

def num2datetime(t):
    """Convert array with serial datenumbers (float, days since 01-01-1970 00:00) to datetime objects.""" 
    return num2datetime64(t).astype(datetime)
        