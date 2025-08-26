# coding: utf-8
"""
Coordinate conversions and transformations.

This module provides basic functions for coordinate conversions and transformations
for Earth Centered Earth Fixed Cartesian coordinates, local topocentric 
coordinates, and ellipsoidal geodetic latitude, longitude and height. 

Coordinate conversions in ECEF reference frame

- xyz2plh       Cartesian coordinates to Ellipsoidal coordinates
- plh2xyz       Ellipsoidal coordinates to Cartesian coordinates
- inqell        Semi-major axis, flattening and GM for various ellipsoids
- setell        Set the current ellipsoid

Curvilinear coordinate conversions

- ellcurvature  Multiplication factors to convert latitude and longitude angles to meters.

Coordinate transformations between ECEF and local (topocentric) reference frames

- xyz2neu       Cartesian coordinates to local coordinates (North,East,Up)
- neu2xyz       Local coordinates (North,East,Up) to Cartesian coordinates
- xyz2zas       Cartesian coordinates to zenith angle, azimuth and distance
- zas2xyz       Zenith angle, azimuth and distance to Cartesian coordinates

- ellnormal     Return unit normal vector perpendicular to ellipsoid
- ellrotmatrix  Rotation matrix to transform between ECEF and topocentric frames 
- covtransform  Transform and reformat (compact) co-variance matrices

Print functions

- printcrd      Print a table with coordinates and optional co-variances
- printxyz      Print array with cartesian coordinates
- printplh      Print array with geodetic/ellipsoidal coordinates 
- deg2dms       Convert latitude/longitude to degree, minute, second notation

The functions are optimized for Python's numpy library, it accepts mostly numpy
alike ndarrays with the coordinates (usually three) in the last axis.
"""

__author__ = "Hans van der Marel"
__copyright__ = "Copyright 2022-2024, Hans van der Marel, Delft University of Technology."
__credits__ = ["Hans van der Marel", "Simon van Diepen"]
__license__ = "License Name and Info"
__version__ = "0.9.0"
__maintainer__ = "Hans van der Marel"
__email__ = "h.vandermarel@tudelft.nl"
__status__ = "development"


"""
Created:    20 Aug 2022 by Hans van der Marel from crsutil
Modified:   20 Aug 2022 by Hans van der Marel
             - inqell, xyz2plh and plh2xyz from crsutil with minor to major changes
             - new functions ellnormal, xyz2neu, neu2xyx, xyz2zas, zas2xyz
             - rewrite of the docstrings, reinsert original comments from Matlab
             - major style improvements, modified options, more Pythonic
            16 Jul 2024 by Hans van der Marel
             - new functions ellrotate and covtransform
             - many major and minor changes to other functions
            18 Jul 2024 by Hans van der Marel
             - Minor changes to the docstrings
             - Optional output of rotation matrix in xyz2neu and neu2xyz
            19 Jul 2024 by Hans van der Marel
             - Modified ellnormal, improved docstrings, fix bug in normalize option, 
               added assertion on mode
            22 Jul 2024 by Hans van der Marel
             - port of prtcrd to Python, renamed to printcrd 
             - added functions printxyz, printplh, deg2dms
            27 Jul 2024 by Hans van der Marel
             - new functions setell and ellcurvature
             - changed default for ellipsoid in inqell to 'current'
             - changes to the docstrings
            22 Aug 2024 by Hans van der Marel
             - Major edits to the docstrings to facilitate sphynx

The functions are based the crsutil Malab toolbox written by Hans van der Marel 
starting in 1995. Some of the functions were ported to Python in November 2020
with the help of Ullas Rajvanshi and Simon van Diepen, and became part of the 
crsutil.py module (with only a subset of the Matlab crsutil toolbox).

In August 2022 the crsutil.py was split into two new modules, crstrans.py and
satorb.py, with crstrans.py including new functions for local reference frames,
and a full rewrite, with improved docstrings, reinsertion of original comments,
major style improvements, modified options, more Pythonic.

Copyright 2022-2024, Hans van der Marel, Delft University of Technology. 
"""

# Importing the Libraries
import numpy as np

# ----------------------------------------------------------------------------
#                      GEODETIC FROM/TO CARTESIAN COORDINATES 
# ----------------------------------------------------------------------------
#
#   xyz2plh      - Convert Cartesian coordinates into Ellipsoidal coordinates
#   plh2xyz      - Convert Ellipsoidal coordinates into Cartesian coordinates
#   inqell       - Semi-major axis, flattening and GM for various ellipsoids
#   setell       - Set the current ellipsoid

# Define a dict with the Semi-major axis, inverse flattening and GM for various ellipsoids

ELLIPSOID_PARAMETERS = {
    'AIRY':          [6377563.396, 299.324964, np.nan], 
    'BESSEL':        [6377397.155, 299.1528128, np.nan], 
    'CLARKE':        [6378249.145, 293.465, np.nan],
    'INTERNATIONAL': [6378388.0, 297.00, np.nan],
    'HAYFORD':       [6378388.0, 297.00, 3.986329e14],
    'GRS80':         [6378137.0, 298.257222101, 3.986005e14],
    'WGS84':         [6378137.0, 298.257223563, 3.986005e14],
    'current':       [6378137.0, 298.257223563, 3.986005e14]
}
CURRENT_ELLIPSOID = 'WGS84'

       
def xyz2plh(xyz, ellipsoid='current', method='Bowring', unit='rad'):
    """
    Convert Cartesian XYZ coordinates to Ellipsoidal latitude, longitude and height coordinates.

    Parameters
    ----------
    xyz : array_like with shape (...,3) 
        Cartesian XYZ coordinates. 
    ellipsoid : str or list of floats, default = 'current'
        Text string with the name of the ellipse or a list ``[a, 1/f]`` with the semi-major 
        axis `a` and inverse flattening `1/f`. The current ellipsoid can be set by
        `setell`. Default for the current ellips is 'WGS-84'.
    method : {'Bowring', 'iterative'}, default = 'Bowring'
        Bowring's method is uded by default. Bowring's method is faster, but can 
        only be used on the surface of the Earth. The iterative method is slower 
        and less precise on the surface of the earth, but should be used above 
        10-20 km of altitude.
    unit : {'rad', 'deg'},  default = 'rad'
        Units for the output latitude and longitude.

    Returns
    -------
    plh : ndarray with the same shape (...,3) as `xyz`
        Ellipsoidal coordinates (geographic latitude, longitude and height above the ellipsoid).

    See Also
    --------
    plh2xyz, inqell, setell

    Examples
    --------
    >>> plh = xyz2plh([3925375.1007 , 274488.9665, 5002803.3455], unit='deg')
    >>> print(np.round(plh,4))          
    [52.  4.  0.]

    >>> plhin = [ [50, 0, 0], [51, 1, -4], [52, 4, 2], [53, 3, 3] ]
    >>> xyz = plh2xyz(plhin, unit='deg')
    >>> plh = xyz2plh(xyz, unit='deg')
    >>> print( np.max(np.abs(plh-plhin)) < 1e-9)
    True

    """ 
    
    # Force input array to ndarray and check the shape
    
    xyz = np.array(xyz)
    assert xyz.shape[-1] == 3 , "Cartesian vector xyz must have three coordinates (XYZ)." 
    assert unit in ['rad' ,'deg'], f"Unsupported unit {unit}."
  
    # excentricity e(squared), semi-major (a) and semi - minor (b) axis

    a, f, GM = inqell(ellipsoid)
    
    e2 = 2 * f - f ** 2
    b = (1 - f) * a

    # compute the radius, geographic latitude and N
    
    r = np.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2)

    if method == 'iterative':
        # compute phi via iteration
        Np = xyz[..., 2]
        for i in range(0, 4):
            phi = np.arctan((xyz[..., 2] + e2 * Np) / r)
            N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
            Np = N * np.sin(phi)

    else:
        # compute phi using B.R.Bowring's equation (default method)
        u = np.arctan2(xyz[..., 2] * a, r * b)
        phi = np.arctan2(xyz[..., 2] + (e2 / (1 - e2) * b) * np.sin(u) ** 3, r - (e2 * a) * np.cos(u) ** 3)
        N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
        
    # assemble 

    plh = np.stack(( phi, 
                     np.arctan2(xyz[..., 1], xyz[..., 0]), 
                     r / np.cos(phi) - N  ), axis=-1)
    
    if unit == 'deg':
        plh = np.stack(( plh[...,0] * 180 / np.pi ,plh[...,1] * 180 / np.pi , plh[...,2] ), axis=-1) 

    return plh


def plh2xyz(plh, ellipsoid='current', unit='rad'):
    """
    Convert ellipsoidal coordinates (latitude, longitude and height) to Cartesian coordinates.

    Parameters
    ----------
    plh : array_like shape (...,3) 
        Ellipsoidal coordinates (geographic latitude, longitude and height above the ellipsoid).
    ellipsoid : str or list of floats, default = 'current'
        Text string with the name of the ellipse or a list ``[a, 1/f]`` with the semi-major 
        axis `a` and inverse flattening `1/f`. The current ellipsoid can be set by
        `setell`. Default for the current ellips is 'WGS-84'. 
    unit : {'rad', 'deg'},  default = 'rad'
        Units for the input latitude and longitude.

    Returns
    -------
    xyz : ndarray with the same shape (...,3) as `plh`
        Cartesian XYZ coordinates.
        
    See Also
    --------
    xyz2plh, inqell, setell
           
    Examples
    --------
    >>> xyz = plh2xyz([52, 4, 0], unit='deg')
    >>> print(np.round(xyz,4))
    [3925375.1007  274488.9665 5002803.3455]

    >>> plhin = [ [50, 0, 0], [51, 1, -4], [52, 4, 2], [53, 3, 3] ]
    >>> xyz = plh2xyz(plhin, unit='deg')
    >>> plh = xyz2plh(xyz, unit='deg')
    >>> print( np.max(np.abs(plh-plhin)) < 1e-9)
    True

    """ 

    # Force input array to ndarray, check the shape, optionally convert units
    
    plh = np.array(plh)
    assert plh.shape[-1] == 3 , "Ellipsoidal coordinate vector(s) plh must have three coordinates (Latitude, longitude, height)." 
    assert unit in ['rad' ,'deg'], f"Unsupported unit {unit}."
    if unit == 'deg':
        plh = np.stack(( plh[...,0] * np.pi / 180 ,plh[...,1] * np.pi / 180 , plh[...,2] ), axis=-1) 
  
    # excentricity e(squared) and semi - minor axis

    a, f, GM = inqell(ellipsoid)   
    e2 = 2 * f - f ** 2
    
    # Produce the result
    
    N = a / np.sqrt(1 - e2 * np.sin(plh[..., 0]) ** 2)
    xyz = np.stack((
        (N + plh[..., 2]) * np.cos(plh[..., 0]) * np.cos(plh[..., 1]),
        (N + plh[..., 2]) * np.cos(plh[..., 0]) * np.sin(plh[..., 1]),
        (N - e2 * N + plh[..., 2]) * np.sin(plh[..., 0]) ) , axis=-1 )

    return xyz


def inqell(ellipsoid='current'):
    """
    Semi-major axis, flattening and GM for various ellipsoids.

    Parameters
    ----------
    ellipsoid: str or list of floats, default = 'current'
        Text string with the name of the ellipse or a list ``[a, 1/f]`` or ``[a, 1/f, GM]``  
        with the semi-major axis `a` , inverse flattening `1/f` and optionally `GM`. 
        The current ellipsoid can be set by `setell`. Default for the current ellips is 
        'WGS-84'. 

    Returns
    -------
    a, f, GM: float
        Semi-major axis `a`, flattening `f` and `GM`.
           
    See Also
    --------
    setell, xyz2plh, plh2xyz
    
    Examples
    --------
    >>> a, f, GM = inqell()
    >>> print(a,1/f)
    6378137.0 298.257223563

    >>> a, f, GM = inqell('GRS80')
    >>> print(a,1/f)
    6378137.0 298.257222101

    >>> a, f, _ = inqell([a, 1/f])
    >>> print(a,1/f)
    6378137.0 298.257222101
    
    """ 

    global ELLIPSOID_PARAMETERS

    if isinstance(ellipsoid, str):
        if ELLIPSOID_PARAMETERS.get(ellipsoid) == None:
            raise ValueError(f"Unknown ellipsoid {ellipsoid}.")
        a, finv, GM = ELLIPSOID_PARAMETERS[ellipsoid]
    elif isinstance(ellipsoid, list) or isinstance(ellipsoid, np.ndarray):
        if len(ellipsoid) == 3:
            a, finv, GM  = ellipsoid 
        elif len(ellipsoid) == 2:
            a, finv = ellipsoid
            GM = np.nan
        else:
            raise AssertionError("ellipsoid must be a text string, or a list with 2 or 3 elements.") 
    else:
        raise AssertionError("ellipsoid must be a text string, or a list with 2 or 3 elements.")
    
    return a, 1/finv, GM

def setell(ellipsoid=None):
    """
    Set current ellipsoid.

    Parameters
    ----------
    ellipsoid : None or str or list of floats, default = None
        If None returns the name of the current ellipsoid, if str set the 
        current ellipsoid to one given in str, if list of floats ``[a, 1/f]`` 
        or ``[a, 1/f, GM]`` set a user specified ellipsoid with semi-major axis
        `a` , inverse flattening `1/f` and optionally `GM`. 
        The current ellipsoid is by default 'WGS-84'. 

    Returns
    -------
    current_ellipsoid : str
        Name of the current ellipsoid.
           
    See Also
    --------
    inqell
    
    Examples
    --------
    >>> setell()
    Current ellipsoid: WGS84
    'WGS84'

    >>> setell('GRS80')
    'GRS80'
    
    """ 

    global CURRENT_ELLIPSOID, ELLIPSOID_PARAMETERS

    if isinstance(ellipsoid, str):
        if ELLIPSOID_PARAMETERS.get(ellipsoid) == None:
            raise ValueError(f"Unknown ellipsoid {ellipsoid}.")
        CURRENT_ELLIPSOID = ellipsoid
        ELLIPSOID_PARAMETERS['current'] = ELLIPSOID_PARAMETERS[ellipsoid]
    elif isinstance(ellipsoid, list) or isinstance(ellipsoid, np.ndarray):
        if len(ellipsoid) == 3:
            a, finv, GM  = ellipsoid 
        elif len(ellipsoid) == 2:
            a, finv = ellipsoid
            GM = np.nan
        else:
            raise AssertionError("ellipsoid must be a text string, or a list with 2 or 3 elements.") 
        CURRENT_ELLIPSOID = 'user_defined'
        ELLIPSOID_PARAMETERS['current'] = [ a, finv, GM ]
    elif ellipsoid == None:
        print('Current ellipsoid:', CURRENT_ELLIPSOID)
    else:
        raise AssertionError("ellipsoid must be a text string, or a list with 2 or 3 elements, or None")
    
    return CURRENT_ELLIPSOID


# ----------------------------------------------------------------------------
#                 CONVERSION TO/FROM LOCAL TOPOCENTRIC COORDINATES 
# ----------------------------------------------------------------------------
#
#   xyz2neu      - Cartesian XYZ coordinates to local coordinates (North,East,Up)
#   neu2xyz      - Local coordinates (North,East,Up) to Cartesian (XYZ coordinates)
#
#   xyz2zas      - Cartesian coordinates to Zenith angle, azimuth and distance
#   zas2xyz      - Zenith angle, azimuth and distance to cartesian coordinates
#
#   ellcurvature - Multiplication factors to convert latitude and longitude angles to meters.
#   ellnormal    - Unit normal vector (ECEF) at a given location perpendicular to the ellipsoid.
#   ellrotmatrix - Rotation matrix/matrices to transform coordinate differences.
#   covtransform - Transform and reformat (compact) co-variance matrices

def xyz2neu(xyz, ref, origin='ref', mode='plh', unit='rad', rotmatrix=False):
    """
    Convert Cartesian XYZ coordinates to local coordinates (North,East,Up).

    Parameters
    ----------
    xyz : array_like with shape (...,3) 
        Cartesian XYZ coordinates. The origin is either in the center of the Earth or at the
        position given by the second argument `ref`.
    ref : array_like with shape (...,3) or (...,2)
        Reference position(s) on the ellipsoid. The type of coordinates are specified by `mode`.
        `ref` can be a vector with a single coordinate triplet/doublet or have a similar shape
        to `xyz`. The North, East, Up coordinates for the(se) point(s) are (0, 0, 0). In case `ref`
        is a single coordinate triplet/doublet then the same reference points is used for all
        the points in `xyz`. 
    origin : {'ref', 'ecef'}, default='ref'
        Origin of the coordinates in `xyz`. If origin='ref' then the XYZ coordinates in `xyz`
        are with respect to the point(s) given in `ref`. The other possibility is coordinates
        in the ECEF reference frame with the origin at the center of the Earth.
    mode : {'plh', 'xyz','normal'}, default = 'plh'
        Coordinate type for `ref`. Possible values are:    
            
         - 'plh' : `ref` contains the geographic latitude and longitude with the unit specified by 
           `unit`. A third coordinate with the height is optional.
         - 'xyz' : `ref` contains cartesian XYZ coordinates in the ECEF of a point (close) to the 
           ellipsoid. The normal vector is computed for a point on the ellipsoid above or below `ref`.
         - 'normal' : `ref` contains a normal vector with Cartesian coordinates. It does
           not necessarily have to be a unit vector (the function returns the unit vector). This
           options only works when the origin of the `xyz` coordinates is `ref` (origin='ref').
           
    unit : {'rad', 'deg'},  default = 'rad'
        Units for latitude and longitude, only useful in case mode='plh' option is used.
    rotmatrix : bool, default=False
        Output the rotation matrix(es) as an optional output argument. The rotation matrix 
        can be used for instance to convert the covariance matrix (when available and needed).

    Returns
    -------
    neu : ndarray with shape (...,3) similar to `xyz`
        Local North, East, Up coordinates `neu` with respect to the point(s) in `ref`.
    R : ndarray with shape (...,3,3), optional
        Rotation matrix(es), with ``neu = dxyz @ R``. In case `ref` is a vector `R` is a 3-by-3 matrix , 
        in case `ref` is a multidimensional array, `R` has one more dimension than `ref` with
        shape (...,3,3). The rotation matrix can be used for instance to convert the covariance matrix 
        (when available and needed). The rotation matrix is the "transpose" of the rotation matrix of 
        `xyz2neu` (but the same as `ellrotmatrix`).

    Notes
    -----
    The rotation matrix differs from the Matlab version in the sense that it is the transpose.
    This is intentional because of differences between Python and Matlab array storage, broadcasting 
    rules, and matrix multiplication. Possible methods are::

       neu = dxyz @ R
       neu = np.matmul(dxyz, R)
       neu = np.einsum('...j,...ji',dxyz,R)
       
    The third method using Einstein summation is the most general. The first two methods only 
    work for dxyz.shape (:,3) or (3,), and with R.shape(3,3), but fail with higher dimensions. 
    The rotation matrix also differs between the inverse functions `xyz2neu` and `neu2xyz` who return
    transposed versions to each other so that the above mathematics remain the same.          
    
    To convert covariance matrices we recommend to use the function covtransform.

    See Also
    --------
    neu2xyz, covtransform, ellnormal, ellrotmatrix
          
    Examples
    --------
    Create some test data
    
    >>> ref = [ 52*np.pi/180, 4*np.pi/180 ]
    >>> refxyz = plh2xyz([52, 4, 0],unit='deg')
    >>> dxyz = [1, 1, 1]
    >>> xyz = refxyz + np.array(dxyz)

    The following examples all have the same ouput
    
    >>> neu = xyz2neu(dxyz, ref)
    >>> neu = xyz2neu(dxyz, [52, 4], unit='deg')
    >>> neu = xyz2neu(dxyz, refxyz, mode='xyz')
    >>> neu = xyz2neu(xyz , refxyz, mode='xyz', origin='ecef')
    >>> neu = xyz2neu(xyz, [52, 4, 0], origin='ecef', unit='deg')
    >>> print(neu)
    [-0.22539858  0.92780758  1.44511888]

    You can use only latitude and longitude in `ref`, but not in ECEF mode where 
    `ref` MUST have three coordinates
    
    >>> neu = xyz2neu(dxyz, [52, 4], unit='deg')  # ok
    >>> neu = xyz2neu(xyz, [52 , 4], unit='deg', origin='ecef')  # fails
    Traceback (most recent call last):
      ...
    AssertionError: Reference coordinate vector ref must have three coordinates when used in combination with origin='ecef'.

    Examples with multi dimensional coordinate input `xyz`. The second parameter `ref` can either
    be a row vector or have the same shape as the first input `xyz`. The number of dimensions is not
    restricted to two.

    >>> dxyz = [[ 1, 1, 1],[ 2, 2, 2],[ 3, 0, 0],[ 0, 4, 1]] 
    >>> neu = xyz2neu(dxyz, ref)                    # ref is a single coordinate doublet
    >>> print(neu)
    [[-0.22539858  0.92780758  1.44511888]
     [-0.45079715  1.85561515  2.89023776]
     [-2.3582736  -0.20926942  1.84248526]
     [ 0.39578607  3.9902562   0.95979625]]
    >>> neu1 = xyz2neu(dxyz, [ref ,ref, ref, ref])   # matching shapes (ref can be different for each point)
    >>> neu2 = xyz2neu(dxyz, [ refxyz ,refxyz, refxyz, refxyz], mode='xyz')
    >>> print( np.max(np.abs(neu1-neu)) < 1e-8, np.max(np.abs(neu2-neu)) < 1e-8)
    True True
    
    Example with additional rotation matrix output

    >>> neu, R = xyz2neu(dxyz, ref, rotmatrix=True)  # Rotation matrix output
    >>> print(R)
    [[-0.7860912  -0.06975647  0.61416175]
     [-0.05496885  0.99756405  0.04294637]
     [ 0.61566148  0.          0.78801075]]
    >>> print(np.all(neu - np.array(dxyz) @ R < 1e-14))   
    True
    
    The following two examples do not give the same results because the up-direction 
    is different (geodetic versus astronomic)

    >>> xyz2neu([1, 1, 1], refxyz, mode='xyz')     # same result as neu = xyz2neu(dxyz,ref)
    array([-0.22539858,  0.92780758,  1.44511888])
    >>> xyz2neu([1, 1, 1], refxyz, mode='normal')
    array([-0.2206844 ,  0.92780758,  1.44584629])
    
    """ 
        
    # Force input arrays to ndarray and check the shapes
    
    xyz = np.array(xyz)
    ref = np.array(ref)
    
    assert xyz.shape[-1] == 3 , "Cartesian vector xyz must have three coordinates (XYZ)." 
    assert origin in ['ref' ,'ecef'], f"Unsupported origin {origin}."

    assert ref.shape[-1] == 2 or ref.shape[-1] == 3  , "Reference coordinate vector ref must have two or three coordinates." 
    if ref.ndim > 1:
        t1=xyz.shape
        t2=ref.shape
        assert t1[:len(t1)-1] == t2[:len(t2)-1], "Reference coordinate vector must have dimension one or similar shape to xyz."

    # coordinates with respect to the reference point
    
    if origin == 'ecef':
        if mode == 'plh': 
            assert ref.shape[-1] == 3  , "Reference coordinate vector ref must have three coordinates when used in combination with origin='ecef'." 
            xyz = xyz - plh2xyz(ref, unit=unit)
        elif mode == 'xyz': 
            xyz = xyz - ref
        else:
            raise ValueError("Normal vector mode is not supported with origin='ecef'.")

    # Compute normal vector at the reference position(s)
    
    n = ellnormal(ref, mode=mode, unit=unit)
    
    # Do the actual computation 

    # neu = [ ( -n(:,1).*xyz(:,1) - n(:,2).*xyz(:,2) ) .* n(:,3) ./ cphi + cphi.*xyz(:,3)     ...
    #         ( -n(:,2).*xyz(:,1) + n(:,1).*xyz(:,2) ) ./  cphi                              ...
    #            n(:,1).*xyz(:,1) + n(:,2).*xyz(:,2) + n(:,3).*xyz(:,3) ]
    
    cphi = np.sqrt(1-n[...,2]**2)
    ip = n[...,0]*xyz[...,0] + n[...,1]*xyz[...,1] + n[...,2]*xyz[...,2]

    neu = np.stack((  ( ip * -n[...,2]  + xyz[...,2] ) / cphi ,
                      ( -n[...,1]*xyz[...,0] + n[...,0]*xyz[...,1] ) /  cphi ,
                        ip  ), axis=-1 )
      
    # Optionally output the rotation matrix/matrices (see also ellrotmatrix)

    if not rotmatrix:
        return neu

    else:
        R = np.array( [[ -n[...,0]*n[...,2]/cphi ,  -n[...,1]/cphi        ,  n[...,0] ] ,  
                       [ -n[...,1]*n[...,2]/cphi ,   n[...,0]/cphi        ,  n[...,1] ] ,
                       [                    cphi ,   np.zeros_like(cphi)  ,  n[...,2] ] ] )  
        if R.ndim > 2:
            R = np.moveaxis(R, 0, -1)
            R = np.moveaxis(R, 0, -1)

        return neu, R



def neu2xyz(neu, ref, origin='ref', mode='plh', unit='rad', rotmatrix=False):
    """
    Convert local coordinates (North,East,Up) to Cartesian coordinates aligned to the ECEF frame.

    Parameters
    ----------
    neu : array_like with shape (...,3) 
        Local North, East, Up coordinates with respect to the point(s) in `ref`.
    ref : array_like with shape (...,3) or (...,2)
        Reference position(s) on the ellipsoid. The type of coordinates are specified by `mode`.
        'ref' can be a vector with a single coordinate triplet/doublet or have a similar shape
        to `neu`. The North, East, Up coordinates for the(se) point(s) are (0, 0, 0). In case `ref`
        is a single coordinate triplet/doublet then the same reference points is used for all
        the points in `neu`. 
    origin : {'ref', 'ecef'}, default='ref'
        Origin of the coordinates in the output XYZ coordinates. If origin='ref' then the XYZ 
        coordinates in the output are with respect to the point(s) given in `ref`. The other 
        possibility is coordinates in the ECEF reference frame with the origin at the center 
        of the Earth.
    mode : {'plh', 'xyz','normal'}, default = 'plh'
        Coordinate type for `ref`. Possible values are: 
            
         - 'plh' : `ref` contains the geographic latitude and longitude with the unit 
           specified by `unit`. A third coordinate with the height is optional.
         - 'xyz' : `ref` contains cartesian XYZ coordinates in the ECEF of a point (close) 
           to the ellipsoid. The normal vector is computed for a point on the ellipsoid 
           above or below `ref`.
         - 'normal' : `ref` contains a normal vector with Cartesian coordinates. It does 
           not necessarily have to be a unit vector (the function returns the unit vector).
           
    unit : {'rad', 'deg'},  default = 'rad'
        Units for latitude and longitude, only useful in case the 'plh' options is used.
    rotmatrix : bool, default=False
        Output the rotation matrix(es) as an optional output argument. The rotation matrix 
        can be used for instance to convert the covariance matrix (when available and needed).

    Returns
    -------
    xyz : ndarray with shape (...,3) similar to `neu`
        Cartesian XYZ coordinates. The origin is either in the center of the Earth or at the
        position given by `ref`.
    R : ndarray with shape (...,3,3), optional
        Rotation matrix(es), with ``dxyz = neu @ R``. In case `ref` is a vector `R` is a 3-by-3 matrix , 
        in case `ref` is a multidimensional array, `R` has one more dimension than `ref` with
        shape (...,3,3). The rotation matrix can be used for instance to convert the covariance matrix 
        (when available and needed). The rotation matrix is the "transpose" of the rotation matrix of 
        `xyz2neu` and `ellrotmatrix`. 

    Notes
    -----
    The rotation matrix differs from the Matlab version in the sense that it is the transpose.
    This is intentional because of differences between Python and Matlab array storage, broadcasting 
    rules, and matrix multiplication. Possible methods are::

       dxyz = neu @ R
       dxyz = np.matmul(neu, R)
       dxyz = np.einsum('...j,...ji',neu,R)
       
    The third method using Einstein summation is the most general. The first two methods only 
    work for dxyz.shape (:,3) or (3,), and with R.shape(3,3), but fail with higher dimensions. 
    The rotation matrix also differs between the inverse functions `xyz2neu` and `neu2xyz` who return
    transposed versions to each other so that the above mathematics remain the same.          
    
    To convert covariance matrices we recommend to use the function covtransform.

    See Also
    --------
    xyz2neu, covtransform, ellnormal, ellrotmatrix                     

    Examples
    --------
    Create some test data

    >>> ref = [ 52*np.pi/180, 4*np.pi/180 ]
    >>> refxyz = plh2xyz([52, 4, 0],unit='deg')
    >>> neu = [1, 1, 1]
    
    The following examples all have the same output as coordinate delta's

    >>> dxyz = neu2xyz(neu, ref)
    >>> dxyz = neu2xyz(neu, [52, 4], unit='deg')
    >>> dxyz = neu2xyz(neu, refxyz, mode='xyz')
    >>> print(dxyz)
    [-0.24168592  0.98554157  1.40367223]

    The following examples refer to the same point, but as absolute coordinates

    >>> xyz = neu2xyz(neu, refxyz, mode='xyz', origin='ecef')
    >>> xyz = neu2xyz(neu, [52, 4, 0], origin='ecef', unit='deg')
    >>> print(np.round(xyz,4))
    [3925374.8591  274489.952  5002804.7492]
    
    You can use only latitude and longitude in `ref`, but not in ECEF mode where 
    `ref` MUST have three coordinates

    >>> dxyz = neu2xyz(neu, [52, 4], unit='deg')
    >>> xyz = neu2xyz(neu, [52 , 4], unit='deg', origin='ecef')
    Traceback (most recent call last):
      ...
    AssertionError: Reference coordinate vector ref must have three coordinates when used in combination with origin='ecef'.

    Examples with multi dimensional coordinate input `neu`. The second parameter `ref` can either
    be a row vector or have the same shape as the first input `neu`. The number of dimensions is not
    restricted to two.

    >>> neu = [[ 1, 1, 1],[ 2, 2, 2],[ 3, 0, 0],[ 0, 4, 1]] 
    >>> dxyz = neu2xyz(neu, ref)                    # ref is a single coordinate doublet
    >>> print(dxyz)
    [[-0.24168592  0.98554157  1.40367223]
     [-0.48337184  1.97108314  2.80734446]
     [-2.3582736  -0.16490655  1.84698443]
     [ 0.33513586  4.03320257  0.78801075]]
    >>> dxyz1 = neu2xyz(neu, [ref ,ref, ref, ref])   # matching shapes (ref can be different for each point)
    >>> dxyz2 = neu2xyz(neu, [ refxyz ,refxyz, refxyz, refxyz], mode='xyz')
    >>> print( np.max(np.abs(dxyz1-dxyz)) < 1e-8, np.max(np.abs(dxyz2-dxyz)) < 1e-8)
    True True

    Example with additional rotation matrix output

    >>> dxyz, R = neu2xyz(neu, ref, rotmatrix=True)  # Rotation matrix output
    >>> print(R)
    [[-0.7860912  -0.05496885  0.61566148]
     [-0.06975647  0.99756405  0.        ]
     [ 0.61416175  0.04294637  0.78801075]]
    >>> print(np.all(dxyz - np.array(neu) @ R < 1e-14))  
    True
    
    The following two examples do not give the same results because the up-direction 
    is different (geodetic versus astronomic)

    >>> neu2xyz([1, 1, 1], refxyz, mode='xyz')     # same result as dxyz = neu2xyz(neu,ref)
    array([-0.24168592,  0.98554157,  1.40367223])
    >>> neu2xyz([1, 1, 1], refxyz, mode='normal')
    array([-0.23711835,  0.98586097,  1.40422685])
        
    """ 
    
    # Force input arrays to ndarray and check the shapes
    
    neu = np.array(neu)
    ref = np.array(ref)
    
    assert neu.shape[-1] == 3 , "Coordinate vector neu must have three coordinates (NEU)." 
    assert origin in ['ref' ,'ecef'], f"Unsupported origin {origin}."

    assert ref.shape[-1] == 2 or ref.shape[-1] == 3  , "Reference coordinate vector ref must have two or three coordinates." 
    if ref.ndim > 1:
        t1=neu.shape
        t2=ref.shape
        assert t1[:len(t1)-1] == t2[:len(t2)-1], "Reference coordinate vector must have dimension one or similar shape to neu."

    # Compute normal vector at the reference position(s)
    
    n = ellnormal(ref, mode=mode, unit=unit)
    
    # Do the actual computation 
    
    # xyz = [  -n(:,1).*n(:,3)./cphi.*neu(:,1)  - n(:,2)./cphi.*neu(:,2) + n(:,1).*neu(:,3)   ...
    #          -n(:,2).* n(:,3)./cphi.*neu(:,1) + n(:,1)./cphi.*neu(:,2) + n(:,2).*neu(:,3)   ...
    #           cphi.*neu(:,1)                                           + n(:,3).*neu(:,3)    ]

    cphi = np.sqrt(1-n[...,2]**2)

    xyz = np.stack(( 
        ( -n[...,0]*n[...,2]*neu[...,0] - n[...,1]*neu[...,1] ) / cphi + n[...,0]*neu[...,2]   ,
        ( -n[...,1]*n[...,2]*neu[...,0] + n[...,0]*neu[...,1] ) / cphi + n[...,1]*neu[...,2]   ,
                        cphi*neu[...,0]                                + n[...,2]*neu[...,2]   ) , axis=-1)
        
    # coordinates with respect to the reference point
    
    if origin == 'ecef':
        if mode == 'plh': 
            assert ref.shape[-1] == 3  , "Reference coordinate vector ref must have three coordinates when used in combination with origin='ecef'." 
            xyz = xyz + plh2xyz(ref, unit=unit)
        elif mode == 'xyz': 
            xyz = xyz + ref
        else:
            raise ValueError("Normal vector mode is not supported with origin='ecef'.")

    # Optionally output the rotation matrix/matrices (see also ellrotmatrix)

    if not rotmatrix:
        return xyz

    else:
        R = np.array( [[ -n[...,0]*n[...,2]/cphi ,  -n[...,1]/cphi        ,  n[...,0] ] ,  
                       [ -n[...,1]*n[...,2]/cphi ,   n[...,0]/cphi        ,  n[...,1] ] ,
                       [                    cphi ,   np.zeros_like(cphi)  ,  n[...,2] ] ] )  
        if R.ndim > 2:
            R = np.moveaxis(R, 0, -1)
            R = np.moveaxis(R, 0, -1)

        R = np.swapaxes(R,-2,-1)    # return the transpose
        
        return xyz, R


def xyz2zas(xyz, ref, origin='ref', mode='plh', refunit='rad', zasunit='rad/m'):
    """
    Cartesian XYZ coordinates to zenith angle, azimuth and distance.

    Parameters
    ----------
    xyz : array_like with shape (...,3) 
        Cartesian XYZ coordinates. The origin is either in the center of the Earth or at the
        position given by `ref`.
    ref : array_like shape (...,3) or (...,2)
        Reference position(s) on the ellipsoid. The type of coordinates are specified by `mode`.
        `ref` can be a vector with a single coordinate triplet/doublet or have a similar shape
        to `xyz`. The North, East, Up coordinates for the(se) point(s) are (0, 0, 0). In case `ref`
        is a single coordinate triplet/doublet then the same reference points is used for all
        the points in `xyz`. 
    origin : {'ref', 'ecef'}, default='ref'
        Origin of the coordinates in `xyz`. If origin='ref' then the XYZ coordinates in `xyz`
        are with respect to the point(s) given in `ref`. The other possibility is coordinates
        in the ECEF reference frame with the origin at the center of the Earth.
    mode : {'plh', 'xyz','normal'}, default = 'plh'
        Coordinate type for `ref`. Possible values are:
            
         - 'plh' : `ref` contains the geographic latitude and longitude with the unit 
           specified by `unit`. A third coordinate with the height is optional.
         - 'xyz' : `ref` contains cartesian XYZ coordinates in the ECEF of a point (close) 
           to the ellipsoid. The normal vector is computed for a point on the ellipsoid 
           above or below `ref`.
         - 'normal' : `ref` contains a normal vector with Cartesian coordinates. It does 
           not necessarily have to be a unit vector (the function returns the unit vector).
           
    refunit : {'rad', 'deg'},  default = 'rad'
        Units for latitude and longitude, only useful in case the 'plh' options is used.
    zasunit : {'rad/m', 'deg/m'},  default = 'rad/m'
        Units for zenith angle and azimuth.

    Returns
    -------
    zas : ndarray with shape similar to `xyz`.
        Zenith angle (z), azimuth angle (a) and distance (s) from `ref` to `xyz`.
           
    Examples
    --------
    Define a test case
    
    >>> ref = [52*np.pi/180, 4*np.pi/180 ]
    >>> refxyz = plh2xyz([52, 4, 0],unit='deg')
    >>> dxyz = [1, 1, 1]
    >>> xyz = refxyz + np.array(dxyz)

    The following examples have the same result
    
    >>> zas = xyz2zas(dxyz, ref)
    >>> zas = xyz2zas(dxyz, [52, 4], refunit='deg')
    >>> zas = xyz2zas(dxyz, refxyz, mode='xyz')
    >>> zas = xyz2zas(xyz , refxyz, mode='xyz', origin='ecef')
    >>> zas = xyz2zas(xyz, [52, 4, 0], origin='ecef', refunit='deg')
    >>> print(zas)
    [0.58386231 1.80911628 1.73205081]
    
    Check that the inverse function returns the original
    
    >>> zas2xyz(zas,ref)
    array([1., 1., 1.])
    
    The following examples also have all the same result

    >>> dxyz = [[ 1, 1, 1],[ 2, 2, 2],[ 3, 0, 0],[ 0, 4, 1]] 
    >>> zas = xyz2zas(dxyz, ref)                    # ref is a single coordinate doublet
    >>> zas = xyz2zas(dxyz, refxyz, mode='xyz')    
    >>> zas = xyz2zas(dxyz, [ref ,ref, ref, ref])   # matching shapes (ref can be different for each point)
    >>> zas = xyz2zas(dxyz, [ refxyz ,refxyz, refxyz, refxyz], mode='xyz')
    >>> print(zas)
    [[ 0.58386231  1.80911628  1.73205081]
     [ 0.58386231  1.80911628  3.46410162]
     [ 0.90947297 -3.05308608  3.        ]
     [ 1.33585617  1.47193157  4.12310563]]
    
    The following two examples do not give the same results because the up-direction 
    is different (geodetic versus astronomic)

    >>> dxyz = [1, 1, 1]
    >>> xyz2zas(dxyz, refxyz, mode='xyz')     # same result as z, a, s = xyz2zas(dxyz,ref)
    array([0.58386231, 1.80911628, 1.73205081])
    >>> xyz2zas(dxyz, refxyz, mode='normal')
    array([0.58310003, 1.80431289, 1.73205081])
       
    """ 
    
    # Force input arrays to ndarray and check the shapes
    
    xyz = np.array(xyz)
    ref = np.array(ref)
    
    assert xyz.shape[-1] == 3 , "Cartesian vector xyz must have three coordinates (XYZ)." 
    assert origin in ['ref' ,'ecef'], f"Unsupported origin {origin}."

    assert ref.shape[-1] == 2 or ref.shape[-1] == 3  , "Reference coordinate vector ref must have two or three coordinates." 
    if ref.ndim > 1:
        t1=xyz.shape
        t2=ref.shape
        assert t1[:len(t1)-1] == t2[:len(t2)-1], "Reference coordinate vector must have dimension one or similar shape to xyz."

    # coordinates with respect to the reference point
    
    if origin == 'ecef':
        if mode == 'plh': 
            assert ref.shape[-1] == 3  , "Reference coordinate vector ref must have three coordinates when used in combination with origin='ecef'." 
            xyz = xyz - plh2xyz(ref, unit=refunit)
        elif mode == 'xyz': 
            xyz = xyz - ref
        else:
            raise ValueError("Normal vector mode is not supported with origin='ecef'.")

    # Compute normal vector at the reference position(s)
    
    n = ellnormal(ref, mode=mode, unit=refunit)
   
    # Do the actual computation 

    # neu = [ ( -n(:,1).*xyz(:,1) - n(:,2).*xyz(:,2) ) .* n(:,3) ./ cphi + cphi.*xyz(:,3)     ...
    #         ( -n(:,2).*xyz(:,1) + n(:,1).*xyz(:,2) ) ./  cphi                              ...
    #            n(:,1).*xyz(:,1) + n(:,2).*xyz(:,2) + n(:,3).*xyz(:,3) ]
    #
    # cphi= sqrt(1-n(:,3).^2);
    # ip=n(:,1).*xyz(:,1) + n(:,2).*xyz(:,2) + n(:,3).*xyz(:,3);
    #
    # neu = [ (  ip .* -n(:,3)  + xyz(:,3) ) ./ cphi               ...
    #         ( -n(:,2).*xyz(:,1) + n(:,1).*xyz(:,2) ) ./  cphi    ...
    #            ip                                                  ];
    #
    # s = sqrt( xyz(:,1).*xyz(:,1) + xyz(:,2).*xyz(:,2) + xyz(:,3).*xyz(:,3) );
    # z = acos(neu(:,3)./s);
    # a = atan2(neu(:,2),neu(:,1));
    
    ip = n[...,0]*xyz[...,0] + n[...,1]*xyz[...,1] + n[...,2]*xyz[...,2]

    s = np.sqrt( xyz[...,0]**2 + xyz[...,1]**2 + xyz[...,2]**2 )
    z = np.arccos( ip / s )
    a = np.arctan2(  -n[...,1]*xyz[...,0] + n[...,0]*xyz[...,1] ,   ip * -n[...,2]  + xyz[...,2] )

    if zasunit in ['deg', 'deg/m']:
        r2d = 180 / np.pi
    else:
        r2d = 1.

    zas = np.stack(( r2d*z, r2d*a , s), axis=-1 )
      
    return zas


def zas2xyz(zas, ref, origin='ref', mode='plh', refunit='rad', zasunit='rad/m'):
    """
    Zenith angle, azimuth and distance to cartesian XYZ coordinates.

    Parameters
    ----------
    zas : array_like with shape (...,3) 
        Zenith angle, azimuth angle and distance from `ref` to `xyz`.
    ref : array_like with shape (...,3) or (...,2)
        Reference position(s) on the ellipsoid. The type of coordinates are specified by `mode`.
        `ref` can be a vector with a single coordinate triplet/doublet or have a similar shape
        to `neu`. The North, East, Up coordinates for the(se) point(s) are (0, 0, 0). In case `ref`
        is a single coordinate triplet/doublet then the same reference points is used for all
        the points in `neu`. 
    origin : {'ref', 'ecef'}, default='ref'
        Origin of the coordinates in the output XYZ coordinates. If origin='ref' then the XYZ 
        coordinates in the output are with respect to the point(s) given in `ref`. The other 
        possibility is coordinates in the ECEF reference frame with the origin at the center 
        of the Earth.
    mode : {'plh', 'xyz','normal'}, default = 'plh'
        Coordinate type for `ref`. Possible values are:
            
         - 'plh' : `ref` contains the geographic latitude and longitude with the unit 
           specified by `unit`. A third coordinate with the height is optional.
         - 'xyz' : `ref` contains cartesian XYZ coordinates in the ECEF of a point (close) 
           to the ellipsoid. The normal vector is computed for a point on the ellipsoid 
           above or below `ref`.
         - 'normal' : `ref` contains a normal vector with Cartesian coordinates. It does 
           not necessaraly have to be a unit vector (the function returns the unit vector).
           
    refunit : {'rad', 'deg'},  default = 'rad'
        Units for latitude and longitude, only useful in case the 'plh' options is used.
    zasunit : {'rad/m', 'deg/m'},  default = 'rad/m'
        Units for latitude and longitude, only useful in case the 'plh' options is used.

    Returns
    -------
    xyz : ndarray with shape (...,3) 
        Cartesian XYZ coordinates. The origin is either in the center of the Earth or at the
        position given by `ref`.
           
    Examples
    --------
    Define a test case
    
    >>> ref = [52*np.pi/180, 4*np.pi/180, 0 ]
    >>> refxyz = plh2xyz([52, 4, 0],unit='deg')
    >>> zas = [0.1, np.pi/2, 1000]

    The following examples have the same result

    >>> dxyz = zas2xyz(zas, ref)
    >>> dxyz = zas2xyz(zas, [52, 4, 0], refunit='deg')
    >>> dxyz = zas2xyz(zas, refxyz, mode='xyz')
    >>> print(dxyz)
    [604.12947719 142.32204802 784.07398212]

    Same, but with `ref` coordinates added

    >>> xyz = zas2xyz(zas, refxyz, origin='ecef', mode='xyz')
    >>> print(np.round(xyz,4))
    [3925979.2303  274631.2885 5003587.4194]

    The following two examples do not give the same results because the up-direction 
    is different (geodetic versus astronomic)

    >>> zas2xyz(zas, refxyz, mode='xyz')   # same result as dxyz=zas2xyz(zas,ref)
    array([604.12947719, 142.32204802, 784.07398212])
    >>> zas2xyz(zas, refxyz, mode='normal')
    array([606.67710196, 142.50019529, 782.07198409])

    Multi dimensional examples with all the same result

    >>> zas = xyz2zas([[ 1, 1, 1],[ 2, 2, 2],[ 3, 0, 0],[ 0, 4, 1]],ref)
    >>> dxyz = zas2xyz(zas, ref)                    # ref is a single coordinate doublet
    >>> dxyz = zas2xyz(zas, refxyz, mode='xyz')    
    >>> dxyz = zas2xyz(zas, [ref ,ref, ref, ref])   # matching shapes (ref can be different for each point)
    >>> print(np.round(dxyz,14))
    [[1. 1. 1.]
     [2. 2. 2.]
     [3. 0. 0.]
     [0. 4. 1.]]
    
    """ 
    
    # Force input array zas to ndarray, convert to neu, call neu2xyz to do the work
    
    zas = np.array(zas)
    
    if zasunit in ['deg', 'deg/m']:
        d2r = np.pi/180
    else:
        d2r = 1.

    neu= np.stack(( zas[...,2]*np.cos(d2r*zas[...,1])*np.sin(d2r*zas[...,0]) ,
                    zas[...,2]*np.sin(d2r*zas[...,1])*np.sin(d2r*zas[...,0]) ,
                    zas[...,2]*np.cos(d2r*zas[...,0]) )       , axis=-1)
        
    xyz = neu2xyz(neu, ref, origin=origin, mode=mode, unit=refunit)
    
    return xyz


def ellcurvature(lat, height, unit='rad', ellipsoid='current'):
    """
    Multiplication factors to convert latitude and longitude angles to meters.
    
    Parameters
    ----------
    lat : array_like or float
        Latitude in radians or degrees.  
    height : array_like or float
        Height in meters.
    unit : {'rad', 'deg'},  default = 'rad'
        Unit of latitude (input) and for multiplication factors (output).
    ellipsoid : str or list of floats, default = 'current'
        Text string with the name of the ellipse or a list ``[a, 1/f]`` with the semi-major 
        axis `a` and inverse flattening `1/f`. The current ellipsoid can be set by
        `setell`. Default for the current ellips is 'WGS-84'.

    Returns
    -------
    flat, flon : float
        Conversion factors for latitude and longitude in `unit`.
                       
    Examples
    --------
    >>> flat, flon = ellcurvature(52, 0, unit='deg')
    >>> print(flat, flon)
    111267.44260909088 68677.7753788433
    
    >>> print(f'In Delft one arcsec is {flat/3600:.3f} m in latitude and {flon/3600 :.3f} m in longitude.')
    In Delft one arcsec is 30.908 m in latitude and 19.077 m in longitude.
    
    """ 

    assert unit in ['deg' ,'rad'], f"Unsupported units {unit}."

    # semi-major axis a, flattening f and eccentricy squared e2  
    
    a, f, GM = inqell(ellipsoid)
    e2 = 2*f - 2*f**2         # eccentricity squared
          
    # the reference position is somewhere in the middle (mean)

    if unit in ['deg', 'deg/m']:
        d2r = np.pi / 180
    else:
        d2r = 1.

    latref = d2r * np.mean(lat)    
    heightref = np.mean(height)

    # curvatures (radius) for the chosen ellipsoid
    
    N_curvature = a / np.sqrt(1-e2*np.sin(latref)**2)       
    M_curvature = a * (1-e2) / (1-e2*np.sin(latref)**2)**(3/2) 
    
    flat = d2r * ( M_curvature + heightref ) 
    flon = d2r * ( N_curvature + heightref ) * np.cos(latref)

    return flat, flon


def ellnormal(ref, mode='plh', unit='rad'):
    """
    Compute unit normal vector (ECEF) perpendicular to the ellipsoid at a given location.
    
    Parameters
    ----------
    ref : array_like with shape (...,3) or (...,2)
        Reference position(s) on the ellipsoid. The type of coordinates are specified by `mode`.
        `ref` can be a vector with more than one coordinate triplet/doublet.  
    mode : {'plh', 'xyz','normalize'}, default = 'plh'
        Coordinate type for `ref`. Possible values are:  
            
        - 'plh' : `ref` contains the geographic latitude and longitude with the unit 
          specified by `unit`. A third coordinate with the height is optional, but not used.
        - 'xyz' : `ref` contains cartesian XYZ coordinates in the ECEF of a point (close) 
          to the ellipsoid. The normal vector is computed for a point on the ellipsoid 
          above or below `ref`.
        - 'normal' : `ref` contains a vector with Cartesian coordinates. The function 
          returns the normalized unit vector.
           
    unit : {'rad', 'deg'},  default = 'rad'
        Units for latitude and longitude, only useful in case the mode='plh' option is used.

    Returns
    -------
    n : ndarray with shape (...,3)
        Unit normal vector(s) in Cartesian XYZ coordinates.
        
    See Also
    --------
    ellrotmatrix, xyz2neu, neu2xyz
                
    Examples
    --------
    >>> n = ellnormal([52*np.pi/180, 4*np.pi/180])
    >>> print(n)
    [0.61416175 0.04294637 0.78801075]
    >>> n = ellnormal([52, 4], unit='deg')
    >>> print(n)
    [0.61416175 0.04294637 0.78801075]

    >>> n = ellnormal([[50, 0], [52, 4]] , unit='deg')
    >>> print(n)
    [[0.64278761 0.         0.76604444]
     [0.61416175 0.04294637 0.78801075]]
    >>> n = ellnormal([[50, 0, 0], [52, 4, 2]] , unit='deg')
    >>> print(n)
    [[0.64278761 0.         0.76604444]
     [0.61416175 0.04294637 0.78801075]]

    >>> xyz = plh2xyz([[50, 0, 0], [52, 4, 2]] , unit='deg')
    >>> n  = ellnormal(xyz, mode='xyz')
    >>> print(n)
    [[0.64278761 0.         0.76604444]
     [0.61416175 0.04294637 0.78801075]]

    The next example does not give the same result as above because the normal 
    direction is different 

    >>> n = ellnormal(xyz, mode='normal')
    >>> print(n)
    [[0.64531918 0.         0.76391306]
     [0.61672217 0.04312542 0.7859987 ]]

    """ 

    assert mode in ['plh' ,'xyz', 'normal'], f"Unsupported mode {mode}."
    
    ref = np.array(ref)
    
    if mode == 'plh':
        assert ref.shape[-1] == 2 or ref.shape[-1] == 3  , "Geographic coordinate vector ref must contain at least latitude and longitude (height is optional)." 
        assert unit in ['rad' ,'deg'], f"Unsupported unit {unit}."
        if unit == 'deg':
            ref = ref[...,[0,1]] * np.pi / 180 
        n = np.stack(( np.cos(ref[...,0]) * np.cos(ref[...,1]) ,
                       np.cos(ref[...,0]) * np.sin(ref[...,1]) ,
                       np.sin(ref[...,0]) ), axis=-1)
    
    elif mode == 'xyz':
        assert ref.shape[-1] == 3 , "Cartesian coordinate vector ref must have three coordinates (XYZ)." 
        ref = xyz2plh(ref)
        n = np.stack(( np.cos(ref[...,0]) * np.cos(ref[...,1]) ,
                       np.cos(ref[...,0]) * np.sin(ref[...,1]) ,
                       np.sin(ref[...,0]) ), axis=-1)
    
    elif mode == 'normal':
        # ref contains normal vector -> normalize to unit vector
        assert ref.shape[-1] == 3 , "Normal vector ref must have three coordinates (XYZ)." 
        n = ref / np.sqrt(np.sum(ref**2, axis=-1, keepdims=True))
    
    return n


def ellrotmatrix(ref, mode='plh', unit='rad'):
    """
    Rotation matrix for ECEF to topocentric coordinates.
    
    Compute rotation matrix/matrices to transform coordinate differences alligned to an 
    ECEF frame into a topocentric reference frame with North, East, Up coordinates. 
    
    Parameters
    ----------
    ref : array_like with shape (...,3) or (...,2)
        Reference position(s) on the ellipsoid. The type of coordinates are specified by `mode`.
        `ref` can be a vector with more than one coordinate triplet/doublet.  
    mode : {'plh', 'xyz','normal'}, default = 'plh'
        Coordinate type for `ref`. Possible values are:
            
         - 'plh' : `ref` contains the geographic latitude and longitude with the unit 
           specified by `unit`. A third coordinate with the height is optional, but not used.
         - 'xyz' : `ref` contains cartesian XYZ coordinates in the ECEF of a point (close) 
           to the ellipsoid. The normal vector is computed for a point on the ellipsoid 
           above or below `ref`.
         - 'normal' : `ref` contains a normal vector with Cartesian coordinates. It does 
           not necessaraly have to be a unit vector.
           
    unit : {'rad', 'deg'},  default = 'rad'
        Units for latitude and longitude, only useful in case the mode='plh' option is used.

    Returns
    -------
    R : ndarray with shape (...,3,3)
        Rotation matrix(es) to transform `dxyz` into `neu`, with ``neu = dxyz @ R``. In case `ref` 
        is a vector, `R` becomes a 3-by-3 matrix , in case `ref` is a multidimensional array, `R` 
        will have one more dimension than `ref` with shape (...,3,3). 
    
    Notes
    -----
    The parameters are passed to `ellnormal` to compute the unit normal vector from which
    the rotation matrix is computed. 

    The rotation is defined as::
    
        neu = dxyz @  R     or    neu = R.T @ dxyz.T   , and    Qneu =  R.T @ Qxyz @ R     
        dxyz =  neu @ R.T   or    dxyz = R @ neu.T     , and    Qxyz =  R @ Qneu @ R.T
        
    For a rotation matrix with more than two dimensions, Einstein multiplication should be used::

        neu = np.einsum('...ji,...j',R,dxyz)   or  neu = np.einsum('...j,...ji',dxyz,R)
        dxyz = np.einsum('...ij,...j',R,neu)   or  dxyz = np.einsum('...ij,...j',R,neu)
    
    Einstein multiplication also works for two dimensions. For coordinate transformations the 
    results are identical to the more efficient and powerful functions `xyz2neu` and `neu2xyz`.
    
    The rotation matrix is used by `covtransform` to convert covariance matrices.
    
    See Also
    --------
    ellnormal, covtransform
                           
    Examples
    --------
    >>> R = ellrotmatrix([52, 4], unit='deg')  
    >>> print(R)
    [[-0.7860912  -0.06975647  0.61416175]
     [-0.05496885  0.99756405  0.04294637]
     [ 0.61566148  0.          0.78801075]]
     
    >>> dxyz=np.array([[1, 1, 1],[2, 1, 3]])
    >>> neu = dxyz @ R
    >>> print(neu)
    [[-0.22539858  0.92780758  1.44511888]
     [ 0.21983318  0.8580511   3.63530214]]
    >>> np.einsum('...ji,...j',R,dxyz)
    array([[-0.22539858,  0.92780758,  1.44511888],
           [ 0.21983318,  0.8580511 ,  3.63530214]])
    >>> np.einsum('...ij,...j',R,neu)
    array([[1., 1., 1.],
           [2., 1., 3.]])

    >>> R = ellrotmatrix([[52, 4],[52, -34]], unit='deg')  
    >>> print(R.shape)
    (2, 3, 3)
    >>> dxyz=np.array([[1, 1, 1],[2, 1, 3]])
    >>> np.einsum('...ji,...j',R,dxyz)
    array([[-0.22539858,  0.92780758,  1.44511888],
           [ 0.9810534 ,  1.94742338,  3.04057172]])
    
    """ 

    # normal vector
        
    n = ellnormal(ref, mode=mode, unit=unit)
        
    cphi = np.sqrt(1-n[...,2]**2)
      
    # Rotation matrix
    
    #    | n |   | -n(k,1)*n(k,3)/cphi  -n(k,2)*n(k,3)/cphi  cphi   |  | x |         | x |
    #    | e | = | -n(k,2)/cphi          n(k,1)/cphi         0      |  | y | = R.T @ | y |
    #    | u |   |  n(k,1)               n(k,2)              n(k,3) |  | z |         | z |
    
    #    [n, e, u] = [x, y, z]  | -n(k,1)*n(k,3)/cphi -n(k,2)/cphi   n(k,1) | 
    #                           | -n(k,2)*n(k,3)/cphi  n(k,1)/cphi   n(k,2) |  
    #                           |  cphi                0             n(k,3) | 
    #                [x, y, z] @ R
        
    R = np.array( [[ -n[...,0]*n[...,2]/cphi ,  -n[...,1]/cphi        ,  n[...,0] ] ,  
                   [ -n[...,1]*n[...,2]/cphi ,   n[...,0]/cphi        ,  n[...,1] ] ,
                   [                    cphi ,   np.zeros_like(cphi)  ,  n[...,2] ] ] )  
    if R.ndim > 2:
        R = np.moveaxis(R, 0, -1)
        R = np.moveaxis(R, 0, -1)
    
    return R

def covtransform(qin, fmtin, fmtout, rotmatrix=None):
    """
    Transform and reformat compact co-variance matrices/vectors.

    Parameters
    ----------
    qin : array_like shape (...,6) or shape (...,3,3) 
        Input (compact) co-variance matrix/vector.
    fmtin : str
        Format of the input compact co-variance matrix/vector, see notes.
    fmtout : str
        Format of the output compact co-variance matrix/vector, see notes.
    rotmatrix : array_like with shape (...,3,3), default=None
        Optional rotation matrix to obtain ``Qout = R.T @ Qin @ R`` .         

    Returns
    -------
    qout : ndarray with shape (...,6) or shape (...,3,3)
        Output (compact) co-variance matrix/vector.

    Notes
    -----
    Supported input and output (compact) "co-variance" matrix/vector formats are::

      qmat    matrix [ [ qx qxy qxz ] 
                       [ qxy qy qyz ] 
                       [ qxy qyz qz ] ]

      qvec    vector [ qx, qy, qz, qxy, qxz, qyz ]
      scor    vector [ sx, sy, sz, rxy, rxz, ryz ]    (geo++, NRCAN)
      scof    vector [ sx, sy, sz, cxy, cxz, cyz ]    (gamit/globk)
      scov    vector [ sx, sy, sz, sxy, sxz, syz ]
 
      qvecd   vector [ qx, qy, qz, qxy, qyz, qxz ] 
      scord   vector [ sx, sy, sz, rxy, ryz, rxz ]    
      scofd   vector [ sx, sy, sz, cxy, cyz, cxz ]    
      scovd   vector [ sx, sy, sz, sxy, syz, sxz ]    (rtklib)
 
    with ``si=sqrt(qi)``, ``rij=qij/(sqrt(qi)*sqrt(qj))``, ``cij=sign(rij)*sqrt(abs(rij))`` 
    and ``sij=sign(qij)*sqrt(abs(qij))``. 
    
    The vector formats ending with "d" are store by diagonal, as is illustrated by::
    
        qmat = [[ 1 4 6 ]   ->  qvecd = [ 1 2 3 4 5 6 ]  ->  qvec = [ 1 2 3 4 6 5] 
                [ 4 2 5 ]                                                     |_|
                [ 6 5 3 ]] 
    
    The other vector formats have their last two positions swapped compared to the 
    store by diagonal formats.

    The input rotation matrix is transposed compared to what is often found in literature and what is
    used by the Matlab versions. This is intentional, as the rotation matrix from `xyz2neu` and `neu2xyz`
    is defined by::
    
        xout = xin @ R
        qout = R.T @ qin @ R 

    These equations can only be used for the most simple use cases with simple shapes and is NOT 
    recommended for coding. Instead, Einstein summation is used in this function::
    
        xout = np.einsum('...j,...ji',xin, R)
        
        tmp = np.einsum('...ij, ...jk -> ...ik', qin, R)
        qout = np.einsum('...ji, ...jk -> ...ik', R, tmp)

    See Also
    --------
    xyz2neu, neu2xyz, printcrd, ellrotmatrix

    Examples
    --------
    Examples reformatting compact co-variance matrices
    
    >>> covtransform([ 0.2061, 0.0708, 0.6352, -0.0109, 0.1514, -0.0188 ], 'scovd', 'qmat')
    array([[ 4.2477210e-02, -1.1881000e-04, -3.5344000e-04],
           [-1.1881000e-04,  5.0126400e-03,  2.2921960e-02],
           [-3.5344000e-04,  2.2921960e-02,  4.0347904e-01]])

    >>> covtransform([[ 1, 2, 3, 0.4, 0.5, -0.6], [ 2, 1, 3, 0.5, 0.4, -0.6]], 'scord','qvec')
    array([[ 1. ,  4. ,  9. ,  0.8, -1.8,  3. ],
           [ 4. ,  1. ,  9. ,  1. , -3.6,  1.2]])

    Example transforming compact xyz co-variance matrix into neu compact covariance matrix,
    using example data from NRCAN PPP processing

    >>> xyz = [ 3923153.4730, 327268.0593, 5001437.4151 ]
    >>> plh = [ 51+58/60+46.39902/3600, 4+46/60+6.78968/3600,  44.3163]
    >>> scorxyz = [ 0.0049, 0.0020, 0.0058, 0.1538,  0.8101,  0.1681 ]
    >>> scorneu = [ 0.0023, 0.0020, 0.0072, 0.0357, -0.0769, -0.0086 ]

    >>> neu, R = xyz2neu(xyz, plh, unit='deg', origin='ecef', rotmatrix='true')
    >>> print(R)
    [[-0.78506419 -0.0831309   0.61381062]
     [-0.06548977  0.99653864  0.05120386]
     [ 0.61594262  0.          0.78779102]]

    >>> scorneu_test = covtransform(scorxyz,fmtin='scor',fmtout='scor',rotmatrix=R)
    >>> print(np.round(scorneu_test,4))
    [ 0.0023  0.002   0.0072  0.0337 -0.0572 -0.0058]
    >>> print(np.round(scorneu_test - scorneu,4))
    [ 0.     -0.      0.     -0.002   0.0197  0.0028]
     
    """ 

    """
    Created:     8 March 2014 by Hans van der Marel for Malab
    Modified:   11 July 2018 by Hans van der Marel
                 - new formats scof and scofd (square root of correlation)
                21 July 2024 by Hans van der Marel
                 - Port to Python
                 
    Based on code originally developed for Matlab(TM) in 2014 and 2018 by the author.

    Copyright Hans van der Marel, Delft University of Technology, 2024
    """
  
    # Check input and output formats
    assert fmtin in ['qmat' ,'qvec', 'scor', 'scof' ,'scov', 'qvecd' ,'scord', 'scofd', 'scovd'], f"Unsupported input format {fmtin}."
    assert fmtout in ['qmat' ,'qvec', 'scor', 'scof' ,'scov', 'qvecd' ,'scord', 'scofd', 'scovd'], f"Unsupported output format {fmtout}."
    
    # Force input array to ndarray and check the shape
    
    qin = np.array(qin)
    if fmtin == 'qmat':
        assert qin.shape[-1] == 3 & qin.shape[-2] == 3 , "Input co-variance matrix must be 3-by-3." 
        # ndim = qin.ndim - 2
        nshape = qin.shape[:-2]
    else:
        assert qin.shape[-1] == 6 , "Input compact co-variance matrix must have six elements." 
        # ndim = qin.ndim - 1
        nshape = qin.shape[:-1]
        

    # Convert input into intermediate qvec format

    qvec = np.zeros( nshape + (6,) )

    if fmtin == 'qmat':
        # convert matrix format to qvec format
        qvec[...,0] = qin[...,0,0]
        qvec[...,1] = qin[...,1,1]
        qvec[...,2] = qin[...,2,2]
        qvec[...,3] = ( qin[...,1,0] + qin[...,0,1] ) / 2
        qvec[...,4] = ( qin[...,2,0] + qin[...,0,2] ) / 2
        qvec[...,5] = ( qin[...,2,1] + qin[...,1,2] ) / 2
        fmttmp = 'qvec'
    else:
        qvec = qin
        fmttmp = fmtin

    if fmttmp in [ 'qvecd', 'scord', 'scofd', 'scovd']: 
        # interchange order of elements
        qvec[...,[4, 5]] = qvec[...,[5, 4]]
        fmttmp = fmttmp[:-1]
        
    if fmttmp == 'scor':
        # sdx sdy sdz rxy rxz ryz
        qvec[...,3] = qvec[...,3] * qvec[...,1] * qvec[...,0] 
        qvec[...,4] = qvec[...,4] * qvec[...,2] * qvec[...,0]
        qvec[...,5] = qvec[...,5] * qvec[...,2] * qvec[...,1]
        qvec[...,0] = qvec[...,0]**2
        qvec[...,1] = qvec[...,1]**2
        qvec[...,2] = qvec[...,2]**2
    elif fmttmp == 'scof':
        # sdx sdy sdz cxy cxz cyz
        qvec[...,3] = np.sign(qvec[...,3]) * qvec[...,3]**2  * qvec[...,1] * qvec[...,0] 
        qvec[...,4] = np.sign(qvec[...,4]) * qvec[...,4]**2 * qvec[...,2] * qvec[...,0]
        qvec[...,5] = np.sign(qvec[...,5]) * qvec[...,5]**2 * qvec[...,2] * qvec[...,1]
        qvec[...,0] = qvec[...,0]**2
        qvec[...,1] = qvec[...,1]**2
        qvec[...,2] = qvec[...,2]**2
    elif fmttmp == 'scov':
        # sdx sdy sdz sdxy sdxz sdyz
        qvec[...,3] = np.sign(qvec[...,3]) * qvec[...,3]**2 
        qvec[...,4] = np.sign(qvec[...,4]) * qvec[...,4]**2 
        qvec[...,5] = np.sign(qvec[...,5]) * qvec[...,5]**2 
        qvec[...,0] = qvec[...,0]**2
        qvec[...,1] = qvec[...,1]**2
        qvec[...,2] = qvec[...,2]**2

    # Convert qvec into matrix if required for output or transformation with rotation matrix

    if fmtout == 'qmat' or rotmatrix is not None:
        if fmtin == 'qmat':
            qout = qin    
        else:
            # convert qvec format into matrix
            qout = np.zeros( nshape + (3,3,) )
            qout[...,0,0] = qvec[...,0] 
            qout[...,1,1] = qvec[...,1] 
            qout[...,2,2] = qvec[...,2]
            qout[...,1,0] = qvec[...,3] 
            qout[...,0,1] = qvec[...,3]
            qout[...,2,0] = qvec[...,4] 
            qout[...,0,2] = qvec[...,4]
            qout[...,1,2] = qvec[...,5] 
            qout[...,2,1] = qvec[...,5]

    # Transform (rotate) the co-variance matrix  Qout = R @ Qin @ R.T

    if rotmatrix is not None:
        # Compute "qout = R.T @ qout @ R" using einstein summation, matrix multiplication
        # with "@" does not work for dimensions > 2. Einstein summation allows to specify
        # broadcasting rules through ellipses.
        #
        # Note: The computation could be made more efficient (1) because the co-variance matrices
        # qin and qout are symmetric and (2) because we can replace the matrix operation
        # by operations directly on qvec. The matrix product "qout = R.T @ qout @ R" can
        # be written as 
        #                  qout = sum_ij ( qin[i,j] * ( R[j,:].T @ R[i,:]) )
        # whereby R[j,:].T @ R[i,:] are outer products producing 3-by-3 matrices for each 
        # non-redundant combination of i,j=0:3 . This is a fun fact only, the performance gain
        # is not noteworthy. 
        
        #qout = np.einsum('...ij, ...kj -> ...ik', qout, rotmatrix)
        #qout = np.einsum('...ij, ...jk -> ...ik', rotmatrix, qout)
        qout = np.einsum('...ij, ...jk -> ...ik', qout, rotmatrix)
        qout = np.einsum('...ji, ...jk -> ...ik', rotmatrix, qout)

        if  fmtout != 'qmat':
            # convert matrix format to qvec format (if required)
            qvec[...,0] = qout[...,0,0]
            qvec[...,1] = qout[...,1,1]
            qvec[...,2] = qout[...,2,2]
            qvec[...,3] = ( qout[...,1,0] + qout[...,0,1] ) / 2
            qvec[...,4] = ( qout[...,2,0] + qout[...,0,2] ) / 2
            qvec[...,5] = ( qout[...,2,1] + qout[...,1,2] ) / 2

    # Return co-variance in matrix format (if requested)
    
    if fmtout == 'qmat':
        return qout
        
    # Convert intermediate qvec format into output vector format 
  
    if fmtout in ['scor', 'scord']:
        # sdx sdy sdz rxy rxz ryz
        qvec[...,0] = np.sqrt(qvec[...,0])
        qvec[...,1] = np.sqrt(qvec[...,1])
        qvec[...,2] = np.sqrt(qvec[...,2])
        qvec[...,3] = qvec[...,3] / ( qvec[...,1] * qvec[...,0] ) 
        qvec[...,4] = qvec[...,4] / ( qvec[...,2] * qvec[...,0] )
        qvec[...,5] = qvec[...,5] / ( qvec[...,2] * qvec[...,1] )
    elif fmtout in ['scof', 'scofd']:
        # sdx sdy sdz cxy cxz cyz
        qvec[...,0] = np.sqrt(qvec[...,0])
        qvec[...,1] = np.sqrt(qvec[...,1])
        qvec[...,2] = np.sqrt(qvec[...,2])
        qvec[...,3] = np.sign(qvec[...,3]) * np.sqrt(np.abs(qvec[...,3]) / ( qvec[...,1] * qvec[...,0] ) )  
        qvec[...,4] = np.sign(qvec[...,4]) * np.sqrt(np.abs(qvec[...,4]) / ( qvec[...,2] * qvec[...,0] ) )
        qvec[...,5] = np.sign(qvec[...,5]) * np.sqrt(np.abs(qvec[...,5]) / ( qvec[...,2] * qvec[...,1] ) )
    elif fmtout in ['scov', 'scovd']:
        # sdx sdy sdz sdxy sdxz sdyz
        qvec[...,0] = np.sqrt(qvec[...,0])
        qvec[...,1] = np.sqrt(qvec[...,1])
        qvec[...,2] = np.sqrt(qvec[...,2])
        qvec[...,3] = np.sign(qvec[...,3]) * np.sqrt(np.abs(qvec[...,3])) 
        qvec[...,4] = np.sign(qvec[...,4]) * np.sqrt(np.abs(qvec[...,4]))
        qvec[...,5] = np.sign(qvec[...,5]) * np.sqrt(np.abs(qvec[...,5]))

    if fmtout in [ 'qvecd', 'scord', 'scofd', 'scovd']: 
        # interchange order of elements
        qvec[...,[4, 5]] = qvec[...,[5, 4]]

    return qvec
 
# ----------------------------------------------------------------------------
#                          PRINT FUNCTIONS
# ----------------------------------------------------------------------------
#
#   printcrd     - Print a table with coordinates and optional co-variances
#   printxyz     - Print array with cartesian coordinates
#   printplh     - Print array with geodetic/ellipsoidal coordinates 
#   deg2dms      - Convert latitude/longitude to degree, minute, second notation

def printcrd(crd, prtfmt="xyz", labels=[], sdcov=[], sdcovfmt="std", unit='rad/m', title=""):
    """
    Print a table with coordinates and optional standard deviation/co-variance information.

    Parameters
    ----------
    crd : array_like with shape (...,3) 
        Array with Cartesian or geographic coordinates.
    prtfmt : {'xyz', 'neu','map', 'plh', 'dms','dmspretty', ...}, default='xyz'
        Coordinate format for printing.
        
        - 'xyz' ('ecef'):  Cartesian XYZ coordinates (default)
        - 'neu' ('local'): Coordinates in local NEU system
        - 'map': Coordinates in map projection system
        - 'plh' ('geodetic'): Geographic coordinates printed in decimal degrees
        - 'dms' ('sexagesimal'): Geographic coordiantes printed in degrees, minutes, seconds. 
        - 'dmspretty': As 'dms', with degree, minute and second symbols.
        
        For the options 'dms', 'dmspretty' and 'plh' `crd` must contain the latitude and 
        longitude in either radians or degrees (with unit='deg/m')
    labels : array_like, optional
        Optional list with row labels, e.g. station names.        
    sdcov : array_like with shape (...,3) or shape (...,6) 
        Optional standard deviations (...,3) and/or co-variances (...,6). 
    sdcovfmt : {'std', 'qvec', 'scor', 'scov', ...}, optional
        Format of the input compact co-variance matrix/vector, see `covtransform` for
        all for possible formats. Default is 'std'
    unit : {'rad/m', 'deg/m', 'deg', 'rad'}, optional
        Units of input `crd` array, , default 'rad/m'.
    title : str, optional
        Optional title string.
        
    Examples
    --------
    Create a test case
    
    >>> crd = np.array([[ 1, 2, 3 ] , [ 4, 5, 6]])
    >>> sdcov = np.hstack((crd,crd))
    >>> station = [ 'DELF' , 'Stolwijk']
    
    Basic examples without much additional information
    
    >>> printcrd(crd, prtfmt='map')
              x[m]          y[m]        h[m]
    <BLANKLINE>
            1.0000        2.0000      3.0000
            4.0000        5.0000      6.0000
            
    >>> printcrd(crd, prtfmt='dmspretty', unit='deg')
             Lat[dms]        Lon[dms]       H[m]
    <BLANKLINE>
        1°00'00.0000"   2°00'00.0000"     3.0000
        4°00'00.0000"   5°00'00.0000"     6.0000

    More elaborate examples with labels, standard deviations and covariance information
        
    >>> printcrd(crd, prtfmt='xyz', labels=station, sdcov=sdcov, sdcovfmt="std")  
                        X[m]            Y[m]            Z[m]      sx[m]    sy[m]    sz[m]
    <BLANKLINE>
    DELF              1.0000          2.0000          3.0000     1.0000   2.0000   3.0000
    Stolwijk          4.0000          5.0000          6.0000     4.0000   5.0000   6.0000
    
    >>> printcrd(crd, prtfmt='plh', labels=station, sdcov=sdcov, sdcovfmt="qvec", unit='deg')
                    Lat[deg]        Lon[deg]       H[m]      sn[m]    se[m]    su[m]   sne[m]   snu[m]   seu[m]
    <BLANKLINE>
    DELF         1.000000000     2.000000000     3.0000     1.0000   1.0000   1.0000   1.0000   1.0000   1.0000
    Stolwijk     4.000000000     5.000000000     6.0000     2.0000   2.0000   2.0000   2.0000   2.0000   2.0000
     
    """ 

    # Force input array to ndarray and check the first twp parameters 
    crd = np.array(crd)
    assert crd.shape[-1] == 3 , "Coordinate vector must have three coordinates." 
    assert prtfmt in ['ecef', 'xyz', 'local', 'neu', 'map', 'geodetic', 'plh', 'sexagesimal', 'dms', 'dmspretty'], f"Unsupported print format {prtfmt}."
    assert unit in ['rad/m', 'deg/m', 'deg', 'rad'], f"Unsupported unit {unit}."

    # Check optional labels
    labels=np.array(labels)
    if labels.size  != 0:
        assert crd.shape[:-1] == labels.shape , "Station list size does not match coordinate list." 

    # Check standard deviation / covariances
    sdcov = np.array(sdcov)
    if sdcov.size != 0:
        assert sdcov.shape[-1] == 3 or sdcov.shape[-1] == 6, "sdcov must have 3 or 6 elements in each item." 
        assert sdcov.shape[:-1] == crd.shape[:-1] , "Sdcov size does not match crd size." 
        # covfmt will be asserted later

    # Prepare header and format strings

    if prtfmt in ['xyz', 'ecef']:
        crdheader = '            X[m]            Y[m]            Z[m]'
        covheader = '      sx[m]    sy[m]    sz[m]   sxy[m]   sxz[m]   syz[m]'
        crdstr = '{:>16.4f}{:>16.4f}{:>16.4f}'
    elif prtfmt in ['neu', 'local']:
        crdheader = '        N[m]        E[m]        U[m]'
        covheader = '      sn[m]    se[m]    su[m]   sne[m]   snu[m]   seu[m]'
        crdstr = '{:>12.4f}{:>12.4f}{:>12.4f}'
    elif prtfmt in ['map']:
        crdheader = '          x[m]          y[m]        h[m]'
        covheader = '      sx[m]    sy[m]    sh[m]   sxy[m]   sxh[m]   syh[m]'
        crdstr = '{:>14.4f}{:>14.4f}{:>12.4f}' 
    elif prtfmt in ['plh', 'geodetic']:
        crdheader = '        Lat[deg]        Lon[deg]       H[m]'
        covheader = '      sn[m]    se[m]    su[m]   sne[m]   snu[m]   seu[m]'
        crdstr = '{:>16.9f}{:>16.9f}{:>11.4f}'
    elif prtfmt in ['dms', 'sexagesimal', 'dmspretty']:
        crdheader = '         Lat[dms]        Lon[dms]       H[m]'
        covheader = '      sn[m]    se[m]    su[m]   sne[m]   snu[m]   seu[m]'
        crdstr = ' {:>16s}{:>16s}{:>11.4f}'
        if prtfmt == 'dmspretty':
            pretty=True
        else:
            pretty=False

    # Prepare co-variance header and format strings

    if sdcov.size != 0:
        if sdcovfmt in ['std']:
           covheader = covheader[0:29]
           covstr = '  {:>9.4f}{:>9.4f}{:>9.4f}'
        else:
           sdcov = covtransform(sdcov, sdcovfmt, 'scov')
           covstr = '  {:>9.4f}{:>9.4f}{:>9.4f}{:>9.4f}{:>9.4f}{:>9.4f}'
    else:
       covheader = ''
       covstr = ''

    # Prepare label header and lenght

    if labels.size != 0:
        labellength = np.max(np.char.str_len(labels))
        labelstr = '{:' + '{:d}'.format(labellength) + 's}'
        labelheader = labelstr.format(' ')
    else:
        labelheader = ''

    # conversion factor for lat/lon

    if unit == 'rad/m':
        r2d=180./np.pi
    else:
        r2d=1.
    
    # Print headers

    if title != '':
        print(title)
        print()
    #print(labelheader + crdheader + covheader)
    print(labelheader + crdheader + covheader)
    print()

    # Print body

    n = crd.shape[-2]
    for k in range(n):
        line = ''
        if labelheader != "":
            line = line + labelstr.format(labels[k])
        if prtfmt in ['dms', 'sexagesimal', 'dmspretty']:
            line = line + crdstr.format(deg2dms(crd[k,0]*r2d, unit='deg', pretty=pretty),deg2dms(crd[k,1]*r2d, unit='deg',pretty=pretty),crd[k,2])    
        elif prtfmt in ['plh', 'geodetic']:
            line = line + crdstr.format(crd[k,0]*r2d,crd[k,1]*r2d,crd[k,2])    
        else:
            line = line + crdstr.format(crd[k,0],crd[k,1],crd[k,2])   
        if sdcov.size != 0:
            if sdcovfmt in ['std']:
                line = line + covstr.format(sdcov[k,0],sdcov[k,1],sdcov[k,2])
            else:
                line = line + covstr.format(sdcov[k,0],sdcov[k,1],sdcov[k,2],sdcov[k,3],sdcov[k,4],sdcov[k,5])
        print(line)

    return

def printxyz(xyz):
    """
    Print Cartesian coordinates in fixed format with 0.1 mm precision.
    
    Parameters
    ----------
    xyz : array_like with shape (...,3) 
        Cartesian coordinates.

    Examples
    --------
    >>> printxyz([4123456.23000, 0.3, 0.123451])
    [ 4123456.2300  0.3000  0.1235]
    
    """
    
    with np.printoptions(formatter={'float': '{: 0.4f}'.format}, suppress=True):
        print(np.round(xyz,4))

    return
    
def printplh(plh, unit='rad/m'):
    """
    Print Geographic (Lat, Lon, Height) coordinates in fixed format with 0.1 mm precision.

    Parameters
    ----------
    plh : array_like with shape (...,3) 
        Ellipsoidal coordinates (geographic latitude, longitude and height above the ellipsoid).
    unit : {'rad/m', 'deg/m', 'rad', 'deg'},  default = 'rad/m'
        Units for the input latitude and longitude (and height).

    Notes
    -----
    One degree is about 111 km, one millidegree (10^-3) is about 111 m, one micro degree 
    (10^-6) is 0.111 m,  one pico degree (10^-9)  is 0.111 mm, so 9 digits will do
    for the latitude and longitude, and 4 digits for the height.

    Examples
    --------
    >>> printplh([52.1234567891, 4., 110.000001], unit='deg')
    [ 52.12345679   4.         110.        ]
    
    """
    
    tmp=np.array(plh)
    assert tmp.shape[-1] == 3 , "Ellipsoidal coordinate vector(s) plh must have three coordinates (Latitude, longitude, height)." 
    assert unit in ['rad/m' ,'deg/m', 'rad' ,'deg'], f"Unsupported unit {unit}."
    if unit in ['rad', 'rad/m']:
        tmp[...,0:2] = tmp[...,0:2]*180/np.pi

    tmp[...,0:2] = np.round(tmp[...,0:2],9)
    tmp[...,2]= np.round(tmp[...,2],4)
    
    with np.printoptions(suppress=True):
        print(tmp)

    return

def deg2dms(value, pretty=False, unit='rad'):
    """
    Convert latitude/longitude angles into a string with degrees, minutes and second format.

    Parameters
    ----------
    value : float
        Value to convert.
    unit : {'rad', 'deg', 'rad/m', 'deg/m'},  optional, default = 'rad'
        Unit of the values (only 'rad' and 'deg' are meaningful).
    pretty : boolean, optional, default=False
        Pretty print with degree, minute and second symbol.
        
    Notes
    -----
    One degree is about 111 km along the meridian, one minute is 1850 m (one nautical mile), one 
    second is 31 m, one milliarcsecond (mas) is 31 mm. The print resolution for the latitude, or
    longitude at the equator, is 0.01 mas, or 0.31 mm, in normal mode, and  0.1 mas, or 3.1 mm, 
    in pretty print mode. 
    
    Examples
    --------
    >>> deg2dms(0.9)
    ' 51 34 58.32562'

    >>> print(deg2dms(23.234567, unit='deg', pretty=True))
     23°14'04.4412"
    
    """

    if unit in ['rad', 'rad/m']:
        value1 = np.abs(value) * 180 / np.pi
    else:
        value1 = np.abs(value)

    value3 = ( value1 * 3600 )  % 60
    value2 = ( value1 * 60) % 60
    value1 = np.sign(value) * np.round(value1-value2/60-value3/3600)  

    if pretty:
        dms='{:>3.0f}\xb0{:02.0f}\x27{:>07.4f}"'.format(value1, value2, value3)
        #dms='{:>3.0f}\xb0{:02.0f}\x60{:>07.4f}"'.format(value1, value2, value3)
    else:
        dms='{:>3.0f} {:02.0f} {:>07.5f}'.format(value1, value2, value3)

    return dms
    