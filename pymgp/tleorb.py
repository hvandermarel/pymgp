"""
Two-Line Element (TLE) orbits.

Compute satellite position and velocity using two-line elements available from
Celestrak (https://celestrak.org).

Functions
---------
Get and read NORAD Two Line Elements in TLE structure array

- tleget        Retrieve NORAD Two Line Elements from www.celestrak.com
- tleread       Read NORAD Two Line Elements from file.
- tleprint      print NORAD Two Line Elements.

Find satellites and select dates 

- tlefind       Find named satellites in the NORAD Two Line Elements.
- tledatenum    Compute Matlab datenumbers from a date range.

Compute satellite positions and orbit propagation

- tle2orb     - Compute orbital elements from NORAD Two Line Elements
- tle2vec     - Satellite position and velocity from NORAD Two Line Elements.

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
satorb:
    Satellite orbit module (used by this module)
tleplot:
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
"""

__author__ = "Hans van der Marel"
__copyright__ = "Copyright 2012-2024, Hans van der Marel, Delft University of Technology."
__credits__ = ["Hans van der Marel", "Simon van Diepen"]
__license__ = "License Name and Info"
__version__ = "0.9.0"
__maintainer__ = "Hans van der Marel"
__email__ = "h.vandermarel@tudelft.nl"
__status__ = "development"

"""
Created:    30 Dec 2012 by Hans van der Marel for Matlab
Modified:   13 Sep 2017 by Hans van der Marel for Matlab
             - added tle2orb with improved orbit propagation
            12 Nov 2020 by Hans van der Marel and Simon van Diepen
             - initial port to Python (part of `tleplot` module)
            12 Jul 2024 by Hans van der Marel
             - refactored `tleplot` module into `tlefunc` and `tleplot`
             - this module `tlefunc` contains all functions not dealing with
               plotting, needed for the ectActivationTimes.py module
             2 Aug 2024 by Hans van der Marel
             - changes name from `tlefunc` to `tleorb`
             - imports from new module `satorb` instead of `crsutil`
             - updated docstrings to numpy style
             - fixed multiple set reading bug in tleget
             - support for higher dimensional arrays and multiple satellites
               (as is the case in the original matlab code) in tle2vec
             - return time and satellite names from tle2vec, initiates upstream
               adjustments in the calling sequence xsat, vsat, *_= tle2vec(...)
             - many new examples and documentation improvements
             5 Aug 2024 by Hans van der Marel
             - New function tleprint added
            22 Aug 2024 by Hans van der Marel
             - Major edits to the docstrings to facilitate sphynx
                                            
Based on code originally developed for Matlab(TM) in 2010-2016 by the author.

Copyright Hans van der Marel, Delft University of Technology, 2012-2024
"""

# Import modules

import os
import numpy as np

from collections import namedtuple
import urllib.request as url
from datetime import datetime
from dateutil.parser import parse as parsedate
import ssl
from satorb import keplerm, orb2vec, datetime2num

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def tleget(tleset, tlefile=""):
    """Retrieve NORAD Two Line Elements from www.celestrak.com.

    Download current NORAD two line elements from Celestrak (www.celestrak.com) 
    and saves the two line elements to a file on the local computer.

    Parameters
    ----------
    tleset : array_like, list of strings, string
        String, list of strings, or ndarray with the name(s) of the two line 
        element set(s). This can be the name of a file on Celestrak, or, a 
        family of satellites. Common sets of two line elements are 'GPS' 
        ('gps-ops'), 'GLONASS' ('glo-ops'), 'GALILEO', 'BEIDOU', 'SBAS'; 'GNSS' 
        or 'SATNAV' to do all satellite navigation systems; 'resource' for Earth 
        resource satellites, etc. For a full list see the Celestrak website.
    tlefile : string, optional
        Name of local file to save the two-line elements to. Default is the
        filename on Celestrak or the set name.

    Returns
    -------
    tlefile : string 
        Name of the local file with two-line elements.

    Notes
    -----    
    The mean orbital elements contained in TLE are Earth-centered-inertial (ECI)
    coordinates with respect to the true equator of date and the mean equinox
    of date. They do not include the effect of nutation. The TLE are 
    compatible with the SGP4 and SDP4 orbit propagators of NORAD. The 
    two line elements are read with the `tleread()` function.

    See Also
    --------
    tleread, tle2vec
    
    Examples
    --------
    >>> tlefile = tleget('resource')
    Saving TLE set to default name
    TLEGET: Downloaded resource.txt from http://celestrak.com/NORAD/elements/
    Saved TLE to resource.txt
    
    >>> tlefile = tleget('gnss')
    Saving TLE set to default name
    TLEGET: Downloaded gps-ops.txt from http://celestrak.com/NORAD/elements/
    TLEGET: Downloaded glo-ops.txt from http://celestrak.com/NORAD/elements/
    TLEGET: Downloaded galileo.txt from http://celestrak.com/NORAD/elements/
    TLEGET: Downloaded beidou.txt from http://celestrak.com/NORAD/elements/
    TLEGET: Downloaded sbas.txt from http://celestrak.com/NORAD/elements/
    Saved TLE to gnss.txt
    
    >>> tlefile = tleget(['gps', 'galileo'],'gps-galileo-20240802.tle')
    TLEGET: Downloaded gps-ops.txt from http://celestrak.com/NORAD/elements/
    TLEGET: Downloaded galileo.txt from http://celestrak.com/NORAD/elements/
    Saved TLE to gps-galileo-20240802.tle
    
    """
    
    if type(tleset) not in [np.ndarray, list, str]:
        raise ValueError("tleset is not a list, string or array!")

    celestrakurl = 'http://celestrak.com/NORAD/elements/'
    if tlefile == "":
        print("Saving TLE set to default name")
        if type(tleset) == str:
            tlefile = tleset + ".txt"
        else:
            raise ValueError("tlefile must be defined when using multiple tleset!")

    if type(tleset) == str:
        tleset = [tleset]
    elif type(tleset) == np.ndarray:
        tleset = list(tleset)

    tleset2 = []
    for k in range(len(tleset)):
        tlesetk = tleset[k].lower()
        if tlesetk in ['gnss', 'satnav']:
            tleset2.append('gps-ops')
            tleset2.append('glo-ops')
            tleset2.append('galileo')
            tleset2.append('beidou')
            tleset2.append('sbas')
        elif tlesetk == 'gps':
            tleset2.append('gps-ops')
        elif tlesetk in ['glonass', 'glo']:
            tleset2.append('glo-ops')
        else:
            tleset2.append(tlesetk)

    s = ''
    fname = []
    for tlelink in tleset2:
        fullurl = celestrakurl + tlelink + '.txt'
        try:
            sk = str(url.urlopen(fullurl, context=ctx).read())
        except IOError:
            print("TLEGET: Could not retrieve {}.txt from {}".format(tlelink, celestrakurl))
            continue
        s += sk[2:-1]
        fname.append(tlelink)
        print("TLEGET: Downloaded {}.txt from {}".format(tlelink, celestrakurl))

    f = open(tlefile, "w")
    if "\\r" in s:
        s = s.replace("\\r", "")
    if "\\n" in s:
        s = s.replace("\\n", "\n")
		
    f.write(str(s))
    f.close()

    print("Saved TLE to {}".format(tlefile))

    return tlefile

def tleread(tlefile, verbose=1):
    """Read NORAD Two Line Elements from file.

    Reads NORAD two line elements from file and return the two-line elements 
    (TLE's) as a list of named tuples.

    Parameters
    ----------
    tlefile : string
        Name of the file with two-line elements.
    verbose : int, optional
        Verbosity level. If 'verbose=1' (default) an overview of the TLE's is 
        printed, if 'verbose=0' the function is quiet.

    Returns
    -------
    tle : list of named tuples
        List with two line elements.

    Notes
    -----
    The mean orbital elements contained in tle are Earth-centered-inertial (ECI)
    coordinates with respect to the true equator of date and the mean equinox
    of date. They do not include the effect of nutation. The tle are 
    compatible with the SGP4 and SDP4 orbit propagators of NORAD.

    Files with TLE's can be obtained from www.celestrak.com. You may use
    the function `tleget` to do this. The TLE files can have an optional line 
    with a twenty-four character name before the traditional Two Line Element 
    format (Three-Line Elemement Set).
    
    See Also
    --------
    tleget, tle2vec, tlefind, tledatenum, tle2orb
    
    Examples
    --------
    >>> tle = tleread('gps-20240802.tle')
    <BLANKLINE>
    Satellite              Reference_Epoch    a [km]    ecc [-]  inc [deg] RAAN [deg] argp [deg]    E [deg]    Period
    <BLANKLINE>
    GPS BIIR-2  (PRN 13)    2024-214.14951  26560.17  0.0082086    55.6823   125.3746    53.0305   307.7241  11:57:58
    GPS BIIR-4  (PRN 20)    2024-214.54131  26559.29  0.0038299    54.7037    47.0333   215.0749   153.9228  11:57:56
    ...
    GPS BIII-5  (PRN 11)    2024-214.96247  26561.08  0.0016693    55.3656   359.9043   218.6075   141.7489  11:58:00
    GPS BIII-6  (PRN 28)    2024-214.66082  26560.55  0.0000873    55.0874   175.3851    64.9315   121.2604  11:57:59

    >>> tle = tleread('gps-20240802.tle', verbose=0)
    >>> print('type(tle)',type(tle), '   length(tle)',len(tle), '   type(tle[0])',type(tle[0]))
    type(tle) <class 'list'>    length(tle) 31    type(tle[0]) <class 'tleorb.TLE'>
    
    """
    
    # Constants (WGS-84)
    mu = 398600.5  # km^3/s^2
    #Re = 6378.137  # km (WGS84 earth radius)
    d2r = np.pi / 180
    
    
    # Define internal function to compute eccentric anomaly
    def eanomaly(m0, ecc0, T0L=1e-10):
        E = m0
        f = [1, 1]
        fdot = 1
        while abs(np.array([f])).max() > T0L:

            f = m0 - E + ecc0 * np.sin(E)
            fdot = -1 + ecc0 * np.cos(E)
            E -= f / fdot

        return E


    # Define named tuple TLE (similar to Matlab stuct)
    tlestruct = namedtuple("TLE",
                           "name satid ephtype year epoch t0 ecc0 inc0 raan0 argp0 m0 n0 ndot nddot bstar revnum a0 e0")

    # Open the file with TLEs'
    if not os.path.exists(tlefile):
        raise ValueError("Filename {} with TLE elements not found!".format(tlefile))

    f = open(tlefile, "r")
    lines = f.read().split("\n")
    f.close()
    
    if verbose:
        print('\nSatellite              Reference_Epoch    a [km]    ecc [-]  inc [deg] RAAN [deg] argp [deg]    '
              'E [deg]    Period\n')

    # Read TLE elements

    """
    Data for each satellite consists of three lines in the following format:

             1         2         3         4         5         6         7
    1234567890123456789012345678901234567890123456789012345678901234567890

    AAAAAAAAAAAAAAAAAAAAAAAA
    1 NNNNNU NNNNNAAA NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNN
    2 NNNNN NNN.NNNN NNN.NNNN NNNNNNN NNN.NNNN NNN.NNNN NN.NNNNNNNNNNNNNN

    Line 0 is a twenty-four character name (to be consistent with the name 
    length in the NORAD SATCAT). Line 0 is optional and may be preceeded with
    a zero.  

    Lines 1 and 2 are the standard Two-Line Orbital Element Set Format 
    identical to that used by NORAD and NASA. The format description is:

    Line 1

    Column 	Description
    01      Line Number of Element Data
    03-07 	Satellite Number
    08      Classification (U=Unclassified)
    10-11 	International Designator (Last two digits of launch year)
    12-14 	International Designator (Launch number of the year)
    15-17 	International Designator (Piece of the launch)
    19-20 	Epoch Year (Last two digits of year)
    21-32 	Epoch (Day of the year and fractional portion of the day)
    34-43 	First Time Derivative of the Mean Motion
    45-52 	Second Time Derivative of Mean Motion (decimal point assumed)
    54-61 	BSTAR drag term (decimal point assumed)
    63      Ephemeris type
    65-68 	Element number
    69      Checksum (Modulo 10)

    Line 2

    Column 	Description
    01      Line Number of Element Data
    03-07   Satellite Number
    09-16 	Inclination [Degrees]
    18-25 	Right Ascension of the Ascending Node [Degrees]
    27-33 	Eccentricity (decimal point assumed)
    35-42 	Argument of Perigee [Degrees]
    44-51 	Mean Anomaly [Degrees]
    53-63 	Mean Motion [Revs per day]
    64-68 	Revolution number at epoch [Revs]
    69      Checksum (Modulo 10)
 
    All other columns are blank or fixed.

    The mean orbital elements in TLE are Earth-centered-inertial (ECI)
    coordinates with respect to the true equator of date and the mean equinox 
    of date. They do not include the effect of nutation.
    """
   
    tlelist = []
    ntle = 0
    line0 = ""
    line1 = ""
    line2 = ""
    for line in lines:
        # Skip blank lines
        if line == "":
            continue
        # Decode the line with the satellite name (optional) and read next two lines
        if line[0:2] == "1 ":     # Line 1
            if line1 != "":
                raise ValueError("Invalid line format, received 2 line1: {}, {}.".format(line1, line))
            line1 = line
        elif line[0:2] == "2 ":   # Line 2
            if line1 == "":
                raise ValueError("Received line2 when line1 not yet read! {}".format(line))
            if line2 != "":
                raise ValueError("Invalid line format, received 2 line2: {}, {}.".format(line2, line))
            line2 = line
        else:
            if line0 != "":
                raise ValueError("Invalid line format, received 2 line0: {}, {}.".format(line0, line))
            line0 = line

        if line2 != "":
            # Now we have the second line we can start the decoding
            
            # Decode the zero line (if available)
            if line0 != "":
                if line0[:2] == "0 ":
                    satname = line0[2:]
                else:
                    satname = line0
            else:
                satname = "Undefined"

            # Decode the first line with TLE data

            satid = eval(line1[2:7].lstrip("0"))
            classification = line1[7]
            intldesg = line1[9:17]
            epochyr = eval(line1[18:20].lstrip("0"))
            if epochyr < 57:
                epochyr += 2000
            else:
                epochyr += 1900

            epochdays = eval(line1[20:32].lstrip("0"))
            ndot = eval(line1[33:43].lstrip("0"))
            nddot = eval(line1[44:50].lstrip("0"))
            nexp = eval(line1[50:52].lstrip("0"))
            nddot *= 1e-5 * 10**nexp
            bstar = eval(line1[53:59].lstrip("0"))
            ibexp = eval(line1[59:61].lstrip("0"))
            bstar *= 1e-5 * 10**ibexp
            ephtype = line1[62]
            elnum = eval(line1[64:68].lstrip("0"))

            # Decode the second line with TLE data

            if eval(line2[2:7].lstrip("0")) != satid:
                raise ValueError("satid from line 1 ({}) does not match satid from line 2 ({})!".format(satid,
                                                                                                        eval(line2[2:7])
                                                                                                        ))
            inc0 = eval(line2[7:16].lstrip("0"))
            raan0 = eval(line2[16:25].lstrip("0"))
            ecc0 = eval(line2[26:33].lstrip("0")) * 1e-7
            argp0 = eval(line2[33:42].lstrip("0"))
            m0 = eval(line2[42:51].lstrip("0"))
            n0 = eval(line2[51:63].lstrip("0"))
            revnum = eval(line2[63:68].lstrip("0"))

            # Complete orbital elements
            t0 = datetime2num(datetime(year=int(epochyr), month=1, day=1)) + epochdays - 1
            a0 = (mu/(n0*2*np.pi/(24*3600))**2)**(1/3)
            e0 = eanomaly(m0, ecc0)
            OE = [a0, ecc0, inc0, raan0, argp0, e0]

            # Print some data from the TLE's
            if verbose:
                ihour = np.floor(24/n0)
                imin = np.floor(24*60/n0 - ihour*60)
                isec = round(24*3600/n0-ihour*3600-imin*60)
                if isec == 60:
                    isec = 0
                    imin += 1
                    if imin == 60:
                        imin = 0
                        ihour += 1
                tt = '{:0>2.0f}:{:0>2.0f}:{:0>2.0f}'.format(ihour, imin, isec)
                print("{:<24s}{:4d}-{:>3.5f}".format(satname, epochyr, epochdays) +
                      "  {:>8.2f}  {:>9.7f}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {}".format(*OE, tt))

            # Fill output structure           
            ntle += 1
            tlelist.append(tlestruct(satname.strip(" "), (satid, classification, intldesg.strip(" ")), (ephtype, elnum),
                                     epochyr, epochdays, t0, ecc0, inc0 * d2r, raan0 * d2r, argp0 * d2r, m0*d2r,
                                     n0 * 2 * np.pi, ndot * 2 * np.pi, nddot * 2 * np.pi, bstar, revnum, a0 * 1000,
                                     e0 * d2r))
            # Clear lines
            line0 = ""
            line1 = ""
            line2 = ""

    return tlelist

def tleprint(tle):
    """Print NORAD Two Line Elements.

    Parameters
    ----------
    tle : list of named tuples
        List with two line elements.
    
    See Also
    --------
    tleget, tleread
    
    Examples
    --------
    >>> tleprint('gps-20240802.tle')
    <BLANKLINE>
    Satellite              Reference_Epoch    a [km]    ecc [-]  inc [deg] RAAN [deg] argp [deg]    E [deg]    Period
    <BLANKLINE>
    GPS BIIR-2  (PRN 13)    2024-214.14951  26560.17  0.0082086    55.6823   125.3746    53.0305   307.7241  11:57:58
    GPS BIIR-4  (PRN 20)    2024-214.54131  26559.29  0.0038299    54.7037    47.0333   215.0749   153.9228  11:57:56
    ...
    GPS BIII-5  (PRN 11)    2024-214.96247  26561.08  0.0016693    55.3656   359.9043   218.6075   141.7489  11:58:00
    GPS BIII-6  (PRN 28)    2024-214.66082  26560.55  0.0000873    55.0874   175.3851    64.9315   121.2604  11:57:59
    
    """
        
    r2d = 180 / np.pi 

    print('\nSatellite              Reference_Epoch    a [km]    ecc [-]  inc [deg] RAAN [deg] argp [deg]    '
          'E [deg]    Period\n')

    for tle_ in tle:
    
        # Compute oribital period 
        n0 = tle_.n0 / (2*np.pi) 
        ihour = np.floor(24/n0)
        imin = np.floor(24*60/n0 - ihour*60)
        isec = round(24*3600/n0-ihour*3600-imin*60)
        if isec == 60:
            isec = 0
            imin += 1
            if imin == 60:
                imin = 0
                ihour += 1
        tt = '{:0>2.0f}:{:0>2.0f}:{:0>2.0f}'.format(ihour, imin, isec)
           
        # Print line           
        print("{:<24s}{:4d}-{:>3.5f}".format(tle_.name, tle_.year, tle_.epoch) +
            "  {:>8.2f}  {:>9.7f}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {}".format(
            tle_.a0 / 1000, tle_.ecc0, tle_.inc0*r2d, tle_.raan0*r2d, tle_.argp0*r2d, tle_.e0*r2d, tt) )
        
    return

def tlefind(tle, satid, verbose=1):
    """Find named satellites in the NORAD Two Line Elements.

    Search for named satellites in the list of TLE's and return an index
    array and list of names for the found satellites.
    
    Parameters
    ----------
    tle : list of named tuples
        List of named tuples with two-line elements.
    satid : str, int, array_like of int or str
        String or array with the satellite names to look for, an index, or index
        array. For strings, the functions looks for satellites that start with
        `satid`.
    verbose : int, optional
        Verbosity level. If 'verbose=1' (default) print a message for the found
        TLE(s), if 'verbose=0' the function is quiet.
    
    Returns
    -------
    isat : ndarray, int64
        Array with element numbers in the list of named tuples `tle`
    satids : ndarray, str
        Array with satellite names that have been found
    
    Notes
    -----
    The input parameter `satid` accepts the returns `isat` and `satids` of a
    previous function call and should return the same result (on the same set
    of TLE)
    
    See Also
    --------
    tleget, tleread, tle2vec, tledatenum

    Examples
    --------
    >>> tleERS=tleread('resource-20240802.tle', verbose=0)    # read two-line elements
    >>> isat, *_ = tlefind(tleERS, 'COSMO-SKYMED 1')
    Found 1 satellites:
     COSMO-SKYMED 1  (18)
     
    >>> tlefind(tleERS, 'COSMO', verbose=0)
    (array([18, 22, 29, 42], dtype=int64),
     array(['COSMO-SKYMED 1', 'COSMO-SKYMED 2', 'COSMO-SKYMED 3',
            'COSMO-SKYMED 4'], dtype='<U23'))
    
    >>> isat, satids = tlefind(tleERS, [ 'RADARSAT-2', 'SENTINEL-1', 'TERRASAR' ])
    Found 3 satellites:
     TERRASAR-X  (19)
     RADARSAT-2  (23)
     SENTINEL-1A  (65)
    >>> tlefind(tleERS, isat, verbose=0)
    (array([19, 23, 65], dtype=int64),
     array(['TERRASAR-X', 'RADARSAT-2', 'SENTINEL-1A'], dtype='<U23'))
    >>> tlefind(tleERS, isatids, verbose=0)
    (array([19, 23, 65], dtype=int64),
     array(['TERRASAR-X', 'RADARSAT-2', 'SENTINEL-1A'], dtype='<U23'))

    """
    
    # Check the format of satid
    if type(satid) in [str, int, np.int32, np.int64, np.str_]:
        satids = [satid]
    elif type(satid) in [np.ndarray, list]:
        satids = list(satid)
    else:
        raise ValueError("satid is not a string, int or list of both!")

    # Extract satellite names from tle
    satnames = np.array(list(map(lambda s: s.name,tle)))

    # Loop through the satids and find the satellite indices
    isat = np.empty((0,1),int)
    for i in range(len(satids)):

        if isinstance(satids[i],str):
           if "GLONASS" in satids[i].upper():
              satids[i] = 'COSMOS'
           elif "GALILEO" in satids[i].upper():
              satids[i] = 'GSAT'

           l=list(map(lambda s: satids[i] in s,satnames))
           isati = np.argwhere(l)

           if np.size(isati) == 0:
              raise ValueError("Satellite {} not in tle!".format(satids[i]))

           isat = np.concatenate(([isat, isati]))

        else:
           isat = np.append(isat, satids[i])
                  
    isat = np.unique(isat)
    satids = satnames[isat]

    if verbose > 0:
       print("Found {} satellites:".format(len(satids)))
       for i in range(len(satids)):
          print(" {}  ({})".format(tle[isat[i]].name, isat[i]))

    return isat, satids


def tledatenum(daterange):
    """Compute datenumber range from start date, end date and interval.

    Parameters
    ----------
    daterange : string, list, array_like with string, float or datetime64
        The `daterange` is either:
            
            - three element list with start date, end date (or duration in minutes) 
              and data interval in minutes, 
            - string with the date or a list with date strings, or,
            - ndarray with serial datenumbers, datestrings, or datetime64 
        
        The function return of a previous call is a valid input. 
       
    Returns
    -------
    t : ndarray, float64
        Array with serial datenumbers (days since 1-1-1970)    
    
    See Also
    --------
    tleget, tleread, tle2vec, tlefind
    
    Examples
    --------
    Examples with start date, period (min) or end date, and interval (min)
    
    >>> tledatenum(['2013-9-13 0:00:00', 24*60 ,1])
    array([15961.        , 15961.00069444, 15961.00138889, ...,
           15961.99861111, 15961.99930556, 15962.        ])

    >>> tledatenum(['2013-9-13 0:00:00', '2013-9-14 0:00:00', 1])
    array([15961.        , 15961.00069444, 15961.00138889, ...,
           15961.99861111, 15961.99930556, 15962.        ])

    Examples with string input

    >>> tledatenum('2013-9-13')  
    array([15961.])
    
    >>> tledatenum(['2013-9-13 0:0:00', '2013-9-13 1:0:00', '2013-9-13 2:0:00'])
    array([15961.        , 15961.04166667, 15961.08333333])
    
    Example showing the return of a previous call is a valid input
    
    >>> t = tledatenum(['2013-9-13 0:00:00', 24*60 ,1])
    >>> tledatenum(t) 
    array([15961.        , 15961.00069444, 15961.00138889, ...,
           15961.99861111, 15961.99930556, 15962.        ])

    """

    if type(daterange) in [np.ndarray, list]:

        if len(list(daterange)) == 3 and type(daterange[0]) == str:
            if type(daterange[2]) != str:
                t0 = datetime2num(parsedate(daterange[0]))
                if type(daterange[1]) == str:
                    t1 = datetime2num(parsedate(daterange[1]))
                    #t = np.arange(t0, t1 + daterange[2]/(24*60), daterange[2]/(24*60))
                    t = np.arange(t0, t1 + daterange[2]/(24*60)/2, daterange[2]/(24*60))
                else:
                    #t = np.arange(t0, t0 + daterange[1]/(24*60) + daterange[2]/(24*60), daterange[2]/(24*60))
                    t = np.arange(t0, t0 + daterange[1]/(24*60) + daterange[2]/(24*60)/2, daterange[2]/(24*60))
            else:
                t = np.array([datetime2num(parsedate(datestamp)) for datestamp in daterange])
        elif type(daterange[0]) == str:
            t = np.array([datetime2num(parsedate(datestamp)) if type(datestamp) == str else datestamp
                                 for datestamp in daterange])
        else:
            t = daterange
            
    elif type(daterange) == str:
        t = np.array([ datetime2num(parsedate(daterange)) ])
    elif type(daterange) in [int, float, np.float64, np.int32, np.int64]:
        t = np.array([daterange])
    else:
        raise ValueError("Unknown input type {}".format(type(daterange)))

    return t


def tle2orb(tle, t, propagation="J2"):
    """Compute satellite orbital elements from NORAD Two Line Elements.

    Compute satellite orbital elements from NORAD Two Line Elements at
    times t, using basic orbit propagation (with J2).
    
    Parameters
    ----------
    tle : named tuple
        Named tuple with two line elements for a single satellite 
    t : array_like, float, with shape (n,)
        Array with serial date numbers in UT1.
    propagation : {'J2', 'NOJ2', 'SGP4'}, optional
        Propagation method to for the orbital elements::
            
            J2    Include secular effects of J2, but nothing else (default). This gives
                  acceptable results for plotting purposes and planning computations
            NOJ2  Ignores effect of J2 on orbit propagation. Should only be used for 
                  educational purposes
            SGP4  SGP4 orbit propagator of NORAD. Not implemented yet
    
    Returns
    -------
    orb : ndarray with shape (n,6)
        Array with rows of orbital elements '[semi-major axis (m), eccentricity (-),
        inclination (rad), right ascension of ascending node (rad), argument of 
        periapsis (rad]) and true anomaly (rad)]' at time 't[i]'. 
    
    See Also
    --------
    tleget, tleread, tle2vec
    
    
    """
    
    # Define constants
    J2 = 0.00108262998905
    Re = 6378136  # m, radius of Earth
    mu = 3986004418e5  # m^2/s^2 , gravitational constant of Earth

    # Prepare the array with times
    t = tledatenum(t)
    n_epoch = t.shape[0]
    t0 = datetime2num(datetime(year=tle.year, month=1, day=1)) + tle.epoch - 1

    # Orbit propagation
    if propagation.upper() == "J2":
        
       # Compute rate of change of orbital elements
       #
       #   draan/dt = s.*cos(inclination)
       #   dargp/dt = -0.5*s.*(5*cos(inclination-1).^2)
       #   dM/dt = -0.5*s.*sqrt(1-e.^2).*(3*cos(inclination).^2 -1)
       #
       # with s=-J2*3/2*sqrt(mu/a^3)*(Re/p)^2
       #
       # dM/dt is not needed for two line element propagation, but computed nevertheless. 

        p = tle.a0*(1-tle.ecc0**2)
        s = -J2 * 3 / 2 * np.sqrt(mu / tle.a0**3) * (Re / p)**2
        odot = s * np.cos(tle.inc0) * 86400
        wdot = -0.5 * s * (5*np.cos(tle.inc0)**2 - 1) * 86400
        # mdot = -0.5 * s * np.sqrt(1-tle.ecc0**2) * (3*np.cos(tle.inc0)**2 - 1) * 86400

        raan = tle.raan0 + odot * (t-t0)
        argp = tle.argp0 + wdot * (t-t0)

        m = tle.m0 + tle.n0 * (t-t0)
        ignore, nu = keplerm(m + argp, tle.ecc0)
        nu -= argp

        orb = np.hstack([np.array([tle.a0, tle.ecc0, tle.inc0] * n_epoch).reshape((n_epoch, 3)),
                         raan.T[:, np.newaxis], argp.T[:, np.newaxis], nu.T[:, np.newaxis]])

    elif propagation.upper() == "NOJ2":
        
        # Very simple orbit propagation ignoring effect of J2 (use with
        # extreme caution, only for educational purposes)

        m = tle.m0 + tle.n0 * (t - t0)
        ignore, nu = keplerm(m, tle.ecc0)
        orb = np.hstack([np.array([tle.a0, tle.ecc0, tle.inc0, tle.raan0, tle.argp0] * n_epoch).reshape((n_epoch, 5)),
                         nu.T[:, np.newaxis]])

    elif propagation.upper() == "SGP4":
        raise ValueError("SGP4 propagation method not implemented yet!")
    else:
        raise ValueError("Unknown propagation method {}! Please use J2, NOJ2, (or SGP4)!".format(propagation))

    return orb


def tle2vec(tle, t, satid, propagation="J2", verbose=0):
    """Compute satellite position and velocity in ECI from NORAD Two Line Elements.
    
    Compute satellite position and velocity in ECI coordinates from NORAD two 
    line elements, for selected satellites, at given times.
    
    Parameters
    ----------
    tle : list with named tuples
        List with two line elements as named tuples as return by `tleread`.
    t : array_like, float, with shape (n,)
        Array with serial date numbers in UT1 or date range as used by `tledatenum`. 
        Valid inputs are:
            
            - three element list with start date, end date (or duration in minutes) 
              and data interval in minutes, 
            - string with the date or a list with date strings, or,
            - ndarray with serial datenumbers, datestrings, or datetime64 
            
    satid : str, int, array_like with int or str
        String or array with the satellite names, index or index array. For strings, 
        satellites are selected that start with `satid`. Uses `tlefind` to resolve
        the satellite identifiers.
    propagation : {'J2', 'NOJ2', 'SGP4'}, optional
        Propagation method to for the orbital elements::
            
            J2    Include secular effects of J2, but nothing else (default). This gives
                  acceptable results for plotting purposes and planning computations
            NOJ2  Ignores effect of J2 on orbit propagation. Should only be used for 
                  educational purposes
            SGP4  SGP4 orbit propagator of NORAD. Not implemented yet.
            
    verbose : int, optional
        Verbosity level. If 'verbose=1' (default) print a message for the found
        TLE(s), if 'verbose=0' the function is quiet.
    
    Returns
    -------
    xsat, vsat : ndarray with shape (n,3) each
        The satellite position (m) and velocity (m/s) in an ECI reference
        frame. The second to last axis is time t, the last axis are the 
        coordinates X, Y and Z, respectively velocities VX, VY and VZ.
    t : array_like, float, with shape (n,)
        Array with serial date numbers in UT1.
    satids : array_like, str
        Array with the satellite names.

    Notes
    -----
    The default propagation method is J2 which includes the secular 
    effects of J2. This gives acceptable results for plotting purposes and 
    planning computations. Formally, TLE are only compatible with the SGP4 
    orbit propagator of NORAD, but this is not (yet) supported by this 
    function. The other available option is NOJ2, which ignores the effect 
    of J2 on the orbit propagation and should only be used for educational 
    purposes.

    See Also
    --------
    tleget, tleread, tledatenum, tlefind, tle2orb
    
    Examples
    --------
    Read Earth resource satellite TLE's and get position and velocity of RADARSAT-2
    at a specific time
    
    >>> tleERS = tleread('resource-20240802.tle', verbose=0)
    >>> xsat, vsat, *_ = tle2vec(tleERS,'2024-08-02 05:58:10','RADARSAT-2') 
    >>> print(xsat, vsat)
    [[5458404.60763595 4591181.8357549  -732126.12944127]] [[  144.29836955 -1339.50959494 -7333.2677589 ]]

    Compute RADARSAT-2 position and velocities for Aug 2, 2024, with one minute intervals

    >>> xsat, vsat, t, *_ = tle2vec(tleERS,['2024-08-02 0:00', 24*60, 1],'RADARSAT-2')
    >>> print(xsat.shape, vsat.shape, t.shape)
    (1441, 3) (1441, 3) (1441,)
    
    >>> t = tledatenum(['2024-08-02 0:00', 24*60, 1])      # array with date numbers ...
    >>> xsat, vsat, t, *_ = tle2vec(tleERS,t,'RADARSAT-2')
    >>> print(xsat.shape, vsat.shape, t.shape)
    (1441, 3) (1441, 3) (1441,)
  
    Position and velocity, on Aug 2, 2024, for every SENTINEL satellite, with one minute interval
  
    >>> xsat, vsat, t, satids = tle2vec(tleERS,['2024-08-02 0:00', 24*60, 1],'SENTINEL')
    >>> print(xsat.shape, vsat.shape, t.shape, satids.shape)
    >>> print(satids)
    (7, 1441, 3) (7, 1441, 3) (1441,) (7,)
    ['SENTINEL-1A' 'SENTINEL-2A' 'SENTINEL-3A' 'SENTINEL-2B' 'SENTINEL-5P'
     'SENTINEL-3B' 'SENTINEL-6']

    """
      
    # Prepare the dates
    t = tledatenum(t)
    
    # Find the satellite and compute the position and velocity in ECI frame
    isat, satids = tlefind(tle, satid, verbose)
    if len(isat) == 0:
        # No satellite found, return empty without complaining
        vec = np.empty((0,6))
    elif len(isat) > 1:
        # Compute position and velocity in ECI for multiple satellites
        orb = np.empty((len(isat),t.shape[0],6))
        for k in range(len(isat)):
            orb[k,...] = tle2orb(tle[isat[k]], t, propagation)
        vec = orb2vec(orb)
    else:
        # Compute position and velocity in ECI
        orb = tle2orb(tle[isat[0]], t, propagation)
        vec = orb2vec(orb)

    xsat = vec[...,:3]
    vsat = vec[...,3:]

    return xsat, vsat, t, satids



if __name__ == "__main__":
    satid = 'gps'
    satellite = 'GPS BIIR-9  (PRN 21)'
    fname = tleget(satid)
    tlelist = tleread(fname)
    xsat, vsat = tle2vec(tlelist, ['2020-11-8 0:00:00', 24 * 60, 1], satellite)
