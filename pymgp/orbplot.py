"""
Satellite orbit and visibility plots.

Plot satellite orbits in Earth Centered Inertial (ECI) and as ground tracks in
Earth Centered Earth Fixed (ECEF) reference frame, plot information on visibility
from an observer, and look angles, range and range-rate.

Functions
---------
High level plot functions 

- pltposvel         Plot satellite position and velocity
- pltsattrack       Plot satellite orbit tracks in ECI (right-ascension and declination)
- pltgroundtrack    Plot satellite ground track(s).
- pltsatview        Plot elevation angle, azimuth angle, range and range rate
- skyplot           Skyplot (polar plot) with elevation and azimuth of satellite(s)

Low level plot functions

- plot_orbit_3D     Plots 3d orbits of satellites around the Earth

Plot support functions 

- rewrap            Rewrap cyclic data for plotting line segments
- rewrap2d          Rewrap two dimensional cyclic data for plotting line segments
- xlabels           Prepare x-axis labels and title string entries

Supporting functions

- plh2ecef          Convert ECEF latitude, longitude and height to Cartesian coordinates
- plh2eci           Convert ECEF latitude, longitude and height to ECI position and velocity
- xyz2zassp         Compute zenith angle, azimuth and distance
- obj2sat           Compute zenith angle, azimuth, range and range-rate
- xyz2radec         Compute right ascension and declination
- ra2lon            Convert right ascension in ECI to longitude in ECEF

Notes
-----
High level functions can be called multiple times to plot more than one satellite
and several (but not all) support multidimension arrays for plotting multiple satellites.

See also
--------
satorb:
    Satellite orbit module (used by this module)
tleplot:
    Plotting of satellite orbits (relies on this module)

Examples
--------
The driver function `tleplot` calls on many of the functions of this module 
and is an excellent demonstration of this module's capabilities

>>> from tleorb import tleread
>>> form tleorb.plot import tleplot
>>> tle = tleread('resource-20240802.tle')
>>> tleplot(tle,{'2017-10-10 0:00', 24*60 ,1},'SENTINEL-1A',[ 52 4.8  0 ])

Copyright 2012-2024, Hans van der Marel, Delft University of Technology.
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
Modified:   12 Nov 2020 by Hans van der Marel and Simon van Diepen
             - initial port to Python (part of `tleplot` module)
            15 Nov 2021 by Hans van der Marel
             - plots wthout object coordinates possible
            12 Jul 2024 by Hans van der Marel
             - refactored `tleplot` module into `tleorb` and `tleplot` (`orbplot`)
             2 Aug 2024 by Hans van der Marel
             - imports from new module `satorb` instead of `crsutil`
             5 Aug 2024 by Hans van der Marel
             - major overhaul
             - split `tleplot` into `orbplot` with general pupose plotting
               function (no reliance on tle) and a driver function `tleplot`
             - rewrite of all functions using object oriented interface
             - added many new parameters and options
             - support for higher dimensional arrays and multiple satellites
               (as is the case in the original matlab code) in tle2vec
             - support for multiple calls (including recursion) of high level
               plots
             9 Aug 2024 by Hans van der Marel
             - updated docstrings to numpy style
                                             
Based on code originally developed for Matlab(TM) in 2012-2016 by the author.

Copyright Hans van der Marel, Delft University of Technology, 2012-2024
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from scipy.io import loadmat
from satorb import ut2gmst, num2datetime


# ----------------------------------------------------------------------------
# HIGH LEVEL SATELLITE PLOT FUNCTIONS 
#
# - pltposvel(t, xsat, vsat, satid="", crs="ECI")
#       Plot satellite position and velocity
# - pltsattrack(t, xsat, satid="", xobj=None, llhobj=None, cutoff=0, stamps=[3, 1, 5], rsopt=[True, True],
#                    ax=None, figsize=(10,6), color=None, hlcolor=None, fontsize="small")
#       Plot satellite orbit tracks in ECI (right-ascension and declination)
# - pltgroundtrack(xsate, satid="", xobje=None, visible=[], **kwargs)
#       Plot satellite ground track(s).
# - pltsatview(t, zen, azi, robj2sat, rrobj2sat, visible=[], satid="" )
#       Plot elevation angle, azimuth angle, range and range rate
# - skyplot(t, azi, zen, cutoff=0, satnames=[])
#       Skyplot (polar plot) with elevation and azimuth of satellite(s)
# - xxxxxxx(ax, xsat, xobj=None)
#       Plots 3d orbits of satellites around the Earth
#
# ----------------------------------------------------------------------------

"""
    Uniform parameters for Satellite plot functions.

    Parameters
    ----------
    t : array_like, float, shape (n,)
        Array with datenumbers (days since 1-1-1970)
    xsat, vsat : array_like, float, shape (...,n,3)
        Array with Cartesian X, Y, Z satellite coordinates (m) and velocities 
        (m/s) in the reference system given by `crs`.
    xobj: array_like, float, shape(3,) or (n,3)
        Array with the Cartesian X, Y, Z object coordinates (m) in the reference 
        system given by `crs`.
    crs : {'ECI', 'ECEF'}, optional
        Coordinate reference system for `xsat`, `vsat` and `xobj`
    zen, azi, robj2sat, rrobj2sat : array_like, float, shape (n,) or (...,n)
        Arrays with respectively the zenith angle (rad), azimuth angle (rad),
        range (m) and range-rate (m/s). 
    visible : array_like, boolean, optional
        Boolean array, True if the satellite is visible, False when the satellite
        is below the elevation cut-off. The shape must match the data arrays, 
        except for the last dimension.
    satid : string
        Name of the satellite (for functions that support a single satellite only)
    satnames : array_like, str_
        Array with the satellite names (for functions supporting multiple satellites)
    cutoff : float
        Elevation cut-off angle (deg)

    ax:
    
    kwargs: optional
        Any other matplotlib parameter 

    Other parameters
    ----------------
    xsate : array_like, float, shape (...,n,3) or tuple of ndarray
        Used by pltgroundtrack
        - Array with Cartesian ECEF X, Y, Z coordinates (m)
        - Tuple (lon, lat) with lon and lat ndarray with the ECEF longitude 
          and latitude [degrees].
        Find another way to do this? Maybe llh, lat, lon parameters.
        
    Notes
    -----    
    Coast lines are plotted when the file coast.mat is your working directory.

    If you want multiple satellites to be plotted call this function 
    repeatedly for the different satellites, like in the example below.
    
    Examples
    --------
    
    Example calling pltgroundtrack repeatedly
    
    >>> plt.figure("Ground tracks")
    >>> pltgroundtrack((sat1, dsat), visible=visible, satid=satid)
    >>> pltgroundtrack(xsate, satid=satid,linewidth=0.5)
    >>> plt.title("Ground tracks for two satellites.")
    >>> plt.legend() 

    Example skyplot with multiple satellites

    >>> tlegps = tleread('gps.txt','verbose',0)
    >>> t = tledatenum(['2017-11-30',24*60,1])

    >>> latlon = np.radians([ 52, 4.8 ]);
    >>> xobje = 6378136 * [ np.cos(latlon[1]) * np.cos(latlon[2]) , 
    ...                    np.cos(latlon[1]) * np.sin(latlon[2]) ,
    ....                   np.sin(latlon[1]) ]

    >>> nepo = t.shape[0]
    >>> ngps = tlegps.shape[0]
    >>> zgps = np.full([nepo,ngps],np.nan)
    >>> agps = np.full([nepo,ngps],np.nan)
    >>> for k in range(ngps)
    >>>   xgpsi, vgpsi = tle2vec(tlegps, t, k)
    >>>   xgpse, vgpse = eci2ecef(t, xgpsi, vgpsi)
    >>>   lookangles, _ = satlookanglesp(t, [xgpse, vgpse], xobje)
    >>>   zgps[:,k] = lookangles[:,1]
    >>>   agps[:,k] = lookangles[:,2]
          
    >>> plt.figure(title)
    >>> skyplot(t, agps, zgps, cutoff=10)

    (c) Hans van der Marel, Delft University of Technology, 2017-2020
    
    Created:  30 November 2017 by Hans van der Marel for Matlab
    Modified: 19 November 2020 by Hans van der Marel and Simon van Diepen
                 - port to Python
"""

def pltposvel(t, xsat, vsat, satid="", crs="ECI", ax=None, figsize=(10,6), 
                 style={'linewidth':2}, rstyle={'linewidth':2, 'color':'k'}):
    """Plot satellite position and velocity."""

    satid=np.asarray(satid)

    assert xsat.ndim == 2, "xsat and vsat must be two dimensional arrays (one satellite only)"
    assert xsat.shape[-1] == 3, "xsat and vsat must have rows with three elements"
    assert xsat.shape == vsat.shape, "xsat and vsat must have the same shape"
    assert t.shape[-1] == xsat.shape[0], "t and xsat have incompatible shapes"
    assert satid.shape[-1] == 1, "satid can only contain one satellite id"

    satid=satid[0]

    xtimefmt, xlabeldate, *_ = xlabels(t)

    if ax is None:
        #fig, ax = plt.subplots(num="ECI Position and Velocity", nrows=2, ncols=1, figsize=figsize, tight_layout=True)
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize, tight_layout=True)
        # fig.suptitle("{} position and velocity in {} ({})".format(satid, crs, xlabeldate))

    rsat = np.sqrt(np.sum(xsat**2, axis=1))
    velsat = np.sqrt(np.sum(vsat**2, axis=1))

    ax[0].plot(t, np.array(xsat[:, 0])/1000, label="X", **style)
    ax[0].plot(t, np.array(xsat[:, 1])/1000, label="Y", **style)
    ax[0].plot(t, np.array(xsat[:, 2])/1000, label="Z", **style)
    ax[0].plot(t, rsat/1000, label="r", **rstyle)
    ax[0].set_title("{} position and velocity in {} ({})".format(satid, crs, xlabeldate))
    ax[0].set_ylabel("Position [km]")
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter(xtimefmt))
    ax[0].xaxis.set_major_locator(mdates.AutoDateLocator())
    ax[0].legend()

    ax[1].plot(t, np.array(vsat[:, 0])/1000, label="V_X", **style)
    ax[1].plot(t, np.array(vsat[:, 1])/1000, label="V_Y", **style)
    ax[1].plot(t, np.array(vsat[:, 2])/1000, label="V_Z", **style)
    ax[1].plot(t, velsat/1000, label="v", **rstyle)
    ax[1].set_ylabel("Velocity [km/s]")
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter(xtimefmt))
    ax[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    ax[1].set_xlabel("Date {}".format(xlabeldate))
    ax[1].legend()

    return ax

def pltsatview(t, zen, azi, robj2sat, rrobj2sat, satid="", cutoff=0, ax=None, figsize=(10,6), 
               style={'linestyle':'-', 'linewidth':0.5, 'color':'k'}, hlstyle={'color':'g', 'linewidth':'3'}):
    """Plot elevation angle, azimuth angle, range and range rate."""

    satid=np.asarray(satid)

    assert zen.ndim == 1, "zen, azi, robj2sat and rrobjsat must be one dimensional arrays (one satellite only)"
    assert azi.shape == zen.shape, "azi and zen must have the same shape"
    assert robj2sat.shape == zen.shape, "robj2sat and zen must have the same shape"
    assert rrobj2sat.shape == zen.shape, "rrobj2sat and zen must have the same shape"
    assert t.shape == zen.shape, "t and zen have incompatible shapes"
    assert satid.shape[-1] == 1, "satid can only contain one satellite id"

    satid=satid[0]

    xtimefmt, xlabeldate, *_ = xlabels(t)

    if ax is None:
        fig, ax = plt.subplots(num="Viewing angles", nrows=2, ncols=2, figsize=figsize, tight_layout=True)
        fig.suptitle("{} ({})".format(satid, xlabeldate))

    elevation = np.pi/2 - zen   
    visible = elevation >= np.radians(cutoff)
    vislabel = f'El. > {cutoff} deg'

    tmasked = np.ma.array(t, mask=~visible)
    elevationmasked = np.ma.array(elevation, mask=~visible)
    robj2satmasked = np.ma.array(robj2sat, mask=~visible)
    rrobj2satmasked = np.ma.array(rrobj2sat, mask=~visible)
    
    twrap, aziwrap, ins = rewrap(t, azi * 180/np.pi, method='rewrap')
    visiblewrap = np.insert(visible, ins, False)
    azimaskedwrap = np.ma.array(aziwrap, mask=~visiblewrap)

    ax[0,0].plot(t, elevation * 180/np.pi, label='Elevation', **style)
    ax[0,0].plot(tmasked, elevationmasked * 180/np.pi, label=vislabel, **hlstyle)
    ax[0,0].set_title('Elevation angle [deg]')
    ax[0,0].xaxis.set_major_formatter(mdates.DateFormatter(xtimefmt))
    ax[0,0].xaxis.set_major_locator(mdates.AutoDateLocator())
    ax[0,0].legend()

    ax[1,0].plot(twrap, aziwrap, label='Azimuth', **style)
    ax[1,0].plot(twrap, azimaskedwrap, label=vislabel, **hlstyle)
    ax[1,0].set_title('Azimuth angle [deg]')
    ax[1,0].xaxis.set_major_formatter(mdates.DateFormatter(xtimefmt))
    ax[1,0].xaxis.set_major_locator(mdates.AutoDateLocator())
    ax[1,0].legend()

    ax[0,1].plot(t, robj2sat / 1000, label='Range', **style)
    ax[0,1].plot(tmasked, robj2satmasked / 1000, label=vislabel, **hlstyle)
    ax[0,1].set_title('Range [km]')
    ax[0,1].yaxis.tick_right()
    ax[0,1].xaxis.set_major_formatter(mdates.DateFormatter(xtimefmt))
    ax[0,1].xaxis.set_major_locator(mdates.AutoDateLocator())
    ax[0,1].legend()

    ax[1,1].plot(t, rrobj2sat / 1000, label='Range rate', **style)
    ax[1,1].plot(tmasked, rrobj2satmasked / 1000, label=vislabel, **hlstyle)
    ax[1,1].set_title('Range rate [km/s]')
    ax[1,1].yaxis.tick_right()
    ax[1,1].xaxis.set_major_formatter(mdates.DateFormatter(xtimefmt))
    ax[1,1].xaxis.set_major_locator(mdates.AutoDateLocator())
    ax[1,1].legend()
    
    return ax
    

def pltsattrack(t, xsat, satid="", xobj=None, llhobj=None, cutoff=0, stamps=[3, 1, 5], rsopt=[True, True],
         ax=None, figsize=(10,6), color=None, hlcolor=None, fontsize="small"):
    """Plot satellite orbit tracks in ECI (right-ascension and declination).""" 

    # Optionally compute xobj from llhobj (lat, lon, height of the object in ECEF), if both xobj
    # and llhobj are givem, xobj takes precedence
    if not llhobj is None and xobj is None:
        xobj, vobj = plh2eci(t, llhobj, unit='deg')

    # Compute right-ascension and declination of the observer
    if not xobj is None:
        aobj, dobj = xyz2radec(xobj, unit='deg')

    # Initialize the plot ax=None, if not, add satellite track to existing axis ax

    xtimefmt, xlabeldate, xlabelrange = xlabels(t)

    if ax is None:
        fig, ax = plt.subplots(num="ECI tracks", figsize=figsize, tight_layout=True)

        # ax.set_title("{} orbit track in ECI ({})".format(satid,xlabeldate))
        ax.set_title("Satellite orbit track in ECI ({})".format(xlabelrange))
    
        # Plot right ascension and declination of the observer (only on initialization)
        
        if not xobj is None:
            # Unwrap (ra, dec) and plot as lines
            aobjwrap, dobjwrap, *_ = rewrap2d(aobj, dobj)
            ax.plot(aobjwrap, dobjwrap, color='gray', linewidth=1, label="Observer") 
            # plot start symbol (cicle)
            ax.scatter(aobj[0], dobj[0], color="none", edgecolor='black', marker="o", s=100) 
            # plot end symbol (rotated arrow)
            theta = np.degrees(np.arctan2(dobj[-1]-dobj[-2],aobj[-1]-aobj[-2]))
            tf = Affine2D().rotate_deg(theta)
            ax.scatter(aobj[-1], dobj[-1], color="gray", edgecolor='black', marker=MarkerStyle('>', 'full', tf), s=60) 
            # Add legend entries
            ax.plot([0],[0], linewidth=4, color='y', label=f'Elevation > {cutoff} deg')
            #if rsopt[0]:
            #    ax.plot([0],[0], linewidth=0.5, linestyle=':', color='black', label="Rise/set")
    
    # With multiple satellites, call the function recursively
    
    if xsat.ndim > 2:
        for k in range(0,xsat.shape[-3]):
            pltsattrack(t,xsat[k,:,:], satid=satid[k], xobj=xobj, cutoff=cutoff, stamps=stamps, rsopt=rsopt,
                ax=ax, color=color, hlcolor=hlcolor, fontsize=fontsize)
        return

    if satid.ndim > 0:
        satid=satid[0]

    # Plot right ascension and declination of the satellite

    asat, dsat = xyz2radec(xsat, unit='deg')

    # Unwrap ra, dec and plot as lines
    asatwrap, dsatwrap, ins = rewrap2d(asat, dsat)
    if color is None:
        ll, *_ = ax.plot(asatwrap, dsatwrap, linewidth=0.5, linestyle="-", label=satid) 
        satcolor = ll.get_color()
    else: 
        satcolor = color
        ax.plot(asatwrap, dsatwrap, color=satcolor, linewidth=0.5, linestyle="-", label=satid) 
    # plot start symbol (cicle)
    ax.scatter(asat[0], dsat[0], color="none", edgecolor='black', marker="o", s=100) 
    # plot end symbol (rotated arrow)
    theta = np.degrees(np.arctan2(dsat[-1]-dsat[-2],asat[-1]-asat[-2]))
    tf = Affine2D().rotate_deg(theta)
    ax.scatter(asat[-1], dsat[-1], color=satcolor, edgecolor='black', marker=MarkerStyle('>', 'full', tf), s=80) 

    # plot time stamps on the satellite track
    
    if not stamps is None:
        # stamps is [ num stamps at start, numstamps at end, interval ], default is [3, 1, 5]
        for k in range(0,(stamps[0]-1)*stamps[2]+1,stamps[2]):
            ax.scatter(asat[k],dsat[k], marker=".", color='black')
            ax.text(asat[k]-5,dsat[k], num2datetime(t[k]).strftime(xtimefmt),horizontalalignment='right', color=satcolor, fontsize=fontsize)       
        #for k in range(stepsize,t.shape[0],stepsize):
        for k in range(len(t)-(stamps[1]-1)*stamps[2]-1,len(t),stamps[2]):
            ax.scatter(asat[k],dsat[k], marker=".", color='black')
            ax.text(asat[k]+5,dsat[k], num2datetime(t[k]).strftime(xtimefmt), color=satcolor, fontsize=fontsize)       

    # Add satellite visibility information to the plot
    # - plot satellite track in different color (yellow) when visible from observer
    # - highlight observer track when the satellite is visible (using yellow plus symbols -> looks like a thick line)
    # - draw connecting lines (dotted, black)
    # - add hh:mm at rise and set times (rise above the observer track, set below observer track) 

    if not xobj is None:
        # Compute zenith angle and visibility
        zen, azi, *_ = xyz2zassp(xsat, xobj)
        visible = zen <= np.radians( 90 - cutoff )        
        # highlight observer track when the satellite is visible (using yellow plus symbols -> looks like a thick line)
        ax.scatter(aobj[visible], dobj[visible], marker="+", color=satcolor, s=30) #, label="visible observer")
        # plot satellite track in different color (yellow) when visible from observer (unwrapping ra, dec)
        viswrap = np.insert(visible, ins, False)
        asatmask = np.ma.array(asatwrap, mask=~viswrap)
        dsatmask = np.ma.array(dsatwrap, mask=~viswrap)
        if hlcolor is None:
            hlcolor=satcolor
        ax.plot(asatmask, dsatmask, color=hlcolor, linewidth=4) # , label="visible") # label="visible {}".format(satid)) 
        # compute rise and set times, add connecting lines (dotted, black, unwrap), and timestamps at rise and 
        # set on the observer track
        iriseset = np.hstack([0, np.diff(visible*1)])
        for k in np.flatnonzero(iriseset == 1):
            # rise occurence
            if rsopt[1]:
                ax.text(aobj[k], dobj[k]+2, num2datetime(t[k]).strftime(xtimefmt), horizontalalignment='right', color=satcolor, fontsize=fontsize)
            if rsopt[0]:
                x, y, *_ = rewrap2d(np.array([ aobj[k], asat[k] ]), np.array([ dobj[k], dsat[k] ]))
                ax.plot( x, y, linewidth=0.5, color=satcolor, linestyle=':')
        for k in np.flatnonzero(iriseset == -1):
            # set occurence
            if rsopt[1]:
                ax.text(aobj[k-1], dobj[k-1]-6, num2datetime(t[k]).strftime(xtimefmt), color=satcolor, fontsize=fontsize)
            if rsopt[0]:
                x, y, *_ = rewrap2d(np.array([ aobj[k-1], asat[k-1] ]), np.array([ dobj[k-1], dsat[k-1] ]))
                ax.plot(x, y, linewidth=0.5, color=satcolor, linestyle=':')

    #plt.axis([-180, 180, -90, 90])
    ax.set(xlim=(-180, 180), ylim=(-90, 90))
    ax.set_xlabel("Right ascension")
    ax.set_ylabel("Declination")
    ax.legend()

    return ax


def pltgroundtrack(t, xsate, satid="", xobje=None, llhobj=None, cutoff=0, stamps=[3, 1, 5], rsopt=[True, True],
         ax=None, figsize=(10,6), color=None, hlcolor=None, fontsize="small", **kwargs):
    """ Plot satellite ground track(s).
    
    pltgroundtrack(xsate, ...) plot the satellite ground track for a satellite
    with in xsate the Cartesian ECEF coordinates. xsate must be a ndarray with 
    three columns, containing the X, Y, Z Cartesian ECEF coordinates in [m].
    
    pltgroundtrack((lon,lat), ...) plot the satellite ground track for a 
    satellite with the longitude and latitude given in the tuple (lon, lat).
    lon and lat must be ndarray with the ECEF longitude and latitude [degrees].
    
    The function takes optional parameters, satid, visible and/or any matplotlib, 
    argument, with satid the name of the satellite, and visible an optional logical array
    (of the same shape as lon and lat) with True for the elements where the
    satellite is visible.

    Coast lines are plotted when the file coast.mat is your working directory.

    If you want multiple satellites to be plotted call this function 
    repeatedly for the different satellites, like in the example below.
    
    Example:

       plt.figure("Ground tracks")
       pltgroundtrack((sat1, dsat), visible=visible, satid=satid)
       pltgroundtrack(xsate, satid=satid,linewidth=0.5)
       plt.title("Ground tracks for two satellites.")
       plt.legend() 

    See also tleread, tle2vec and eci2ecef.
    """

    """
    (c) Hans van der Marel, Delft University of Technology, 2017-2020
    
    Created:  30 November 2017 by Hans van der Marel for Matlab
    Modified: 19 November 2020 by Hans van der Marel and Simon van Diepen
                 - port to Python
    """

    # Optionally compute xobj from llhobj (lat, lon, height of the object in ECEF), if both xobje
    # and llhobj are given, llhobj takes precedence
    # Compute latitide and longitude of the observer from xobje
    if llhobj is None and not xobje is None:
        llhobj = ecef2plh(xobje, unit='deg')
    if xobje is None and not llhobj is None:
        xobje = np.tile(plh2ecef(llhobj, unit='deg'), (xsate.shape[0], 1))

    # Initialize the plot ax=None, if not, add satellite track to existing axis ax

    xtimefmt, xlabeldate, xlabelrange = xlabels(t)
    
    #if objcrd != None:
    #    pltgroundtrack((lsat, dsat), visible=visible, satid=satid)
    #    plt.scatter(lobj, dobj, color="g", marker="*", s=49, label="observer")
    #else:
    #    pltgroundtrack((lsat, dsat), satid=satid)
    #plt.legend() 

    if ax is None:
        fig, ax = plt.subplots(num="Ground tracks", figsize=figsize, tight_layout=True)

        #plt.title("{} ground tracks ({})".format(satid, xlabelrange))
        ax.set_title("Satellite ground tracks ({})".format(xlabelrange))

    # Plot the axis and base map (only first time)
    
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        ax.axis([-180, 180, -90, 90])
        if os.path.exists("coast.mat"):
            coast = loadmat("coast.mat")
            ax.plot(coast["long"], coast["lat"], color=(0.35, 0.35, 0.35), linewidth=0.5)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        # Plot location of the observer (only on initialization)
        
        if not llhobj is None:
            ax.scatter(llhobj[1], llhobj[0], color="g", marker="*", s=49, label="Observer")
            # Add legend entries
            ax.plot([0],[0], linewidth=4, color='y', label=f'Elevation > {cutoff} deg')
             
    # With multiple satellites, call the function recursively
    
    if xsate.ndim > 2:
        for k in range(0,xsate.shape[-3]):
            ax = pltgroundtrack(t,xsate[k,:,:], satid=satid[k], llhobj=llhobj, cutoff=cutoff, stamps=stamps, rsopt=rsopt,
                ax=ax, color=color, hlcolor=hlcolor, fontsize=fontsize, **kwargs)
        return ax

    if satid.ndim > 0:
        satid=satid[0]
    
    # Check if we have Cartesian or Spherical coordinates

    if isinstance(xsate,tuple):
        # we hape a tuple with longitude and latitude
        lsat = xsate[0]
        dsat = xsate[1]
    elif xsate.shape[1] == 3:
        # we have Cartesian coordinates -> compute longitude and latitude
        lsat = np.arctan2(xsate[:,1], xsate[:,0]) * 180 / np.pi
        dsat = np.arctan(xsate[:,2] / np.sqrt(xsate[:,1]**2 + xsate[:,0]**2) )  * 180 / np.pi
    else:
        raise ValueError("xsate must be a tuple with (lon,lat) or ndarray with 3 columns with Cartesian coordinates!")
                
    # Unwrap ra, dec and plot as lines
    lsatwrap, dsatwrap, ins = rewrap2d(lsat, dsat)
        
    # Plot the ground tracks
       
    if color is None:
        #ll, *_ = ax.plot(asatwrap, dsatwrap, linewidth=0.5, linestyle="-", label=satid) 
        ll, *_ = ax.plot(lsatwrap, dsatwrap, label=satid, **kwargs)
        satcolor = ll.get_color()
    else: 
        satcolor = color
        #ax.plot(asatwrap, dsatwrap, color=satcolor, linewidth=0.5, linestyle="-", label=satid) 
        ll, *_ = ax.plot(lsatwrap, dsatwrap, color=satcolor, label=satid, **kwargs)

    # plot start symbol (cicle)
    ax.scatter(lsat[0], dsat[0], color="none", edgecolor='black', marker="o", s=100) 
    # plot end symbol (rotated arrow)
    theta = np.degrees(np.arctan2(dsat[-1]-dsat[-2],lsat[-1]-lsat[-2]))
    tf = Affine2D().rotate_deg(theta)
    ax.scatter(lsat[-1], dsat[-1], color=satcolor, edgecolor=satcolor, marker=MarkerStyle('>', 'full', tf), s=80) 

    # plot time stamps on the satellite track
    
    if not stamps is None:
        # stamps is [ num stamps at start, numstamps at end, interval ], default is [3, 1, 5]
        for k in range(0,(stamps[0]-1)*stamps[2]+1,stamps[2]):
            ax.scatter(lsat[k],dsat[k], marker=".", color='black')
            ax.text(lsat[k]-5,dsat[k], num2datetime(t[k]).strftime(xtimefmt),horizontalalignment='right', color=satcolor, fontsize=fontsize)       
        #for k in range(stepsize,t.shape[0],stepsize):
        for k in range(len(t)-(stamps[1]-1)*stamps[2]-1,len(t),stamps[2]):
            ax.scatter(lsat[k],dsat[k], marker=".", color='black')
            ax.text(lsat[k]+5,dsat[k], num2datetime(t[k]).strftime(xtimefmt), color=satcolor, fontsize=fontsize)       

    # Add satellite visibility information to the plot
    # - plot satellite track in different color (yellow) when visible from observer
    # - highlight observer track when the satellite is visible (using yellow plus symbols -> looks like a thick line)
    # - draw connecting lines (dotted, black)
    # - add hh:mm at rise and set times (rise above the observer track, set below observer track) 

    if not llhobj is None:
        # Compute zenith angle and visibility
        zen, azi, *_ = xyz2zassp(xsate, xobje)
        visible = zen <= np.radians( 90 - cutoff )        
        # plot satellite track in different color (yellow) when visible from observer (unwrapping ra, dec)
        viswrap = np.insert(visible, ins, False)
        lsatmask = np.ma.array(lsatwrap, mask=~viswrap)
        dsatmask = np.ma.array(dsatwrap, mask=~viswrap)
        if hlcolor is None:
            hlcolor=satcolor
        ax.plot(lsatmask, dsatmask, color=hlcolor, linewidth=4) # , label="visible") # label="visible {}".format(satid)) 
        handles, labels = ax.get_legend_handles_labels()
        handles[1].set_color(hlcolor)
        # compute rise and set times and timestamps at rise and set on the satellite track
        iriseset = np.hstack([0, np.diff(visible*1)])
        for k in np.flatnonzero(iriseset == 1):
            # rise occurence
            if rsopt[1]:
                #theta = np.degrees(np.arctan2(dsat[k+1]-dsat[k],lsat[k+1]-lsat[k])) 
                theta = np.degrees(np.arctan2(dsat[k]-dsat[k-1],lsat[k]-lsat[k-1])) 
                #ax.text(lsat[k], dsat[k], num2datetime(t[k]).strftime(xtimefmt) + '↑', horizontalalignment='right', verticalalignment="bottom", rotation=theta, rotation_mode='anchor', color=satcolor, fontsize=fontsize)
                ax.text(lsat[k], dsat[k], ' ↑' + num2datetime(t[k]).strftime(xtimefmt) , horizontalalignment='left', verticalalignment="top", rotation=theta-90, rotation_mode='anchor', color=satcolor, fontsize=fontsize)
        for k in np.flatnonzero(iriseset == -1):
            # set occurence
            if rsopt[1]:
                theta = np.degrees(np.arctan2(dsat[k]-dsat[k-1],lsat[k]-lsat[k-1])) 
                #ax.text(lsat[k-1], dsat[k-1], '↓' + num2datetime(t[k-1]).strftime(xtimefmt), horizontalalignment='left', verticalalignment="bottom", rotation=theta, rotation_mode='anchor', color=satcolor, fontsize=fontsize)
                ax.text(lsat[k-1], dsat[k-1], ' ↓' + num2datetime(t[k-1]).strftime(xtimefmt), horizontalalignment='left', verticalalignment="bottom", rotation=theta-90, rotation_mode='anchor', color=satcolor, fontsize=fontsize)

    # ax.legend()

    return ax


def skyplot(t, azi, zen, cutoff=0, satnames=[], ax=None, figsize=(10,6)):
    """ Skyplot (polar plot) with elevation and azimuth of satellite(s).
    
    skyplot(t,azi,zen) creates a polar plot with the elevation and 
    azimuth of a satellite. t is an array with the time as datenumbers, azi 
    and zen are arrays with the azimuth (radians) and zenith angle 
    (radians). The first axis of azi and zen must have the same length as
    the array with time t. The second axis represents different satellites,
    i.e.  ntimes, nsat = zen.shape . The optional parameter cutoff is the 
    cutoff elevation (degrees) for plotting the trajectories, the default is
    0 degrees (the local horizon). 

    Example:

       tlegps = tleread('gps.txt','verbose',0)
       t = tledatenum(['2017-11-30',24*60,1])

       latlon = np.radians([ 52, 4.8 ]);
       xobje = 6378136 * [ np.cos(latlon[1]) * np.cos(latlon[2]) , 
                           np.cos(latlon[1]) * np.sin(latlon[2]) ,
                           np.sin(latlon[1]) ]

       nepo = t.shape[0]
       ngps = tlegps.shape[0]
       zgps = np.full([nepo,ngps],np.nan)
       agps = np.full([nepo,ngps],np.nan)
       for k in range(ngps)
          xgpsi, vgpsi = tle2vec(tlegps, t, k)
          xgpse, vgpse = eci2ecef(t, xgpsi, vgpsi)
          lookangles, _ = satlookanglesp(t, [xgpse, vgpse], xobje)
          zgps[:,k] = lookangles[:,1]
          agps[:,k] = lookangles[:,2]
          
       plt.figure(title)
       skyplot(t, agps, zgps, cutoff=10)

    See also tleread, tle2vec, eci2ecef and satlookanglessp. 
    """

    """
    (c) Hans van der Marel, Delft University of Technology, 2017-2020
    
    Created:   5 December 2017 by Hans van der Marel for Matlab
    Modified: 19 November 2020 by Hans van der Marel and Simon van Diepen
                 - port to Python
    """

    def cart2pol(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi

    def pol2cart(phi, rho):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y

    # reshape input arrays zen and azi from 1D to 2D with one column, or keep as 2D 
    zen = zen.reshape(zen.shape[0],-1)
    azi = azi.reshape(azi.shape[0],-1)

    # check the sizes of t, zen and azi
    if  zen.shape[0] != azi.shape[0] or zen.shape[1] != azi.shape[1]  : 
        raise ValueError("zen and azi input parameters must be of the same shape!") 
    elif t.shape[-1] != zen.shape[0]:
        raise ValueError("t must match the first dimension of zen and azi!") 
    
    # Check the optional satnames input
    if type(satnames) == str:
        satnames = [ satnames ]
    satnames = np.array(satnames)
    if satnames.size > 0:                # have optional satnames, check if size matches
        if satnames.shape[0] != zen.shape[1]:
            raise ValueError("the number of satellite names must match the number of columns in zen and azi!")

    # Prepare arrow tip

    xx = np.array([[-1, 0, -1]]).T
    yy = np.array([[0.4, 0, -0.4]]).T
    arrow = xx + 1j * yy
    
    # Create figure axis 

    if ax is None:
        fig, ax = plt.subplots(num="Skyplot", figsize=figsize, tight_layout=True)

    # Plot the base figure

    #plt.figure(title)
    ax.axis([-90, 90, -90, 90])
    ax.axis("equal")
    lcol = "bgrcmk"
    for i in range(0, 360, 30):
        x, y = pol2cart(-np.radians(i-90), 90)
        ax.plot([0, x], [0, y], color='gray', linewidth=1)
        x, y = pol2cart(-np.radians(i-90), 94)
        ax.text(x, y+1, "{}".format(i), fontsize=8, horizontalalignment='center', verticalalignment='center', rotation=-i)

    i_values = np.append(np.arange(0, 91, 15), cutoff)
    for i in i_values:
        az = np.arange(361)
        el = (90-i)*np.ones(az.shape)
        x, y = pol2cart(np.radians(az), el)
        ax.plot(x, y, color='gray', linestyle='-' if i != cutoff else '--', linewidth=1)
        if i not in [cutoff, 0]:
            x, y = pol2cart(-np.radians(-90), 90 - i)
            ax.text(x, y, "{}".format(i), fontsize=8, horizontalalignment='center', verticalalignment='bottom')
            #plt.text(0, 90 - i, "{}".format(i),horizontalalignment='center', verticalalignment='bottom')

    ax.axis('off')
    
    # Plot the tracks
    
    nsat = zen.shape[1]
    for k in range(nsat):

        dt = np.diff(t).min()
        idx1 = np.arange(zen.shape[0])[zen[:,k] < np.radians(90-cutoff) ]
       
        if idx1.shape[0] > 0:
            idx2 = np.append(0, np.arange(idx1.shape[0]-1)[np.diff(t[idx1]) > 3*dt])
            idx2 = np.append(idx2, idx1.shape[0]-1)
           
            for j in range(idx2.shape[0]-1):
    
                idx3 = idx1[idx2[j]+1:idx2[j+1]]
    
                x, y = pol2cart(-azi[idx3,k] + np.pi/2, np.degrees(zen[idx3,k]))
                ax.plot(x, y, linewidth=2, color=lcol[k % len(lcol)])
    
                if idx3.shape[0] > 1:
                    tx, ty = pol2cart(-azi[idx3[-1],k] + np.pi / 2, np.degrees(zen[idx3[-1],k]))
                    txx, tyy = pol2cart(-azi[idx3[-2],k] + np.pi/2, np.degrees(zen[idx3[-2],k]))
                    dd = np.sqrt((tx-txx)**2+(ty-tyy)**2)
                    z = (tx-txx)/dd + 1j * (ty-tyy)/dd
                    a = arrow * z
                    ax.plot(tx + 3*a.real, ty + 3*a.imag, linewidth=2, color=lcol[k % len(lcol)])
                    tx += 6*(tx-txx)/dd
                    ty += 5*(ty-tyy)/dd
                else:
                    try:
                        tx += 5
                        ty += 2
                    except UnboundLocalError:
                        pass

                if satnames.size > 1:
                    ax.text(tx, ty, "{}".format(satnames[k]), fontsize=8, horizontalalignment='center', verticalalignment='bottom', color=lcol[k % len(lcol)])

    # add title

    _, _, xlabelrange = xlabels(t)
   
    if satnames.size == 1:
        ax.set_title("{} Skyplot ({})".format(satnames[0], xlabelrange))
    else:
        ax.set_title("Skyplot ({})".format(xlabelrange))

    return ax    


def plt3dorbit(t, xsat, satid="", xobj=None, llhobj=None, ax=None, figsize=(10,6)):
    """3D orbit plot in ECI and ECEF."""
    
    # Optionally compute xobj from llhobj (lat, lon, height of the object in ECEF), if both xobj
    # and llhobj are givem, xobj takes precedence
    if not llhobj is None and xobj is None:
        xobj, *_ = plh2eci(t, llhobj, unit='deg')

    if satid.ndim > 0:
        satid=satid[0]
    
    # Create figure axis 

    _, _, xlabelrange = xlabels(t)

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=2, num="3D satellite orbit", figsize=figsize, tight_layout=True, subplot_kw={'projection': '3d'})
        fig.suptitle("{} 3D orbit ({})".format(satid, xlabelrange))

    # 3D plot (ECEF)

    if not xobj is None:
        plot_orbit_3D(ax[0], xsat, xobj)
    else:
        plot_orbit_3D(ax[0], xsat)
    ax[0].view_init(30, 30)    # defaults to -60 (azimuth) and 30 (elevation)
    ax[0].set_title('Inertial (ECI)')
    
    # 3D plot (ECEF)
    
    # Compute rotation angle (GMST) around Z-axis

    gst0, omegae = ut2gmst(t[0])
    gst = gst0 + 2*np.pi*omegae*(t-t[0])

    # Rotate observer positions round z-axis (ECI->ECEF) 

    xsate = np.zeros(xsat.shape)
    xsate[:, 0] = np.cos(gst)*xsat[:, 0] + np.sin(gst)*xsat[:, 1]
    xsate[:, 1] = -np.sin(gst)*xsat[:, 0] + np.cos(gst)*xsat[:, 1]
    xsate[:, 2] = xsat[:, 2]

    if not xobj is None:
        xobje = np.zeros(xobj.shape)
        xobje[:, 0] = np.cos(gst) * xobj[:, 0] + np.sin(gst) * xobj[:, 1]
        xobje[:, 1] = -np.sin(gst) * xobj[:, 0] + np.cos(gst) * xobj[:, 1]
        xobje[:, 2] = xobj[:, 2]

    if not xobj is None:
        plot_orbit_3D(ax[1], xsate, xobje)
        ax[1].scatter(xobje[:, 0]/1000, xobje[:, 1]/1000, xobje[:, 2]/1000, marker="*", s=49, color="r")
    else:
        plot_orbit_3D(ax[1], xsate)        
    ax[1].view_init(30, 30)   # defaults to -60 (azimuth) and 30 (elevation), rotate by 90 degree to make Delft visible
    ax[1].set_title('Earth Fixed (ECEF)')

    return ax

# ----------------------------------------------------------------------------
# LOW LEVEL PLOT FUNCTIONS 
#
# - plot_orbit_3D(ax, xsat, xobj=None)
#       Plots 3d orbits of satellites around the Earth
#
# ----------------------------------------------------------------------------

def plot_orbit_3D(ax, xsat, xobj=None):
    """
    Plots 3d orbits of satellites around the Earth
    """

    def set_axes_equal(ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    Re = 6378136

    # Make data
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = Re * np.outer(np.cos(u), np.sin(v)) / 1000
    y = Re * np.outer(np.sin(u), np.sin(v)) / 1000
    z = Re * np.outer(np.ones(np.size(u)), np.cos(v)) / 1000

    # Plot the surface
    ax.plot_surface(x, y, z, shade=True, edgecolor="gray", linewidth=0.5, color=[.9, .9, .9], alpha=.2)
    #ax.plot_surface(x, y, z, color=[.9, .9, .9], alpha=0.2)
    #ax.plot_wireframe(x, y, z, color='gray', linewidth=0.5)
    if type(xobj) in [list, np.array]:
        ax.plot(xobj[:, 0]/1000, xobj[:, 1]/1000, xobj[:, 2]/1000, color='r', linewidth=2, alpha=1)
    ax.plot(xsat[:, 0]/1000, xsat[:, 1]/1000, xsat[:, 2]/1000, color='b', linewidth=2, alpha=1)
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    
    set_axes_equal(ax)

    return

# ----------------------------------------------------------------------------
# PLOT SUPPORT FUNCTIONS 
#
# - rewrap(t, x, method="rewrap", wrapat=[0,360], verbose=0)
#       Rewrap cyclic data for plotting line segments
# - rewrap2d(x, y, method="rewrap", xwrapat=[-180,180], ywrapat=[-90,90], verbose=0)
#       Rewrap cyclic x and y data for plotting line segments    
# - xtimefmt, xlabeldate, xlabelrange = xlabels(t)
#       Prepare x-axis labels and title string entries
# ----------------------------------------------------------------------------
    
def rewrap(t, x, method="rewrap", wrapat=[0,360], verbose=0):
    """Rewrap cyclic data for plotting line segments
    
    Re-wrap cyclic data and insert breaks in lines so that when plotted the lines
    are properly terminated at the borders of the wrapping region. This function 
    solves two issues with plotting lines for cyclic data: 
    - the near vertical (or horizontal) line segments at the wraps, 
    - lines do not extend to the wrapping boundaries (instead they take the opposite 
      direction)
    If unsolved, line plots on cyclic data can be misleading or unclear at best, while
    a scatter plot may not provide all the information.
    
    This version corrects for wraps the ordinates (`x`) only. 

    Parameters
    ----------
    t, x: array_like, float
        The absissae (time) and ordinates (data)
    method: {'rewrap', 'unwrap', split', rewrap_nosplit'}, optional
        The rewrapping method. Defines what action is taken at a data wrap 
            - rewrap : estimate time of wrap by interpolation from the data, and insert 
                   the upper/lower wrap limits and 'np.nan' to cause a split in the line. 
                   The output arrays are larger than the the input arrays (3 additional 
                   data points per wrap)
            - rewrap_noslit : as with `rewrap`, but no 'np.nan' is inserted to split the
                   lines. The output arrays are larger than the the input arrays (2 
                   additional data points per wrap)
            - split : split the lines at a wrap by inserting 'np.nan'. The output arrays 
                   are larger than the the input arrays (1 additional data point per wrap)
            - unwrap : make the data continuous. The length of the arrays is unchanged 
    wrapat: list, optional
        List with the lower and upper wrap boundaries.
    verbose: int, optional
        Vebosity level.

    Returns
    -------
    tout, xout: ndarray, float
        The absissae (time) and rewrapped ordinates (data) for plotting
    ins: ndarray, int64
        Index array for 'np.insert', can be applied to other arrays ('z') of similar shape 
        with 'zout = np.insert(z, ins, new_z_values_to_insert)'

    Notes
    -----
    The ouput should only be used for plotting purposes, as with some options interpolated
    data points and/or `np.nan` are inserted which could bias scientific analysis.

    See also
    --------
    rewrap2d
        Rewrap cyclic data in two dimensions (e.g. longitude and latitude).

    Examples
    --------
    >>> x = np.array( [ 10., 100., 200, 300, 350, 45, 100, 180, 230, 160, 100, 10, 270, 150, 200, 300, 350, 45, 100, 200] )
    >>> t = np.arange(x.size).astype(np.float64)
    >>> rewrap(t, x, method='rewrap')
    (array([ 0.        ,  1.        ,  2.        ,  3.        ,  4.        ,
             4.18181818,  4.18181818,  4.18181818,  5.        ,  6.        ,
             7.        ,  8.        ,  9.        , 10.        , 11.        ,
            11.1       , 11.1       , 11.1       , 12.        , 13.        ,
            14.        , 15.        , 16.        , 16.18181818, 16.18181818,
            16.18181818, 17.        , 18.        , 19.        ]),
     array([ 10., 100., 200., 300., 350., 360.,  nan,   0.,  45., 100., 180.,
            230., 160., 100.,  10.,   0.,  nan, 360., 270., 150., 200., 300.,
            350., 360.,  nan,   0.,  45., 100., 200.]),
    array([ 5, 12, 17,  5, 12, 17,  5, 12, 17], dtype=int64))

    >>> rewrap(t, x, method='rewrap', wrapat=[-180, 180])
    (array([ 0.        ,  1.        ,  1.8       ,  1.8       ,  1.8       ,
             2.        ,  3.        ,  4.        ,  5.        ,  6.        ,
             7.        ,  7.        ,  7.        ,  7.        ,  8.        ,
             8.71428571,  8.71428571,  8.71428571,  9.        , 10.        ,
            11.        , 12.        , 12.75      , 12.75      , 12.75      ,
            13.        , 13.6       , 13.6       , 13.6       , 14.        ,
            15.        , 16.        , 17.        , 18.        , 18.8       ,
           18.8       , 18.8       , 19.        ]),
     array([  10.,  100.,  180.,   nan, -180., -160.,  -60.,  -10.,   45.,
             100.,  180.,   nan, -180., -180., -130., -180.,   nan,  180.,
             160.,  100.,   10.,  -90., -180.,   nan,  180.,  150.,  180.,
              nan, -180., -160.,  -60.,  -10.,   45.,  100.,  180.,   nan,
            -180., -160.]),
     array([ 2,  7,  9, 13, 14, 19,  2,  7,  9, 13, 14, 19,  2,  7,  9, 13, 14,
            19], dtype=int64))

    >>> tout, xout, *_ = rewrap(t, x, method='rewrap')
    >>> plt.figure
    >>> plt.plot(t,x, linewidth='0.5', linestyle=':', color='r')
    >>> plt.scatter(t,x)
    >>> plt.plot(tout, xout)

    """

    # Check the function inputs

    assert method in ['unwrap', 'split', 'rewrap', 'rewrap_nosplit'], f"Incorrect value {method} for method."
    assert np.all(np.diff(t) >= 0),  "Input data (t, x) must be sorted on t."
    assert len(wrapat) ==2 and wrapat[1] > wrapat[0],  "Parameter wrapat has an incorrect value."

    # Set period and modify the data to be in the desired range

    period = wrapat[1] - wrapat[0]
    x =  x - period*np.floor((x - wrapat[0])/period) 
    if verbose:
        print('x:', x)
 
    # Boolean with wrap occurences (bx), integer number of cycles (sx) and cumulative sum of cycles (csx)
    bx = np.abs(np.diff(x)) > period/2
    sx = np.sign(np.diff(x)).astype(np.int64)
    sx[~bx]=0
    bx = np.hstack([ False, bx])
    sx = np.hstack([ 0, -sx])
    csx = np.cumsum(sx)
    if verbose:
        print('#wraps:', np.sum(bx)+1)
        print('bx:', bx)        
        print('sx:', sx)
        print('csx:', csx)

    # Index to wraps
    isx = np.nonzero(sx)
    isw = isx[0]
    if verbose:
        print('isx:', isx)
        print('isw:', isw)

    if method == 'unwrap':
        # Continuous data
        tout = t
        xout = x + csx * period
    elif method == 'split':
        # Insert nan's at the warps 
        tout = np.insert(t, isw, np.nan)
        xout = np.insert(x, isw, np.nan)
    else:
        # Estimate the time of a wrap by interpolation
        icycle = np.round(( x - wrapat[0])/period)
        dx = x - wrapat[0] - icycle*period
        dxa = np.abs(dx)
        tbreak = t[isw-1] + dxa[isw-1] / ( dxa[isw-1] + dxa[isw]  ) * ( t[isw] - t[isw-1] )
        if verbose:
            print('icycle:', icycle)
            print('dx:', dx)
            print('tbreak:', tbreak)
        # Prepare the data to be inserted
        if method == "rewrap":
            ins = np.hstack([isw, isw, isw])
            tins = np.hstack([tbreak, tbreak, tbreak])
            xins = np.hstack([ wrapat[0]+icycle[isw-1]*period, tbreak*np.nan, wrapat[0]+icycle[isw]*period]) 
        elif method == "rewrap_nosplit":
            ins = np.hstack([isw, isw])
            tins = np.hstack([tbreak, tbreak])
            xins = np.hstack([ wrapat[0]+icycle[isw-1]*period, wrapat[0]+icycle[isw]*period]) 
        # Insert the data
        tout = np.insert(t, ins, tins)
        xout = np.insert(x, ins, xins)

    if verbose:
        print('tout:', tout)
        print('xout:', xout)

    return tout, xout, ins

def rewrap2d(x, y, method="rewrap", xwrapat=[-180,180], ywrapat=[-90,90], verbose=0):
    """Rewrap cyclic x and y data for plotting line segments
    
    Re-wrap cyclic data in two dimensions and insert breaks in the lines so that when 
    plotted they are properly terminated at the borders of the 2D wrapping region. This 
    function solves two issues with plotting lines for cyclic data: 
    - the near vertical and horizontal line segments at the wraps, 
    - lines do not extend to the wrapping boundaries (instead they take the opposite 
      direction)
    If unsolved, line plots on cyclic data can be misleading or unclear at best, while
    a scatter plot may not provide all the information. 

    This version corrects for wraps in both the x and y data. The default values are
    choosen to operate typically on longitude (`x`) and latitude (`y`) data for mapping
    purposes.

    Parameters
    ----------
    x, y: array_like, float
        The absissae (x) and ordinates (y), e.g. longitude and latitude
    method: {'rewrap', 'unwrap', split', rewrap_nosplit'}, optional
        The rewrapping method. Defines what action is taken at a data wrap 
            - rewrap : estimate time of wrap by interpolation from the data, and insert 
                   the upper/lower wrap limits and 'np.nan' to cause a split in the line. 
                   The output arrays are larger than the the input arrays (3 additional 
                   data points per wrap)
            - split : split the lines at a wrap by inserting 'np.nan'. The output arrays 
                   are larger than the the input arrays (1 additional data point per wrap)
            - unwrap : make the data continuous. The length of the arrays is unchanged 
    xwrapat, ywrapat: list, optional
        List with the lower and upper wrap boundaries.
    verbose: int, optional
        Vebosity level.

    Returns
    -------
    xout, yout: ndarray, float
        The rewrapped abcissae (x) and rewrapped ordinates (y) for plotting
    ins: ndarray, int64
        Index array for 'np.insert', can be applied to other arrays ('z') of similar shape 
        with 'zout = np.insert(z, ins, new_z_values_to_insert)'

    Notes
    -----
    The ouput should only be used for plotting purposes, as with some options interpolated
    data points and/or `np.nan` are inserted which could bias scientific analysis.

    See also
    --------
    rewrap
        Rewrap cyclic data in one dimension.

    """

    # Check the function inputs

    assert method in ['unwrap', 'split', 'rewrap', 'rewrap_nosplit'], f"Incorrect value {method} for method."
    assert len(xwrapat) ==2 and xwrapat[1] > xwrapat[0],  "Parameter xwrapat has an incorrect value."
    assert len(ywrapat) ==2 and ywrapat[1] > ywrapat[0],  "Parameter ywrapat has an incorrect value."

    # Set period and modify the data to be in the desired range

    xperiod = xwrapat[1] - xwrapat[0]
    yperiod = ywrapat[1] - ywrapat[0]
    x =  x - xperiod*np.floor((x - xwrapat[0])/xperiod) 
    y =  y - yperiod*np.floor((y - ywrapat[0])/yperiod) 
    if verbose:
        print('x:', x)
        print('y:', y)
 
    # Boolean with wrap occurences (bx), integer number of cycles (sx) and cumulative sum of cycles (csx)
    bx = np.abs(np.diff(x)) > xperiod/2
    sx = np.sign(np.diff(x)).astype(np.int64)
    sx[~bx]=0
    bx = np.hstack([ False, bx])
    sx = np.hstack([ 0, -sx])
    csx = np.cumsum(sx)
    if verbose:
        print('#xwraps:', np.sum(bx)+1)
        print('bx:', bx)        
        print('sx:', sx)
        print('csx:', csx)

    # Boolean with wrap occurences (bx), integer number of cycles (sx) and cumulative sum of cycles (csx)
    by = np.abs(np.diff(y)) > yperiod/2
    sy = np.sign(np.diff(y)).astype(np.int64)
    sy[~by]=0
    by = np.hstack([ False, by])
    sy = np.hstack([ 0, -sy])
    csy = np.cumsum(sy)
    if verbose:
        print('#ywraps:', np.sum(by)+1)
        print('by:', by)        
        print('sy:', sy)
        print('csy:', csy)

    # Index to wraps in x 
    isx = np.nonzero(sx)
    iswx = isx[0]
    isy = np.nonzero(sy)
    iswy = isy[0]
    if verbose:
        print('isx:', isx)
        print('iswx:', iswx)
        print('isy:', isy)
        print('iswy:', iswy)

    if method == 'unwrap':
        # Continuous data
        xout = x + csx * xperiod
        yout = y + csy * yperiod
    elif method == 'split':
        # Insert nan's at the warps 
        ins = np.unique(np.hstack([iswx, iswy]))
        xout = np.insert(x, ins, np.nan)
        yout = np.insert(y, ins, np.nan)
    else:
        # Estimate the time of a x wrap by interpolation
        ixcycle = np.round(( x - xwrapat[0])/xperiod)
        dx = x - xwrapat[0] - ixcycle*xperiod
        dxa = np.abs(dx)
        ybreak = y[iswx-1] + dxa[iswx-1] / ( dxa[iswx-1] + dxa[iswx]  ) * ( y[iswx] - y[iswx-1] )
        if verbose:
            print('ixcycle:', ixcycle)
            print('dx:', dx)
            print('ybreak:', ybreak)
        # Estimate the time of a y wrap by interpolation
        iycycle = np.round(( y - ywrapat[0])/yperiod)
        dy = y - ywrapat[0] - iycycle*yperiod
        dya = np.abs(dy)
        xbreak = x[iswy-1] + dya[iswy-1] / ( dya[iswy-1] + dya[iswy]  ) * ( x[iswy] - x[iswy-1] )
        if verbose:
            print('iycycle:', iycycle)
            print('dy:', dy)
            print('xbreak:', xbreak)
        # Prepare the data to be inserted
        ins = np.hstack([iswx, iswx, iswx, iswy, iswy, iswy])
        xins = np.hstack([ xwrapat[0]+ixcycle[iswx-1]*xperiod, ybreak*np.nan, xwrapat[0]+ixcycle[iswx]*xperiod, xbreak, xbreak, xbreak])
        yins = np.hstack([ ybreak, ybreak, ybreak, ywrapat[0]+iycycle[iswy-1]*yperiod, xbreak*np.nan, ywrapat[0]+iycycle[iswy]*yperiod]) 
        
        # Insert the data
        xout = np.insert(x, ins, xins)
        yout = np.insert(y, ins, yins)

    return xout, yout, ins
    

    

def xlabels(t):
    """Prepare x-axis labels and title string entries"""
    
    kmid=int(t.shape[0]/2)
    if ( t[-1] - t[0] ) < 1/120:
       xtimefmt = '%H:%M:%S'
       xlabelfmt = '%Y-%m-%d'
    elif ( t[-1] - t[0] ) < 3: 
       xtimefmt = '%H:%M'
       xlabelfmt = '%Y-%m-%d'
    elif ( t[-1] - t[0] ) < 6: 
       xtimefmt = '%m-%d %Hh'
       xlabelfmt = '%Y-%m-%d'
    else: 
       xtimefmt = '%m-%d'
       xlabelfmt = '%Y'

    xlabeldate = num2datetime(t[kmid]).strftime(xlabelfmt)
    xlabelrange = "{} - {}".format(num2datetime(t[0]).isoformat(timespec="minutes"), num2datetime(t[-1]).isoformat(timespec="minutes"))
	
    return xtimefmt, xlabeldate, xlabelrange

# ----------------------------------------------------------------------------
# SUPPORT FUNCTIONS
#
# - plh2ecef(objcrd, unit='rad') -> xobje        
#       Convert ECEF latitude, longitude and height to Cartesian coordinates
# - plh2eci(t, objcrd, unit='rad') -> xobj, vobj
#       Convert ECEF latitude, longitude and height to ECI position and velocity
# - xyz2zassp(xsat, xobj) -> zen, azi, robj2sat
#       Compute zenith angle, azimuth and distance
# - obj2sat(xsat, vsat, xobj, vobj) -> zen, azi, robj2sat, rrobj2sat
#       Compute zenith angle, azimuth, range and range-rate
# - xyz2radec(xyz, unit='rad') -> rasc, decl
#       Compute right ascension and declination
# - ra2lon(t, ra, unit='rad') -> lon
#      Convert right ascension in ECI to longitude in ECEF
#
# ----------------------------------------------------------------------------

def plh2ecef(objcrd, unit='rad'):        
    """Convert ECEF latitude, longitude and height to Cartesian coordinates."""

    Re = 6378136          # Radius of the Earth

    assert unit in ['rad', 'deg'], f"Incorrect unit {unit}."

    # The position of the observer (latitude, longitude and height) is given in ECEF
    # using the input array objcrd

    if unit == 'deg':
        lat = objcrd[0]*np.pi/180
        lon = objcrd[1]*np.pi/180
    else:
        lat = objcrd[0]
        lon = objcrd[1]

    Rs = Re + objcrd[2]
	
    # Position of the observer in ECEF (assume latitude and longitude are for spherical Earth)

    xobje = [Rs*np.cos(lat)*np.cos(lon),
             Rs*np.cos(lat)*np.sin(lon),
             Rs*np.sin(lat)]

    return xobje

def ecef2plh(xobje, unit='rad'):        
    """Convert Cartesian coordinates in ECEF to latitude, longitude and height."""

    Re = 6378136           # Radius of the Earth

    assert unit in ['rad', 'deg'], f"Incorrect unit {unit}."
	
    # Position of the observer in ECEF (assume latitude and longitude are for spherical Earth)

    xobje = np.asarray(xobje)

    objcrd = np.empty_line(xobje)
    objcrd[...,1] = np.arctan2(xobje[..., 1], xobje[..., 0])
    objcrd[...,0] = np.arctan(xobje[..., 2]/np.sqrt(xobje[..., 1]**2+xobje[..., 0]**2))
    objcrd[...,3] = np.sqrt(xobje[..., 0]**2 + xobje[..., 1]**2 + xobje[..., 2]**2) - Re
    if unit == 'deg':
        objcrd[...,0] = objcrd[...,0] * 180/np.pi
        objcrd[...,1] = objcrd[...,1] * 180/np.pi

    return objcrd

def plh2eci(t, objcrd, unit='rad'):        
    """Convert ECEF latitude, longitude and height to ECI position and velocity."""

    # Constants

    Re = 6378136          # Radius of the Earth
    Me = 7.2921151467e-5  # rad/s , rotational velocity of Earth

    # The position of the observer (latitude, longitude and height) is given in ECEF
    # using the input array objcrd

    if unit == 'deg':
        lat = objcrd[0]*np.pi/180
        lon = objcrd[1]*np.pi/180
    else:
        lat = objcrd[0]
        lon = objcrd[1]

    Rs = Re + objcrd[2]
	    
    # The transformation from an ECEF to ECI is a simple rotation around the z-axis
    # a. the rotation angle is GMST (Greenwhich Mean Stellar Time) 
    # b. the rotation around the z-axis can be implemented by replacing the
    #    longitude (in ECEF) by local stellar time (lst) in the ECI
    #
    # The times given in t are in UTC, which is close to UT1 (max 0.9 s difference),
    # which is not important for a plotting application

    # Compute GMST from UT1, for the first epoch in t, using the Matlab function 
    # ut2gmst. The second output returned by ut2gmst is the rotational velocity
    # omegae of the Earth in rev/day

    gst0, omegae = ut2gmst(t[0])

    # Compute local stellar time (in radians) from the longitude, GMST at the initial
    # epoch and the rotational velocity of the Earth (times elapsed time). Note that
    # lst is an array, while lon is a scalar)

    lst = lon+gst0+2*np.pi*omegae*(t-t[0])

    # Compute position and velocity of the observer in ECI using lst (position and 
    # velocity in an ECI change all the time, unlike in a ECEF)

    nepoch = t.shape[0]

    xobj = np.zeros((nepoch, 3))   # pre-allocate memory, makes it run faster
    vobj = np.zeros((nepoch, 3))

    xobj[:, 0] = Rs*np.cos(lat)*np.cos(lst)
    xobj[:, 1] = Rs*np.cos(lat)*np.sin(lst)
    xobj[:, 2] = Rs*np.sin(lat)
    vobj[:, 0] = -Rs*np.cos(lat)*Me*np.sin(lst)
    vobj[:, 1] = Rs*np.cos(lat)*Me*np.cos(lst)
    
    return xobj, vobj

def xyz2zassp(xsat, xobj):
    """Compute zenith angle, azimuth and distance."""

    xobj2sat = xsat-xobj
    robj2sat = np.sqrt(np.sum(xobj2sat**2, axis=1))

    # normal vector (vertical) and unit direction vector to satellite from observer

    robj = np.sqrt(np.sum(xobj**2, axis=1))
    n0 = xobj / robj[:, np.newaxis]
    ers = xobj2sat / robj2sat[:, np.newaxis]
	
    # zenith angle and azimuth of satellite (as seen from object wrt to radial direction)

    ip = np.sum(n0 * ers, axis=1)
    zen = np.arccos(ip)
    azi = np.arctan2(-n0[:, 1]*ers[:, 0] + n0[:, 0]*ers[:, 1], ip*-1*n0[:, 2] + ers[:, 2])
    azi += 2*np.pi
    azi %= 2*np.pi
    
    return zen, azi, robj2sat

def obj2sat(xsat, vsat, xobj, vobj):
    """Compute zenith angle, azimuth, range and range-rate."""

    xobj2sat = xsat-xobj
    vobj2sat = vsat-vobj
    robj2sat = np.sqrt(np.sum(xobj2sat**2, axis=1))
    rrobj2sat = np.sum(vobj2sat * xobj2sat/robj2sat[:, np.newaxis], axis=1)

    # Note that the range rate rrobj2sat is not the same as the relative velocity
    # sqrt(sum(vobj2sat.^2,2)), these are different things
    
    # normal vector (vertical) and unit direction vector to satellite from observer

    robj = np.sqrt(np.sum(xobj**2, axis=1))
    n0 = xobj / robj[:, np.newaxis]
    ers = xobj2sat / robj2sat[:, np.newaxis]
	
    # zenith angle and azimuth of satellite (as seen from object wrt to radial direction)

    ip = np.sum(n0 * ers, axis=1)
    zen = np.arccos(ip)
    azi = np.arctan2(-n0[:, 1]*ers[:, 0] + n0[:, 0]*ers[:, 1], ip*-1*n0[:, 2] + ers[:, 2])
    azi += 2*np.pi
    azi %= 2*np.pi
    
    return zen, azi, robj2sat, rrobj2sat
    
def xyz2radec(xyz, unit='rad'):
    """Compute right ascension (ra) and declination (dec)"""
    
    assert unit in ['rad', 'deg'], f"Incorrect unit {unit}."
    
    ra = np.arctan2(xyz[:, 1], xyz[:, 0])
    dec = np.arctan(xyz[:, 2]/np.sqrt(xyz[:, 1]**2+xyz[:, 0]**2))
    if unit == 'deg':
        ra = np.degrees(ra)
        dec = np.degrees(dec)
    
    return ra, dec

def ra2lon(t, ra, unit='rad'):
    """Convert right ascension in ECI to longitude in ECEF"""
    
    # Substract GMST from right-ascension in ECI to obtain longitude in ECEF

    gst0, omegae = ut2gmst(t[0])
    if unit == 'deg':
        gst = gst0*180/np.pi+360*omegae*(t-t[0])
        lon = ra-gst-360*np.round((ra-gst)/360., 0)   # must be in the range [-180,+180]
    else:
        gst = gst0 +2*omegae*(t-t[0])/np.pi
        lon = ra-gst-2*np.pi*np.round((ra-gst)/(2*np.pi), 0)   # must be in the range [-pi,+pi]
        
    return lon
