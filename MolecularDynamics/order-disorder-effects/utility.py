import matplotlib as mpl
import matplotlib.cm     as cm
from matplotlib import pyplot as plt
import numpy as np


mpl.rcParams['axes.linewidth'] = 2.0 #set the value globally
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['lines.linestyle'] = 'dashed'
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 15
mycmap = cm.plasma
cmap2 = cm.Blues




def add_multiaxes(fig, left, bottom, width, height, hist_width, spacing):
    rect_scatter = [left, bottom, width, height]
    rect_histy = [left + width + spacing, bottom, hist_width, height]
    ax = fig.add_axes(rect_scatter)
    ax_histy = fig.add_axes(rect_histy,  projection='3d')
    ax_histy.get_xaxis().set_visible(False)
    ax_histy.get_yaxis().set_visible(False)
    ax_histy.view_init(elev=30, azim=0, vertical_axis='z')
    return ax, ax_histy

def plot_sphere(ax):
    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)


def cart2sph(x, y, z):
    dxy = np.sqrt(x**2 + y**2)
    r = np.sqrt(dxy**2 + z**2)
    theta =  np.arctan2(y, x) 
    phi = np.arctan2(z, dxy)
    theta = np.rad2deg(theta)
    phi = np.rad2deg(phi)
    # theta, phi = np.rad2deg([theta, phi])
    return theta % 360, phi, r

def sph2cart(theta, phi, r=1):
    theta, phi = np.deg2rad([theta, phi])
    z = r * np.sin(phi)
    rcosphi = r * np.cos(phi)
    x = rcosphi * np.cos(theta)
    y = rcosphi * np.sin(theta)
    return x, y, z


def near( p, pntList, d0 ):
    cnt=0
    for pj in pntList:
        dist=np.linalg.norm( p - pj )
        if dist < d0:
            cnt += 1 - dist/d0
    return cnt

def sphere_quiver(ax, pntList, nbins=100):
    u = pntList[: ,0]
    v = pntList[: ,1]
    w = pntList[: ,2]
 
    
    ax.quiver(0*u, 0*u, 0*u, u,v,w, length=0.025,normalize=True )
    return

def sphere_heatmap(ax, pntList  ):
    nbins=120
    u = np.linspace( 0, 2 * np.pi, nbins)
    v = np.linspace( 0, np.pi, nbins//2 )
    # create the sphere surface
    XX =  np.outer( np.cos( u ), np.sin( v ) )
    YY =  np.outer( np.sin( u ), np.sin( v ) )
    ZZ =  np.outer( np.ones( np.size( u ) ), np.cos( v ) )

    WW = XX.copy()
    for i in range( len( XX ) ):
        for j in range( len( XX[0] ) ):
            p = np.array([XX[ i, j ], YY[ i, j ], ZZ[ i, j ]])
            angle = np.arccos(  (pntList*p).sum(-1) )
            WW[ i, j ] = near(np.array( [x, y, z ] ), pointList, 3)
    WW = WW / np.amax( WW )
    myheatmap = WW

def sphere_histogram(ax, mycmap, pntList, nbins):
    theta, phi, r = cart2sph(pntList[:,0], pntList[:,1], pntList[:,2])
    hist2d, u, v = np.histogram2d(theta, phi, bins=[180,90], range=[[0, 360], [-90, 90]] )
    u =  u[:-1] 
    v =  v[:-1] 
    u =    u / 180 * np.pi
    v =  - v / 180 * np.pi
    U, V = np.meshgrid(u, v)
    print((theta ).max(), (theta ).min())
    print((phi ).max(), (phi ).min())
    # create the sphere surface
    XX = np.cos(U) * np.cos(V)
    YY = np.sin(U) * np.cos(V)
    ZZ = np.sin(V)

    # XX =  np.outer( np.cos( u ), np.cos( v ) )
    # YY =  np.outer( np.sin( u ), np.cos( v ) )
    # ZZ =  np.outer( np.ones( np.size( u ) ), np.sin( v ) )
    ## WW: density
    WW =  hist2d.T  #/ np.cos(V)
    WW = WW  / WW.max()
    ax.plot_surface( XX, YY,  ZZ, cstride=1, rstride=1,  facecolors=mycmap( WW ) )
    return WW
