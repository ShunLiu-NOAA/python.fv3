import pygrib
import matplotlib
matplotlib.use('Agg')
import io
import matplotlib.pyplot as plt
#from PIL import Image
import matplotlib.image as image
from matplotlib.gridspec import GridSpec
import mpl_toolkits
mpl_toolkits.__path__.append('/gpfs/dell2/emc/modeling/noscrub/gwv/py/lib/python/basemap-1.2.1-py3.6-linux-x86_64.egg/mpl_toolkits/')
from mpl_toolkits.basemap import Basemap, maskoceans
import numpy as np
import time,os,sys,multiprocessing
import multiprocessing.pool
import ncepy
from scipy import ndimage
from netCDF4 import Dataset
import pyproj

#--------------Set some classes------------------------#
# Make Python process pools non-daemonic
class NoDaemonProcess(multiprocessing.Process):
  # make 'daemon' attribute always return False
  def _get_daemon(self):
    return False
  def _set_daemon(self, value):
    pass
  daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
  Process = NoDaemonProcess


#--------------Define some functions ------------------#

def clear_plotables(ax,keep_ax_lst,fig):
  #### - step to clear off old plottables but leave the map info - ####
  if len(keep_ax_lst) == 0 :
    print("clear_plotables WARNING keep_ax_lst has length 0. Clearing ALL plottables including map info!")
  cur_ax_children = ax.get_children()[:]
  if len(cur_ax_children) > 0:
    for a in cur_ax_children:
      if a not in keep_ax_lst:
       # if the artist isn't part of the initial set up, remove it
        a.remove()

def compress_and_save(filename):
  #### - compress and save the image - ####
#  ram = io.StringIO()
#  ram = io.BytesIO()
#  plt.savefig(ram, format='png', bbox_inches='tight', dpi=150)
  plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
#  ram.seek(0)
#  im = Image.open(ram)
#  im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
#  im2.save(filename, format='PNG')

def cmap_t2m():
 # Create colormap for 2-m temperature
 # Modified version of the ncl_t2m colormap from Jacob's ncepy code
    r=np.array([255,128,0,  70, 51, 0,  255,0, 0,  51, 255,255,255,255,255,171,128,128,36,162,255])
    g=np.array([0,  0,  0,  70, 102,162,255,92,128,185,255,214,153,102,0,  0,  0,  68, 36,162,255])
    b=np.array([255,128,128,255,255,255,255,0, 0,  102,0,  112,0,  0,  0,  56, 0,  68, 36,162,255])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=np.float(i)/(np.float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_t2m_coltbl = matplotlib.colors.LinearSegmentedColormap('CMAP_T2M_COLTBL',colorDict)
    return cmap_t2m_coltbl


def cmap_t850():
 # Create colormap for 850-mb equivalent potential temperature
    r=np.array([255,128,0,  70, 51, 0,  0,  0, 51, 255,255,255,255,255,171,128,128,96,201])
    g=np.array([0,  0,  0,  70, 102,162,225,92,153,255,214,153,102,0,  0,  0,  68, 96,201])
    b=np.array([255,128,128,255,255,255,162,0, 102,0,  112,0,  0,  0,  56, 0,  68, 96,201])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=np.float(i)/(np.float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_t850_coltbl = matplotlib.colors.LinearSegmentedColormap('CMAP_T850_COLTBL',colorDict)
    return cmap_t850_coltbl


def cmap_terra():
 # Create colormap for terrain height
 # Emerald green to light green to tan to gold to dark red to brown to light brown to white
    r=np.array([0,  152,212,188,127,119,186])
    g=np.array([128,201,208,148,34, 83, 186])
    b=np.array([64, 152,140,0,  34, 64, 186])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=np.float(i)/(np.float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_terra_coltbl = matplotlib.colors.LinearSegmentedColormap('CMAP_TERRA_COLTBL',colorDict)
    cmap_terra_coltbl.set_over(color='#E0EEE0')
    return cmap_terra_coltbl


def extrema(mat,mode='wrap',window=100):
    # find the indices of local extrema (max only) in the input array.
    mx = ndimage.filters.maximum_filter(mat,size=window,mode=mode)
    # (mat == mx) true if pixel is equal to the local max
    return np.nonzero(mat == mx)

#-------------------------------------------------------#

# Necessary to generate figs when not running an Xserver (e.g. via PBS)
# plt.switch_backend('agg')

# Read date/time and forecast hour from command line
ymdh = str(sys.argv[1])
ymd = ymdh[0:8]
year = int(ymdh[0:4])
month = int(ymdh[4:6])
day = int(ymdh[6:8])
hour = int(ymdh[8:10])
cyc = str(hour).zfill(2)
print(year, month, day, hour)

fhr = int(sys.argv[2])
fhrm1 = fhr - 1
fhrm2 = fhr - 2
fhrm6 = fhr - 6
fhrm24 = fhr - 24
fhour = str(fhr).zfill(2)
fhour1 = str(fhrm1).zfill(2)
fhour2 = str(fhrm2).zfill(2)
fhour6 = str(fhrm6).zfill(2)
fhour24 = str(fhrm24).zfill(2)
print('fhour '+fhour)

# Define the input files
data1 = pygrib.open('/gpfs/dell6/ptmp/emc.campara/fv3lamx/fv3lamx.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f'+fhour+'.grib2')
data2 = pygrib.open('/gpfs/dell6/ptmp/emc.campara/fv3lamx_gfsv16/fv3lamx.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f'+fhour+'.grib2')

if (fhr > 2):
  data1_m1 = pygrib.open('/gpfs/dell6/ptmp/emc.campara/fv3lamx/fv3lamx.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f'+fhour1+'.grib2')
  data2_m1 = pygrib.open('/gpfs/dell6/ptmp/emc.campara/fv3lamx_gfsv16/fv3lamx.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f'+fhour1+'.grib2')
  data1_m2 = pygrib.open('/gpfs/dell6/ptmp/emc.campara/fv3lamx/fv3lamx.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f'+fhour2+'.grib2')
  data2_m2 = pygrib.open('/gpfs/dell6/ptmp/emc.campara/fv3lamx_gfsv16/fv3lamx.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f'+fhour2+'.grib2')
if (fhr >= 6):
  data1_m6 = pygrib.open('/gpfs/dell6/ptmp/emc.campara/fv3lamx/fv3lamx.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f'+fhour6+'.grib2')
  data2_m6 = pygrib.open('/gpfs/dell6/ptmp/emc.campara/fv3lamx_gfsv16/fv3lamx.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f'+fhour6+'.grib2')
if (fhr >= 24):
  data1_m24 = pygrib.open('/gpfs/dell6/ptmp/emc.campara/fv3lamx/fv3lamx.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f'+fhour24+'.grib2')
  data2_m24 = pygrib.open('/gpfs/dell6/ptmp/emc.campara/fv3lamx_gfsv16/fv3lamx.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f'+fhour24+'.grib2')

# Get the lats and lons
grids = [data1, data2]
lats = []
lons = []
lats_shift = []
lons_shift = []

for data in grids:
    # Unshifted grid for contours and wind barbs
    lat, lon = data[1].latlons()
    lats.append(lat)
    lons.append(lon)

    # Shift grid for pcolormesh
    lat1 = data[1]['latitudeOfFirstGridPointInDegrees']
    lon1 = data[1]['longitudeOfFirstGridPointInDegrees']
    try:
        nx = data[1]['Nx']
        ny = data[1]['Ny']
    except:
        nx = data[1]['Ni']
        ny = data[1]['Nj']
    dx = data[1]['DxInMetres']
    dy = data[1]['DyInMetres']
    pj = pyproj.Proj(data[1].projparams)
    llcrnrx, llcrnry = pj(lon1,lat1)
    llcrnrx = llcrnrx - (dx/2.)
    llcrnry = llcrnry - (dy/2.)
    x = llcrnrx + dx*np.arange(nx)
    y = llcrnry + dy*np.arange(ny)
    x,y = np.meshgrid(x,y)
    lon, lat = pj(x, y, inverse=True)
    lats_shift.append(lat)
    lons_shift.append(lon)

# Unshifted lat/lon arrays grabbed directly using latlons() method
lat = lats[0]
lon = lons[0]
lat2 = lats[1]
lon2 = lons[1]

# Shifted lat/lon arrays for pcolormesh
lat_shift = lats_shift[0]
lon_shift = lons_shift[0]
lat2_shift = lats_shift[1]
lon2_shift = lons_shift[1]

Lat0 = data1[1]['LaDInDegrees']
Lon0 = data1[1]['LoVInDegrees']
#Lon0 = 262.5
print(Lat0)
print(Lon0)

# Forecast valid date/time
itime = ymdh
vtime = ncepy.ndate(itime,int(fhr))

# Specify plotting domains
#domains = ['conus','BN','CE','CO','LA','MA','NC','NE','NW','OV','SC','SE','SF','SP','SW','UM']
domains=['conus','CO']

###################################################
# Read in all variables and calculate differences #
###################################################
t1a = time.clock()

# Sea level pressure
slp_1 = data1.select(name='Pressure reduced to MSL')[0].values * 0.01
slpsmooth1 = ndimage.filters.gaussian_filter(slp_1, 13.78)
slp_2 = data2.select(name='Pressure reduced to MSL')[0].values * 0.01
slpsmooth2 = ndimage.filters.gaussian_filter(slp_2, 13.78)
slp_dif = slp_2 - slp_1

# 2-m temperature
tmp2m_1 = data1.select(name='2 metre temperature')[0].values
tmp2m_1 = (tmp2m_1 - 273.15)*1.8 + 32.0
tmp2m_2 = data2.select(name='2 metre temperature')[0].values
tmp2m_2 = (tmp2m_2 - 273.15)*1.8 + 32.0
tmp2m_dif = tmp2m_2 - tmp2m_1

# Surface temperature
tmpsfc_1 = data1.select(name='Temperature',typeOfLevel='surface')[0].values
tmpsfc_1 = (tmpsfc_1 - 273.15)*1.8 + 32.0
tmpsfc_2 = data2.select(name='Temperature',typeOfLevel='surface')[0].values
tmpsfc_2 = (tmpsfc_2 - 273.15)*1.8 + 32.0
tmpsfc_dif = tmpsfc_2 - tmpsfc_1

# 2-m dew point temperature
dew2m_1 = data1.select(name='2 metre dewpoint temperature')[0].values
dew2m_1 = (dew2m_1 - 273.15)*1.8 + 32.0
dew2m_2 = data2.select(name='2 metre dewpoint temperature')[0].values
dew2m_2 = (dew2m_2 - 273.15)*1.8 + 32.0
dew2m_dif = dew2m_2 - dew2m_1

# 850-mb equivalent potential temperature
t850_1 = data1.select(name='Temperature',level=850)[0].values
dpt850_1 = data1.select(name='Dew point temperature',level=850)[0].values
q850_1 = data1.select(name='Specific humidity',level=850)[0].values
tlcl_1 = 56.0 + (1.0/((1.0/(dpt850_1-56.0)) + 0.00125*np.log(t850_1/dpt850_1)))
thetae_1 = t850_1*((1000.0/850.0)**(0.2854*(1.0-(0.28*q850_1))))*np.exp(((3376.0/tlcl_1)-2.54)*q850_1*(1.0+(0.81*q850_1)))
t850_2 = data2.select(name='Temperature',level=850)[0].values
dpt850_2 = data2.select(name='Dew point temperature',level=850)[0].values
q850_2 = data2.select(name='Specific humidity',level=850)[0].values
tlcl_2 = 56.0 + (1.0/((1.0/(dpt850_2-56.0)) + 0.00125*np.log(t850_2/dpt850_2)))
thetae_2 = t850_2*((1000.0/850.0)**(0.2854*(1.0-(0.28*q850_2))))*np.exp(((3376.0/tlcl_2)-2.54)*q850_2*(1.0+(0.81*q850_2)))
thetae_dif = thetae_2 - thetae_1

# 850-mb winds
u850_1 = data1.select(name='U component of wind',level=850)[0].values * 1.94384
u850_2 = data2.select(name='U component of wind',level=850)[0].values * 1.94384
v850_1 = data1.select(name='V component of wind',level=850)[0].values * 1.94384
v850_2 = data2.select(name='V component of wind',level=850)[0].values * 1.94384
# Rotate winds from grid relative to Earth relative
#u850_1, v850_1 = ncepy.rotate_wind(Lat0,Lon0,lon,u850_1,v850_1,'lcc',inverse=False)
u850_2, v850_2 = ncepy.rotate_wind(Lat0,Lon0,lon2,u850_2,v850_2,'lcc',inverse=False)

# 500 mb height, wind, vorticity
z500_1 = data1.select(name='Geopotential Height',level=500)[0].values * 0.1
z500_1 = ndimage.filters.gaussian_filter(z500_1, 6.89)
z500_2 = data2.select(name='Geopotential Height',level=500)[0].values * 0.1
z500_2 = ndimage.filters.gaussian_filter(z500_2, 6.89)
z500_dif = z500_2 - z500_1
vort500_1 = data1.select(name='Absolute vorticity',level=500)[0].values * 100000
vort500_1 = ndimage.filters.gaussian_filter(vort500_1,1.7225)
vort500_1[vort500_1 > 1000] = 0	# Mask out undefined values on domain edge
vort500_2 = data2.select(name='Absolute vorticity',level=500)[0].values * 100000
vort500_2 = ndimage.filters.gaussian_filter(vort500_2,1.7225)
vort500_2[vort500_2 > 1000] = 0	# Mask out undefined values on domain edge
u500_1 = data1.select(name='U component of wind',level=500)[0].values * 1.94384
u500_2 = data2.select(name='U component of wind',level=500)[0].values * 1.94384
v500_1 = data1.select(name='V component of wind',level=500)[0].values * 1.94384
v500_2 = data2.select(name='V component of wind',level=500)[0].values * 1.94384
# Rotate winds from grid relative to Earth relative
#u500_1, v500_1 = ncepy.rotate_wind(Lat0,Lon0,lon,u500_1,v500_1,'lcc',inverse=False)
u500_2, v500_2 = ncepy.rotate_wind(Lat0,Lon0,lon2,u500_2,v500_2,'lcc',inverse=False)

# Precipitable water
pw_1 = data1.select(name='Precipitable water',level=0)[0].values * 0.0393701
pw_2 = data2.select(name='Precipitable water',level=0)[0].values * 0.0393701
pw_dif = pw_2 - pw_1

# Percent of frozen precipitation
pofp_1 = data1.select(name='Percent frozen precipitation')[0].values
pofp_2 = data2.select(name='Percent frozen precipitation')[0].values
pofp_dif = pofp_2 - pofp_1

# Total precipitation
qpf_1 = data1.select(name='Total Precipitation',lengthOfTimeRange=fhr)[0].values * 0.0393701
qpf_2 = data2.select(name='Total Precipitation',lengthOfTimeRange=fhr)[0].values * 0.0393701
qpf_dif = qpf_2 - qpf_1

# 3-hr precipitation
if (fhr > 2):  # Do not make 3-hr plots for forecast hours 1 and 2
  qpfm2_1 = data1_m2.select(name='Total Precipitation',lengthOfTimeRange=1)[0].values * 0.0393701
  qpfm1_1 = data1_m1.select(name='Total Precipitation',lengthOfTimeRange=1)[0].values * 0.0393701
  qpfm0_1 = data1.select(name='Total Precipitation',lengthOfTimeRange=1)[0].values * 0.0393701
  qpf3_1 = qpfm2_1 + qpfm1_1 + qpfm0_1
  qpfm2_2 = data2_m2.select(name='Total Precipitation',lengthOfTimeRange=1)[0].values * 0.0393701
  qpfm1_2 = data2_m1.select(name='Total Precipitation',lengthOfTimeRange=1)[0].values * 0.0393701
  qpfm0_2 = data2.select(name='Total Precipitation',lengthOfTimeRange=1)[0].values * 0.0393701
  qpf3_2 = qpfm2_2 + qpfm1_2 + qpfm0_2
  qpf3_dif = qpf3_2 - qpf3_1

# Snow depth
snow_1 = data1.select(name='Snow depth')[0].values * 39.3701
snow_2 = data2.select(name='Snow depth')[0].values * 39.3701
snow_dif = snow_2 - snow_1
if (fhr >=6):	# Do not make 6-hr plots for forecast hours less than 6
  snowm6_1 = data1_m6.select(name='Snow depth')[0].values * 39.3701
  snow6_1 = snow_1 - snowm6_1 
  snowm6_2 = data2_m6.select(name='Snow depth')[0].values * 39.3701
  snow6_2 = snow_2 - snowm6_2
  snow6_dif = snow6_2 - snow6_1

# Echo top height
retop_1 = data1.select(parameterName="197",stepType='instant',nameOfFirstFixedSurface='200')[0].values * (3.28084/1000)
retop_2 = data2.select(parameterName="197",stepType='instant',nameOfFirstFixedSurface='200')[0].values * (3.28084/1000)
retop_dif = retop_2 - retop_1

# Precipitation rate
prate_1 = data1.select(name='Precipitation rate')[0].values * 3600
prate_2 = data2.select(name='Precipitation rate')[0].values * 3600
prate_dif = prate_2 - prate_1


t2a = time.clock()
t3a = round(t2a-t1a, 3)
print(("%.3f seconds to read all messages") % t3a)

# colors for difference plots, only need to define once
difcolors = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','white','white','#EEEE00','#EEC900','darkorange','orangered','red']
difcolors2 = ['white']
difcolors3 = ['blue','dodgerblue','turquoise','white','white','#EEEE00','darkorange','red']

########################################
#    START PLOTTING FOR EACH DOMAIN    #
########################################

def main():

  # Number of processes must coincide with the number of domains to plot
#  pool = multiprocessing.Pool(len(domains))
  pool = MyPool(len(domains))
  pool.map(plot_all,domains)

def plot_all(domain):

  global dom
  dom = domain
  print(('Working on '+dom))

  global fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,m,x,y,x2,y2,x_shift,y_shift,x2_shift,y2_shift,xscale,yscale,im,par
  fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,m,x,y,x2,y2,x_shift,y_shift,x2_shift,y2_shift,xscale,yscale,im,par = create_figure()

  # Split plots into 3 sets with multiprocessing
  sets = [1]
  pool2 = multiprocessing.Pool(len(sets))
  pool2.map(plot_sets,sets)

def create_figure():

  # create figure and axes instances
  fig = plt.figure()
  gs = GridSpec(9,9,wspace=0.0,hspace=0.0)
  ax1 = fig.add_subplot(gs[0:4,0:4])
  ax2 = fig.add_subplot(gs[0:4,5:])
  ax3 = fig.add_subplot(gs[5:,1:8])
  axes = [ax1, ax2, ax3]
  im = image.imread('/gpfs/dell2/emc/modeling/noscrub/Benjamin.Blake/python.fv3/noaa.png')
  par = 1

  # Map corners for each domain
  if dom == 'conus':
    llcrnrlon = -120.5
    llcrnrlat = 21.0 
    urcrnrlon = -64.5
    urcrnrlat = 49.0
    lat_0 = 35.4
    lon_0 = -97.6
    xscale=0.15
    yscale=0.2
  elif dom == 'BN':
    llcrnrlon = -75.75
    llcrnrlat = 40.0
    urcrnrlon = -69.5
    urcrnrlat = 43.0
    lat_0 = 41.0
    lon_0 = -74.6
    xscale=0.14
    yscale=0.19
  elif dom == 'CE':
    llcrnrlon = -103.0
    llcrnrlat = 32.5
    urcrnrlon = -88.5
    urcrnrlat = 41.5
    lat_0 = 35.0
    lon_0 = -97.0
    xscale=0.15
    yscale=0.18
  elif dom == 'CO':
    llcrnrlon = -110.5
    llcrnrlat = 35.0
    urcrnrlon = -100.5
    urcrnrlat = 42.0
    lat_0 = 38.0
    lon_0 = -105.0
    xscale=0.17
    yscale=0.18
  elif dom == 'LA':
    llcrnrlon = -121.0
    llcrnrlat = 32.0
    urcrnrlon = -114.0
    urcrnrlat = 37.0
    lat_0 = 34.0
    lon_0 = -114.0
    xscale=0.16
    yscale=0.18
  elif dom == 'MA':
    llcrnrlon = -82.0
    llcrnrlat = 36.5
    urcrnrlon = -73.5
    urcrnrlat = 42.0
    lat_0 = 37.5
    lon_0 = -80.0
    xscale=0.18
    yscale=0.18
  elif dom == 'NC':
    llcrnrlon = -111.0
    llcrnrlat = 39.0
    urcrnrlon = -93.5
    urcrnrlat = 49.0
    lat_0 = 44.5
    lon_0 = -102.0
    xscale=0.16
    yscale=0.18
  elif dom == 'NE':
    llcrnrlon = -80.0     
    llcrnrlat = 40.5
    urcrnrlon = -66.0
    urcrnrlat = 47.5
    lat_0 = 42.0
    lon_0 = -80.0
    xscale=0.16
    yscale=0.18
  elif dom == 'NW':
    llcrnrlon = -125.5     
    llcrnrlat = 40.5
    urcrnrlon = -109.0
    urcrnrlat = 49.5
    lat_0 = 44.0
    lon_0 = -116.0
    xscale=0.15
    yscale=0.18
  elif dom == 'OV':
    llcrnrlon = -91.5 
    llcrnrlat = 34.75
    urcrnrlon = -80.0
    urcrnrlat = 43.0
    lat_0 = 38.0
    lon_0 = -87.0          
    xscale=0.18
    yscale=0.17
  elif dom == 'SC':
    llcrnrlon = -108.0 
    llcrnrlat = 25.0
    urcrnrlon = -88.0
    urcrnrlat = 37.0
    lat_0 = 32.0
    lon_0 = -98.0      
    xscale=0.14
    yscale=0.18
  elif dom == 'SE':
    llcrnrlon = -91.5 
    llcrnrlat = 24.0
    urcrnrlon = -74.0
    urcrnrlat = 36.5
    lat_0 = 34.0
    lon_0 = -85.0
    xscale=0.17
    yscale=0.18
  elif dom == 'SF':
    llcrnrlon = -123.25 
    llcrnrlat = 37.25
    urcrnrlon = -121.25
    urcrnrlat = 38.5
    lat_0 = 37.5
    lon_0 = -121.0
    xscale=0.16
    yscale=0.19
  elif dom == 'SP':
    llcrnrlon = -125.0
    llcrnrlat = 45.0
    urcrnrlon = -119.5
    urcrnrlat = 49.2
    lat_0 = 46.0
    lon_0 = -120.0
    xscale=0.19
    yscale=0.18
  elif dom == 'SW':
    llcrnrlon = -125.0 
    llcrnrlat = 30.0
    urcrnrlon = -108.0
    urcrnrlat = 42.5
    lat_0 = 37.0
    lon_0 = -113.0
    xscale=0.17
    yscale=0.18
  elif dom == 'UM':
    llcrnrlon = -96.75 
    llcrnrlat = 39.75
    urcrnrlon = -81.0
    urcrnrlat = 49.0
    lat_0 = 44.0
    lon_0 = -91.5
    xscale=0.18
    yscale=0.18

  # Create basemap instance and set the dimensions
  for ax in axes:
    if dom == 'BN' or dom == 'LA' or dom == 'SF' or dom == 'SP':
      m = Basemap(ax=ax,projection='gnom',lat_0=lat_0,lon_0=lon_0,\
                  llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                  llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                  resolution='h')
    elif dom == 'conus':
      m = Basemap(ax=ax,projection='gnom',lat_0=lat_0,lon_0=lon_0,\
                  llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                  llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                  resolution='i')
    else:
      m = Basemap(ax=ax,projection='gnom',lat_0=lat_0,lon_0=lon_0,\
                  llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                  llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                  resolution='l')
    m.fillcontinents(color='LightGrey',zorder=0)
    m.drawcoastlines(linewidth=0.75)
    m.drawstates(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
##  parallels = np.arange(0.,90.,10.)
##  map.drawparallels(parallels,labels=[1,0,0,0],fontsize=6)
##  meridians = np.arange(180.,360.,10.)
##  map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=6)
    x,y = m(lon,lat)
    x2,y2 = m(lon2,lat2)

    x_shift,y_shift   = m(lon_shift,lat_shift)
    x2_shift,y2_shift = m(lon2_shift,lat2_shift) 
 
  # Map/figure has been set up here, save axes instances for use again later
    if par == 1:
      keep_ax_lst_1 = ax.get_children()[:]
    elif par == 2:
      keep_ax_lst_2 = ax.get_children()[:]
    elif par == 3:
      keep_ax_lst_3 = ax.get_children()[:]

    par += 1
  par = 1

  return fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,m,x,y,x2,y2,x_shift,y_shift,x2_shift,y2_shift,xscale,yscale,im,par


def plot_sets(set):
# Add print to see if dom is being passed in
  print(('plot_sets dom variable '+dom))

  global fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,m,x,y,x2,y2,x_shift,y_shift,x2_shift,y2_shift,xscale,yscale,im,par

  if set == 1:
    plot_set_1()
  elif set == 2:
    plot_set_2()
  elif set == 3:
    plot_set_3()

def plot_set_1():
  global fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,m,x,y,x2,y2,x_shift,y_shift,x2_shift,y2_shift,xscale,yscale,im,par

################################
  # Plot SLP
################################
  t1dom = time.clock()
  t1 = time.clock()
  print(('Working on slp for '+dom))

  units = 'mb'
  clevs = [976,980,984,988,992,996,1000,1004,1008,1012,1016,1020,1024,1028,1032,1036,1040,1044,1048,1052]
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  cm = plt.cm.Spectral_r
  cmdif = matplotlib.colors.ListedColormap(difcolors)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1_a = m.pcolormesh(x_shift,y_shift,slpsmooth1,cmap=cm,norm=norm,ax=ax)  
      cbar1 = m.colorbar(cs1_a,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=5)
      cs1_b = m.contour(x,y,slpsmooth1,np.arange(940,1060,4),colors='black',linewidths=1.25,ax=ax)
      plt.clabel(cs1_b,inline=1,fmt='%d',fontsize=6,zorder=12,ax=ax)
      ax.text(.5,1.03,'LAMX(GFSv15) SLP ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)
  # plot highs and lows - window parameter controls the number of highs and lows detected
      ncepy.plt_highs_and_lows(m,slp_1,lon,lat,mode='reflect',window=500)

    elif par == 2:
      cs2_a = m.pcolormesh(x_shift,y_shift,slpsmooth2,cmap=cm,norm=norm,ax=ax)  
      cbar2 = m.colorbar(cs2_a,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=5)
      cs2_b = m.contour(x2,y2,slpsmooth2,np.arange(940,1060,4),colors='black',linewidths=1.25,ax=ax)
      plt.clabel(cs2_b,inline=1,fmt='%d',fontsize=6,zorder=12,ax=ax)
      ax.text(.5,1.03,'LAMX(GFSv16) SLP ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)
  # plot highs and lows - window parameter controls the number of highs and lows detected
      ncepy.plt_highs_and_lows(m,slp_2,lon,lat,mode='reflect',window=500)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,slp_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'LAMX(GFSv16) - LAMX(GFSv15) SLP ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareslp_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot slp for: '+dom) % t3)

#################################
  # Plot 2-m T
#################################
  t1 = time.clock()
  print(('Working on t2m for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '\xb0''F'
  clevs = np.linspace(-16,134,51)
  clevsdif = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
  cm = cmap_t2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,tmp2m_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=[-16,-4,8,20,32,44,56,68,80,92,104,116,128],extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'LAMX(GFSv15) 2-m Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,tmp2m_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=[-16,-4,8,20,32,44,56,68,80,92,104,116,128],extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'LAMX(GFSv16) 2-m Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))       
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,tmp2m_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'LAMX(GFSv16) - LAMX(GFSv15) 2-m Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2')) 
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compare2mt_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 2mt for: '+dom) % t3)

#################################
# Plot SFCT
#################################
  t1 = time.clock()
  print(('Working on tsfc for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '\xb0''F'
  clevs = np.linspace(-16,134,51)
  clevsdif = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
  cm = cmap_t2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,tmpsfc_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=[-16,-4,8,20,32,44,56,68,80,92,104,116,128],extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'LAMX(GFSv15) Surface Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,tmpsfc_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=[-16,-4,8,20,32,44,56,68,80,92,104,116,128],extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'LAMX(GFSv16) Surface Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,tmpsfc_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'LAMX(GFSv16) - LAMX(GFSv15) Surface Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))       
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparetsfc_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot tsfc for: '+dom) % t3)

#################################
  # Plot 2-m Dew Point
#################################
  t1 = time.clock()
  print(('Working on 2mdew for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '\xb0''F'
  clevs = np.linspace(-5,80,35)
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  cm = ncepy.cmap_q2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,dew2m_1,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'LAMX(GFSv15) 2-m Dew Point Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,dew2m_2,cmap=cm,norm=norm,ax=ax)
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'LAMX(GFSv16) 2-m Dew Point Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,dew2m_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'LAMX(GFSv16) - LAMX(GFSv15) 2-m Dew Point Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compare2mdew_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 2mdew for: '+dom) % t3)

#################################
  # Plot 850-mb THETAE
#################################
  t1 = time.clock()
  print(('Working on 850 mb Theta-e for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'K'
# Wind barb density settings for 850, 500, and 250 mb plots
  if dom == 'conus':
    skip = 100
  elif dom == 'SE':
    skip = 40
  elif dom == 'CO' or dom == 'LA' or dom =='MA':
    skip = 18
  elif dom == 'BN':
    skip = 15
  elif dom == 'SP':
    skip = 13
  elif dom == 'SF':
    skip = 4
  else:
    skip = 30
  barblength = 4

  clevs = np.linspace(270,360,31)
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  cm = cmap_t850()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  # Rotate winds to gnomonic projection
  urot_1, vrot_1 = m.rotate_vector(u850_1,v850_1,lon,lat)
  urot_2, vrot_2 = m.rotate_vector(u850_2,v850_2,lon2,lat2)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,thetae_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=[270,276,282,288,294,300,306,312,318,324,330,336,342,348,354,360],extend='both')
      cbar1.set_label(units,fontsize=6)   
      cbar1.ax.tick_params(labelsize=4)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],urot_1[::skip,::skip],vrot_1[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='black',ax=ax)
      ax.text(.5,1.03,'LAMX(GFSv15) 850 mb $\Theta$e ('+units+') and Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,thetae_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=[270,276,282,288,294,300,306,312,318,324,330,336,342,348,354,360],extend='both')
      cbar2.set_label(units,fontsize=6)   
      cbar2.ax.tick_params(labelsize=4)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],urot_2[::skip,::skip],vrot_2[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='black',ax=ax)
      ax.text(.5,1.03,'LAMX(GFSv16) 850 mb $\Theta$e ('+units+') and Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)
    
    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,thetae_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)   
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'LAMX(GFSv16) - LAMX(GFSv15) 850 mb $\Theta$e ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compare850t_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 850 mb Theta-e for: '+dom) % t3)


#################################
  # Plot 500 mb HGT/WIND/VORT
#################################
  t1 = time.clock()
  print(('Working on 500 mb Hgt/Wind/Vort for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'x10${^5}$ s${^{-1}}$'
  vortlevs = [16,20,24,28,32,36,40]
  clevsdif = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
  colorlist = ['yellow','gold','goldenrod','orange','orangered','red']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(vortlevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  # Rotate winds to gnomonic projection
  urot_1, vrot_1 = m.rotate_vector(u500_1,v500_1,lon,lat)
  urot_2, vrot_2 = m.rotate_vector(u500_2,v500_2,lon2,lat2)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1_a = m.pcolormesh(x_shift,y_shift,vort500_1,cmap=cm,norm=norm,ax=ax)
      cs1_a.cmap.set_under('white')
      cs1_a.cmap.set_over('darkred')
      cbar1 = m.colorbar(cs1_a,ax=ax,location='bottom',pad=0.05,ticks=vortlevs,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)

      # plot vorticity maxima as black X's
      local_max = extrema(vort500_1,mode='wrap',window=100)
      xhighs = lon[local_max]
      yhighs = lat[local_max]
      highvals = vort500_1[local_max]
      xyplotted = []
      # don't plot if there is already a X within dmin meters
      yoffset = 0.022*(m.ymax - m.ymin)
      dmin = yoffset
      for x,y,p in zip(xhighs, yhighs, highvals):
        if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin and p > 35:
          dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
          if not dist or min(dist) > dmin:
            ax.text(x,y,'x',fontsize=6,fontweight='bold',\
                    ha='center',va='center',color='black',zorder=10)
            xyplotted.append((x,y))

      m.barbs(lon[::skip,::skip],lat[::skip,::skip],urot_1[::skip,::skip],vrot_1[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='steelblue',ax=ax)
      x,y = m(lon,lat)	# need to redefine to avoid index error
      cs1_b = m.contour(x,y,z500_1,np.arange(486,600,6),colors='black',linewidths=1,ax=ax)
      plt.clabel(cs1_b,inline_spacing=1,fmt='%d',fontsize=6,dorder=12,ax=ax)
      ax.text(.5,1.03,'LAMX(GFSv15) 500 mb Heights (dam), Winds (kts), and $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs2_a = m.pcolormesh(x2_shift,y2_shift,vort500_2,cmap=cm,norm=norm,ax=ax)
      cs2_a.cmap.set_under('white')
      cs2_a.cmap.set_over('darkred')
      cbar2 = m.colorbar(cs2_a,ax=ax,location='bottom',pad=0.05,ticks=vortlevs,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)

      # plot vorticity maxima as black X's
      local_max = extrema(vort500_2,mode='wrap',window=100)
      xhighs = lon[local_max]
      yhighs = lat[local_max]
      highvals = vort500_2[local_max]
      xyplotted = []
      # don't plot if there is already a X within dmin meters
      yoffset = 0.022*(m.ymax - m.ymin)
      dmin = yoffset
      for x,y,p in zip(xhighs, yhighs, highvals):
        if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin and p > 35:
          dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
          if not dist or min(dist) > dmin:
            ax.text(x,y,'x',fontsize=6,fontweight='bold',\
                    ha='center',va='center',color='black',zorder=10)
            xyplotted.append((x,y))

      m.barbs(lon[::skip,::skip],lat[::skip,::skip],urot_2[::skip,::skip],vrot_2[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='steelblue',ax=ax)
#      x,y = m(lon,lat)	# need to redefine to avoid index error
      cs2_b = m.contour(x2,y2,z500_2,np.arange(486,600,6),colors='black',linewidths=1,ax=ax)
      plt.clabel(cs2_b,inline_spacing=1,fmt='%d',fontsize=6,dorder=12,ax=ax)
      ax.text(.5,1.03,'LAMX(GFSv16) 500 mb Heights (dam), Winds (kts), and $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,z500_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6) 
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'LAMX(GFSv16) - LAMX(GFSv15) 500 mb Heights (dam) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compare500_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 500 mb Hgt/Wind/Vort for: '+dom) % t3)

#################################
  # Plot PW
#################################
  t1 = time.clock()
  print(('Working on PW for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'in'
  clevs = [0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25]
  clevsdif = [-1.25,-1,-.75,-.5,-.25,-.1,0.,.1,.25,.50,.75,1,1.25]
  colorlist = ['lightsalmon','khaki','palegreen','cyan','turquoise','cornflowerblue','mediumslateblue','darkorchid','deeppink']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,pw_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('hotpink')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'LAMX(GFSv15) Precipitable Water ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,pw_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('hotpink')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'LAMX(GFSv16) Precipitable Water ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,pw_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,ticks=clevsdif,extend='both')
      cbar3.set_label(units,fontsize=6) 
      cbar3.ax.tick_params(labelsize=4)
      ax.text(.5,1.03,'LAMX(GFSv16) - LAMX(GFSv15) Precipitable Water ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparepw_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot PW for: '+dom) % t3)

#################################
  # Plot % FROZEN PRECIP
#################################
  t1 = time.clock()
  print(('Working on PERCENT FROZEN PRECIP for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '%'
  clevs = [10,20,30,40,50,60,70,80,90,100]
  clevsdif = [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]
  colorlist = ['blue','dodgerblue','deepskyblue','mediumspringgreen','khaki','sandybrown','salmon','crimson','maroon']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,pofp_1,cmap=cm,vmin=10,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs)
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'LAMX(GFSv15) Percent of Frozen Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,pofp_2,cmap=cm,vmin=10,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs)
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'LAMX(GFSv16) Percent of Frozen Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,pofp_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,ticks=clevsdif,extend='both')
      cbar3.set_label(units,fontsize=6) 
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'LAMX(GFSv16) - LAMX(GFSv15) Percent of Frozen Precipitaion ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparepofp_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot PERCENT FROZEN PRECIP for: '+dom) % t3)


#################################
  # Plot Total QPF
#################################
  if (fhr > 0):
    t1 = time.clock()
    print(('Working on total qpf for '+dom))

    # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)
    clear_plotables(ax3,keep_ax_lst_3,fig)

    units = 'in'
    clevs = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,7,10,15,20]
    clevsdif = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
    colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','deepskyblue','cyan','mediumpurple','mediumorchid','darkmagenta','darkred','crimson','orangered','darkorange','goldenrod','gold','yellow']  
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs_1 = m.pcolormesh(x_shift,y_shift,qpf_1,cmap=cm,vmin=0.01,norm=norm,ax=ax)
        cs_1.cmap.set_under('white',alpha=0.)
        cs_1.cmap.set_over('pink')
        cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
        cbar1.set_label(units,fontsize=6)
        cbar1.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar1.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'LAMX(GFSv15) '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 2:
        cs_2 = m.pcolormesh(x2_shift,y2_shift,qpf_2,cmap=cm,vmin=0.01,norm=norm,ax=ax)
        cs_2.cmap.set_under('white',alpha=0.)
        cs_2.cmap.set_over('pink')
        cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
        cbar2.set_label(units,fontsize=6)
        cbar2.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar2.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'LAMX(GFSv16) '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 3:
        cs = m.pcolormesh(x2_shift,y2_shift,qpf_dif,cmap=cmdif,norm=normdif,ax=ax)
        cs.cmap.set_under('darkblue')
        cs.cmap.set_over('darkred')
        cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar3.set_label(units,fontsize=6)
        cbar3.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'LAMX(GFSv16) - LAMX(GFSv15) '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))         
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('compareqpf_'+dom+'_f'+fhour+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot total qpf for: '+dom) % t3)

#################################
  # Plot QPF3
#################################
  if (fhr % 3 == 0) and (fhr > 0):
    t1 = time.clock()
    print(('Working on qpf3 for '+dom))

    # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)
    clear_plotables(ax3,keep_ax_lst_3,fig)

    units = 'in'
    clevs = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,7,10,15,20]
    clevsdif = [-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5]
    colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','deepskyblue','cyan','mediumpurple','mediumorchid','darkmagenta','darkred','crimson','orangered','darkorange','goldenrod','gold','yellow']  
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)
   
    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs_1 = m.pcolormesh(x_shift,y_shift,qpf3_1,cmap=cm,vmin=0.01,norm=norm,ax=ax)
        cs_1.cmap.set_under('white',alpha=0.)
        cs_1.cmap.set_over('pink')
        cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
        cbar1.set_label(units,fontsize=6)
        cbar1.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar1.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'LAMX(GFSv15) 3-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 2:
        cs_2 = m.pcolormesh(x2_shift,y2_shift,qpf3_2,cmap=cm,vmin=0.01,norm=norm,ax=ax)
        cs_2.cmap.set_under('white',alpha=0.)
        cs_2.cmap.set_over('pink')
        cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
        cbar2.set_label(units,fontsize=6)
        cbar2.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar2.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'LAMX(GFSv16) 3-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 3:
        cs = m.pcolormesh(x2_shift,y2_shift,qpf3_dif,cmap=cmdif,norm=normdif,ax=ax)
        cs.cmap.set_under('darkblue')
        cs.cmap.set_over('darkred')
        cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar3.set_label(units,fontsize=6)
        cbar3.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'LAMX(GFSv16) - LAMX(GFSv15) 3-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))         
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('compareqpf3_'+dom+'_f'+fhour+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot qpf3 for: '+dom) % t3)

#################################
  # Plot snow depth
#################################
  t1 = time.clock()
  print(('Working on snow depth for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'in'
  clevs = [0.1,1,2,3,6,9,12,18,24,36,48]
  clevsdif = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
  cm = ncepy.ncl_perc_11Lev()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N) 
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N) 
 
  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,snow_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'LAMX(GFSv15) Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,snow_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels(clevs)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'LAMX(GFSv16) Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,snow_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'LAMX(GFSv16) - LAMX(GFSv15) Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))         
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparesnow_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot snow depth for: '+dom) % t3)

#################################
  # Plot 6-hr change in snow depth
#################################
  if (fhr % 3 == 0) and (fhr >= 6):
    t1 = time.clock()
    print(('Working on 6-hr change in snow depth for '+dom))

    # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)
    clear_plotables(ax3,keep_ax_lst_3,fig)

    units = 'in'
    clevs = [-6,-4,-3,-2,-1,-0.5,0,0.5,1,2,3,4,6]
    clevsdif = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
    colorlist = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','white','white','#EEEE00','#EEC900','darkorange','orangered','red']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs_1 = m.pcolormesh(x_shift,y_shift,snow6_1,cmap=cm,norm=norm,ax=ax)
        cs_1.cmap.set_under('darkblue')
        cs_1.cmap.set_over('darkred')
        cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
        cbar1.set_label(units,fontsize=6)
        cbar1.ax.set_xticklabels(clevs)
        cbar1.ax.tick_params(labelsize=5)
        ax.text(.5,1.03,'LAMX(GFSv15) 6-hr Change in Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 2:
        cs_2 = m.pcolormesh(x2_shift,y2_shift,snow6_2,cmap=cm,norm=norm,ax=ax)
        cs_2.cmap.set_under('darkblue')
        cs_2.cmap.set_over('darkred')
        cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
        cbar2.set_label(units,fontsize=6)
        cbar2.ax.set_xticklabels(clevs)
        cbar2.ax.tick_params(labelsize=5)
        ax.text(.5,1.03,'LAMX(GFSv16) 6-hr Change in Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 3:
        cs = m.pcolormesh(x2_shift,y2_shift,snow6_dif,cmap=cmdif,norm=normdif,ax=ax)
        cs.cmap.set_under('darkblue')
        cs.cmap.set_over('darkred')
        cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar3.set_label(units,fontsize=6)
        cbar3.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'LAMX(GFSv16) - LAMX(GFSv15) 6-hr Change in Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('comparesnow6_'+dom+'_f'+fhour+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot snow depth for: '+dom) % t3)


#################################
  # Plot Echo Top Height
#################################
  t1 = time.clock()
  print(('Working on Echo Top Height for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'kft'
  clevs = [1,5,10,15,20,25,30,35,40]
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  colorlist = ['firebrick','tomato','lightsalmon','goldenrod','#EEEE00','palegreen','mediumspringgreen','limegreen']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,retop_1,cmap=cm,vmin=1,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('darkgreen')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'LAMX(GFSv15) Echo Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,retop_2,cmap=cm,vmin=1,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('darkgreen')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'LAMX(GFSv16) Echo Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,retop_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'LAMX(GFSv16) - LAMX(GFSv15) Echo Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareretop_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Echo Top Height for: '+dom) % t3)

#################################
  # Plot Precipitation Rate
#################################
  t1 = time.clock()
  print(('Working on Precipitation Rate for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'mm/hr'
  clevs = [0.01,0.05,0.1,0.5,1,2.5,5,7.5,10,15,20,30,50,75,100]
  clevsdif = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
  colorlist = ['chartreuse','limegreen','green','darkgreen','blue','dodgerblue','deepskyblue','cyan','darkred','crimson','orangered','darkorange','goldenrod','gold']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,prate_1,cmap=cm,vmin=0.01,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('yellow')
      cbar1 = m.colorbar(cs_1,ax=ax,ticks=clevs,location='bottom',pad=0.05,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'LAMX(GFSv15) Precipitation Rate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,prate_2,cmap=cm,vmin=0.01,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('yellow')
      cbar2 = m.colorbar(cs_2,ax=ax,ticks=clevs,location='bottom',pad=0.05,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels(clevs)
      cbar2.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'LAMX(GFSv16) Precipitation Rate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,prate_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'LAMX(GFSv16) - LAMX(GFSv15) Precipitation Rate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareprate_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Precipitation Rate for: '+dom) % t3)


######################################################

  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all set 1 variables for: "+dom) % t3dom)
  plt.clf()

######################################################


main()

