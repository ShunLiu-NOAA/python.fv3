import pygrib
import matplotlib
matplotlib.use('Agg')
import io
import matplotlib.pyplot as plt
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


def cmap_wv():
 # Create colormap for brightness temperature - water vapor channels 8,9,10
 # Color scale for 170-300 K
    r=np.array([255,255,255,255,255,255,8,10,11,13,14,15,17,18,20,21,22,24,25,27,28,29,31,32,34,35,36,38,39,40,42,43,45,46,47,49,50,52,53,54,56,57,59,60,61,63,64,66,67,67,80,92,105,117,130,142,155,167,180,192,205,217,230,242,255,255,246,236,227,217,208,198,189,179,170,161,151,142,132,123,113,104,94,85,76,66,57,47,38,28,19,9,0,0,13,27,40,54,67,81,94,107,121,134,148,161,174,188,201,215,228,242,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ])
    g=np.array([255,255,255,255,255,255,240,237,234,232,229,227,224,221,219,216,214,211,209,206,203,201,198,196,193,190,188,185,183,180,178,175,172,170,167,165,162,159,157,154,152,149,147,144,141,139,136,134,131,131,139,148,156,164,172,181,189,197,205,214,222,230,238,247,255,255,246,236,227,217,208,198,189,179,170,161,151,142,132,123,113,104,94,85,76,66,57,47,38,28,19,9,0,0,13,27,40,54,67,81,94,107,121,134,148,161,174,188,201,215,228,242,255,255,246,237,229,220,211,202,193,185,176,167,158,149,141,132,123,114,106,97,88,79,70,62,53,44,35,26,18,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ])
    b=np.array([255,255,255,255,255,255,228,223,218,214,209,205,200,195,191,186,182,177,173,168,163,159,154,150,145,140,136,131,127,122,118,113,108,104,99,95,90,85,81,76,72,67,63,58,53,49,44,40,35,35,50,64,79,94,108,123,138,152,167,182,196,211,226,240,255,255,250,245,239,234,229,224,219,214,208,203,198,193,188,182,177,172,167,162,156,151,146,141,136,131,125,120,115,115,109,103,97,91,85,79,73,67,61,54,48,42,36,30,24,18,12,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ])
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
    cmap_wv_coltbl = matplotlib.colors.LinearSegmentedColormap('CMAP_WV_COLTBL',colorDict)
    return cmap_wv_coltbl

def cmap_ir():
 # Create colormap for brightness temperature - infrared channel 13
    r=np.array([255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,127,140,153,165,178,191,204,217,229,242,255,230,205,179,154,128,103,77,52,26,0,26,51,77,102,128,153,179,204,230,255,255,255,255,255,255,255,255,255,255,255,230,204,179,153,128,102,77,51,26,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,254,252,249,247,244,242,239,237,234,232,229,226,224,221,219,216,214,211,209,206,203,201,198,196,193,191,188,186,183,181,178,175,173,170,168,165,163,160,158,155,152,150,147,145,142,140,137,135,132,130,127,124,122,119,117,114,112,109,107,104,101,99,96,94,91,89,86,84,81,79,76,73,71,68,66,63,61,58,56,53,50,48,45,43,40,38,35,33,30,28,25,22,20,17,15,12,10,7,5,2,0,0,0,0,0,0 ])
    g=np.array([255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,13,25,38,51,64,76,89,102,114,127,230,205,179,154,128,103,77,52,26,0,0,0,0,0,0,0,0,0,0,0,26,51,77,102,128,153,179,204,230,255,255,255,255,255,255,255,255,255,255,255,234,213,191,170,149,128,106,85,64,43,21,0,0,13,26,38,51,64,77,89,102,115,128,140,153,166,179,191,204,217,230,242,255,255,255,255,255,255,255,255,254,252,249,247,244,242,239,237,234,232,229,226,224,221,219,216,214,211,209,206,203,201,198,196,193,191,188,186,183,181,178,175,173,170,168,165,163,160,158,155,152,150,147,145,142,140,137,135,132,130,127,124,122,119,117,114,112,109,107,104,101,99,96,94,91,89,86,84,81,79,76,73,71,68,66,63,61,58,56,53,50,48,45,43,40,38,35,33,30,28,25,22,20,17,15,12,10,7,5,2,0,0,0,0,0,0 ])
    b=np.array([255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,127,135,142,150,157,165,173,180,188,195,203,230,205,179,154,128,103,77,52,26,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,19,29,38,48,58,67,77,86,96,105,115,115,122,129,136,143,150,157,164,171,178,185,192,199,206,213,220,227,234,241,248,255,255,255,255,255,255,255,255,254,252,249,247,244,242,239,237,234,232,229,226,224,221,219,216,214,211,209,206,203,201,198,196,193,191,188,186,183,181,178,175,173,170,168,165,163,160,158,155,152,150,147,145,142,140,137,135,132,130,127,124,122,119,117,114,112,109,107,104,101,99,96,94,91,89,86,84,81,79,76,73,71,68,66,63,61,58,56,53,50,48,45,43,40,38,35,33,30,28,25,22,20,17,15,12,10,7,5,2,0,0,0,0,0,0 ])
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
    cmap_ir_coltbl = matplotlib.colors.LinearSegmentedColormap('CMAP_IR_COLTBL',colorDict)
    return cmap_ir_coltbl

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


###################################################
# Read in all variables                           #
###################################################
def read_variables(dom):
  t1a = time.clock()

# Define the input files - different based on which domain you are reading in!
  datagoes = pygrib.open('/gpfs/dell6/ptmp/emc.campara/fv3lamx/fv3lamx.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.rrfsgoestb.f'+fhour+'.grib2')
#  datagoes = pygrib.open('/gpfs/dell3/stmp/Benjamin.Blake/test/fv3lam.t'+cyc+'z.rrfsgoestb.f'+fhour+'.grib2')


# GOES-16 brightness temperatures
  tb_ch8 = datagoes.select(parameterName="21")[0].values
  tb_ch9 = datagoes.select(parameterName="22")[0].values
  tb_ch10 = datagoes.select(parameterName="23")[0].values
  tb_ch13 = datagoes.select(parameterName="26")[0].values

  print(np.min(tb_ch8))
  print(np.max(tb_ch8))
  print(np.min(tb_ch9))
  print(np.max(tb_ch9))
  print(np.min(tb_ch10))
  print(np.max(tb_ch10))
  print(np.min(tb_ch13))
  print(np.max(tb_ch13))


  t2a = time.clock()
  t3a = round(t2a-t1a, 3)
  print(("%.3f seconds to read all messages") % t3a)

  return tb_ch8, tb_ch9, tb_ch10, tb_ch13



##################################### START OF SCRIPT #####################################

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

# Forecast valid date/time
itime = ymdh
vtime = ncepy.ndate(itime,int(fhr))

# Specify plotting domains
domains=['namerica']

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

  global fig,axes,ax1,keep_ax_lst_1,m,x,y,x_shift,y_shift,lon,lat,lon_shift,lat_shift,xscale,yscale,im,par
  fig,axes,ax1,keep_ax_lst_1,m,x,y,x_shift,y_shift,lon,lat,lon_shift,lat_shift,xscale,yscale,im,par = create_figure(dom)

  global tb_ch8, tb_ch9, tb_ch10, tb_ch13

  tb_ch8, tb_ch9, tb_ch10, tb_ch13 = read_variables(dom)

  # Split plots into 2 sets with multiprocessing
#  sets = [1,2,3]
  sets = [1]
  pool2 = multiprocessing.Pool(len(sets))
  pool2.map(plot_sets,sets)

def create_figure(dom):

# Define the input files - different based on which domain you are reading in!
  datagoes = pygrib.open('/gpfs/dell6/ptmp/emc.campara/fv3lamx/fv3lamx.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.rrfsgoestb.f'+fhour+'.grib2')
#  datagoes = pygrib.open('/gpfs/dell3/stmp/Benjamin.Blake/test/fv3lam.t'+cyc+'z.rrfsgoestb.f'+fhour+'.grib2')

# Get the lats and lons
  grids = [datagoes]
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
      try:
          dx = data[1]['DxInMetres']
          dy = data[1]['DyInMetres']
      except:
          dx = data[1]['DiInMetres']
          dy = data[1]['DjInMetres']
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

# Shifted lat/lon arrays for pcolormesh
  lat_shift = lats_shift[0]
  lon_shift = lons_shift[0]

  Lat0 = datagoes[1]['LaDInDegrees']
#  Lon0 = datagoes[1]['LoVInDegrees']
  print(Lat0)
#  print(Lon0)


  # create figure and axes instances
  fig = plt.figure()
  gs = GridSpec(4,4,wspace=0.0,hspace=0.0)
  ax1 = fig.add_subplot(gs[:,:])
  axes = [ax1]
  im = image.imread('/gpfs/dell2/emc/modeling/noscrub/Benjamin.Blake/python.fv3/noaa.png')
  par = 1

  # Map corners for each domain
  if dom == 'ak':
    llcrnrlon = -165.0
    llcrnrlat = 50.0
    urcrnrlon = -121.0
    urcrnrlat = 70.0
    lat_0 = 90.0 
    lon_0 = -150.0
    lat_ts = 60.0
    xscale=0.15
    yscale=0.13
  elif dom == 'hi':
    llcrnrlon = -162.2
    llcrnrlat = 16.5
    urcrnrlon = -152.5
    urcrnrlat = 24.0
    xscale=0.14
    yscale=0.19
  elif dom == 'pr':
    llcrnrlon = -70.7
    llcrnrlat = 14.6
    urcrnrlon = -62.3
    urcrnrlat = 21.8
    xscale=0.15
    yscale=0.18
  elif dom == 'namerica':
    llcrnrlon = -161.0
    llcrnrlat = 8.5
    urcrnrlon = -22.75
    urcrnrlat = 40.25
    lat_0 = 45.0
    lon_0 = -120.0
    lat_ts = 30.0
    xscale=0.08
    yscale=0.12


  # Create basemap instance and set the dimensions
  for ax in axes:
    if dom == 'ak':
      m = Basemap(ax=ax,projection='stere',lat_ts=lat_ts,lat_0=lat_0,lon_0=lon_0,\
                  llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                  llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                  rsphere=6371229,resolution='i')
#                  rsphere=(6378137.00,6356752.3142),resolution='i')
    elif dom == 'hi':
      m = Basemap(ax=ax,projection='cyl',\
                  llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                  llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                  rsphere=6371229,resolution='h')
#                  rsphere=(6378137.00,6356752.3142),resolution='h')
    elif dom == 'pr':
      m = Basemap(ax=ax,projection='cyl',\
                  llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                  llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                  rsphere=6371229,resolution='h')
#                  rsphere=(6378137.00,6356752.3142),resolution='h')
    elif dom == 'namerica':
      m = Basemap(ax=ax,projection='stere',lat_ts=lat_ts,lat_0=lat_0,lon_0=lon_0,\
                  llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,\
                  urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,\
                  rsphere=6371229,resolution='l')

    m.fillcontinents(color='LightGrey',zorder=0)
    m.drawcoastlines(linewidth=0.75)
    m.drawstates(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
##  parallels = np.arange(0.,90.,10.)
##  map.drawparallels(parallels,labels=[1,0,0,0],fontsize=6)
##  meridians = np.arange(180.,360.,10.)
##  map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=6)
    x,y = m(lon,lat)
    x_shift,y_shift   = m(lon_shift,lat_shift)
 
  # Map/figure has been set up here, save axes instances for use again later
    keep_ax_lst_1 = ax.get_children()[:]

    par += 1
  par = 1

  return fig,axes,ax1,keep_ax_lst_1,m,x,y,x_shift,y_shift,lon,lat,lon_shift,lat_shift,xscale,yscale,im,par


def plot_sets(set):
# Add print to see if dom is being passed in
  print(('plot_sets dom variable '+dom))

  global fig,axes,ax1,keep_ax_lst_1,m,x,y,x_shift,y_shift,lon,lat,lon_shift,lat_shift,xscale,yscale,im,par

  if set == 1:
    plot_set_1()
  elif set == 2:
    plot_set_2()
  elif set == 3:
    plot_set_3()

def plot_set_1():
  global fig,axes,ax1,keep_ax_lst_1,m,x,y,x_shift,y_shift,lon,lat,lon_shift,lat_shift,xscale,yscale,im,par


#################################
  # Plot GOES 16 ABI Band 8 BT
#################################
  t1 = time.clock()
  t1dom = time.clock()
  print(('Working on GOES-16 ABI Band 8 BT for '+dom))

  # Clear off old plottables but keep all the map info

  units = 'K'
  clevs = np.linspace(170,300,131)
  cm = cmap_wv()
#  cm = plt.cm.inferno
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,tb_ch8,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('black')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=[175,200,225,250,275,300],extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X GOES-16 ABI Band 8 Brightness Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparetb8_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot GOES-16 ABI Band 8 BT for: '+dom) % t3)


#################################
  # Plot GOES 16 ABI Band 9 BT
#################################
  t1 = time.clock()
  t1dom = time.clock()
  print(('Working on GOES-16 ABI Band 9 BT for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,tb_ch9,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('black')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=[175,200,225,250,275,300],extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X GOES-16 ABI Band 9 Brightness Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparetb9_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot GOES-16 ABI Band 9 BT for: '+dom) % t3)


#################################
  # Plot GOES 16 ABI Band 10 BT
#################################
  t1 = time.clock()
  t1dom = time.clock()
  print(('Working on GOES-16 ABI Band 10 BT for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,tb_ch10,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('black')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=[175,200,225,250,275,300],extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X GOES-16 ABI Band 10 Brightness Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparetb10_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot GOES-16 ABI Band 10 BT for: '+dom) % t3)


#################################
  # Plot GOES 16 ABI Band 13 BT
#################################
  t1 = time.clock()
  t1dom = time.clock()
  print(('Working on GOES-16 ABI Band 13 BT for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  clevs = np.linspace(170,350,181)
  cm = cmap_ir()
#  cm = plt.cm.inferno
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,tb_ch13,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('black')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=[175,200,225,250,275,300,325,350],extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X GOES-16 ABI Band 13 Brightness Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparetb13_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot GOES-16 ABI Band 13 BT for: '+dom) % t3)



######################################################

  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all set 3 variables for: "+dom) % t3dom)
  plt.clf()

######################################################


main()

