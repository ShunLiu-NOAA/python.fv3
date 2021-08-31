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


def colors_ivt():
# Create colormap for ivt
    colors=[(255, 255, 0, 255),(255,229, 0, 255), (255,201, 0, 255), (255, 173, 0, 255), \
            (255, 130, 0, 255), (255, 80, 0, 255), (255, 30, 0, 255), (235, 0, 16, 255), \
            (184, 0, 58, 255), (133, 0, 99, 255), (87, 0, 136, 255)]
    colors = [tuple(ti/255.0 for ti in element) for element in colors]
    print(colors)
    return colors


def colors_iwv():
# Create colormap for iwv
    colors=[(0, 0, 204, 255), (0, 31, 255, 255), (0, 112, 255, 255), (0, 194, 255, 255), \
            (1, 249, 236, 255), (3, 225, 159, 255), (6, 200, 82, 255), (18, 182, 14, 255), \
            (97, 206, 16, 255), (176, 231, 5, 255), (255, 255,0,255), (225, 225, 0,255), \
            (255,195, 0, 255), (255, 165, 0, 255), (255, 112, 0, 255), (255, 59, 0, 255), \
            (255, 7, 0, 255), (208, 0, 38, 255), (154, 0, 82, 255), (100, 0, 125, 255), \
            (127, 61, 165, 255)]
    colors = [tuple(ti/255.0 for ti in element) for element in colors]
    print(colors)
    return colors


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
fhour = str(fhr).zfill(2)
print('fhour '+fhour)

# Forecast valid date/time
itime = ymdh
vtime = ncepy.ndate(itime,int(fhr))

# Specify plotting domains
domains=['namerica']
dom = 'namerica'

# colors for difference plots, only need to define once
difcolors = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','white','white','#EEEE00','#EEC900','darkorange','orangered','red']
difcolors2 = ['white']
difcolors3 = ['blue','dodgerblue','turquoise','white','white','#EEEE00','darkorange','red']

########################################
#    START PLOTTING FOR EACH DOMAIN    #
########################################

def main():

# Define the input files - different based on which domain you are reading in!
  data1 = pygrib.open('/gpfs/dell4/ptmp/emc.campara/fv3lam/fv3lam.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.'+dom+'.f'+fhour+'.grib2')

# Get the lats and lons
  grids = [data1]
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

  Lat0 = data1[1]['LaDInDegrees']
#  Lon0 = data1[1]['LoVInDegrees']
  print(Lat0)
#  print(Lon0)


  # create figure and axes instances
  fig = plt.figure()
  gs = GridSpec(4,4,wspace=0.0,hspace=0.0)
  ax1 = fig.add_subplot(gs[:,:])
  axes = [ax1]
  im = image.imread('/gpfs/dell2/emc/modeling/noscrub/Shun.Liu/python.fv3/noaa.png')
  par = 1

  # Map corners for each domain
  if dom == 'namerica':
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
    if dom == 'namerica':
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


################################
  # Calculate IVT and IWV
################################

  clevs = [500,600,700,800,850,900,925,950,975,1000]

  count = 0
  IWV = 0
  IVT = 0
  IVT_U = 0
  IVT_V = 0
  for lev in clevs:
    spfh = data1.select(name= "Specific humidity", typeOfLevel="isobaricInhPa", level=lev)[0].values
    uwind = data1.select(name='U component of wind', typeOfLevel="isobaricInhPa", level=lev)[0].values
    vwind = data1.select(name='V component of wind', typeOfLevel="isobaricInhPa", level=lev)[0].values
    if count == 0:
        dp = (clevs[1] - clevs[0])*0.5
    elif count == len(clevs)-1:
        dp = (lev - clevs[-2])*0.5
    else:
        dp = (clevs[count+1] - clevs[count-1])*0.5

    q = spfh
    IWV = IWV + q * dp
    IVT_V = IVT_V + q * vwind *dp
    IVT_U = IVT_U + q * uwind *dp

    count = count + 1

  g = 9.81
  IWV = IWV/g*100
  IVT_U = IVT_U/g*100
  IVT_V = IVT_V/g*100
  IVT = np.sqrt(IVT_U*IVT_U + IVT_V*IVT_V)


################################
  # Plot IVT
################################

  t1 = time.clock()

  units = 'kg m${^{-1}}$ s${^{-1}}$'
  skip = 100
  bounds = [249,250,300,400,500,600,700,800,1000,1200,1400,1600,1601]
  colors = colors_ivt()
  cm = matplotlib.colors.ListedColormap(colors)
  norm = matplotlib.colors.BoundaryNorm(bounds, cm.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,IVT,cmap=cm,norm=norm,vmin=250,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over(colors[-1])
#      cs_1.cmap.set_over('mediumpurple')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,boundaries=bounds,ticks=[250,300,400,500,600,700,800,1000,1200,1400,1600],extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM IVT ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      arrows = m.quiver(x_shift[::skip,::skip],y_shift[::skip,::skip],IVT_U[::skip,::skip],IVT_V[::skip,::skip],scale_units='xy')
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareIVT_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot IVT for: '+dom) % t3)

#################################
  # Plot IWV
#################################
  t1 = time.clock()
  print(('Working on IWV for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'mm'
  bounds = np.linspace(18,62,23)
  colors = colors_iwv()
  cm = matplotlib.colors.ListedColormap(colors)        
  norm = matplotlib.colors.BoundaryNorm(bounds, cm.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,IWV,cmap=cm,norm=norm,vmin=20,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over(colors[-1])
#      cs_1.cmap.set_over('indigo')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,boundaries=bounds,ticks=np.linspace(20,60,21),extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM IWV ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareIWV_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot IWV for: '+dom) % t3)

  plt.clf()

######################################################


main()

