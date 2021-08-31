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


def cmap_difcolors():
 # Create colormap for differences
    r=np.array([0,    0, 0.0941, 0.1176, 0,     0.251,  1, 1, 0.9333, 0.9333, 1,     1,      1, 0.75])
    g=np.array([0,    0, 0.4549, 0.5647, 0.749, 0.8784, 1, 1, 0.9333, 0.7882, 0.549, 0.2706, 0, 0])
    b=np.array([0.75, 1, 0.8039, 1,      1,     0.8157, 1, 1, 0,      0,      0,     0,      0, 0])
    xsize=np.arange(np.size(r))
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=np.float(i)/(np.float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_dif_coltbl = matplotlib.colors.LinearSegmentedColormap('CMAP_DIF_COLTBL',colorDict)
    return cmap_dif_coltbl


##################################### START OF SCRIPT #####################################

# Read date/time and tmmark from command line
ymdh = str(sys.argv[1])
ymd = ymdh[0:8]
year = int(ymdh[0:4])
month = int(ymdh[4:6])
day = int(ymdh[6:8])
hour = int(ymdh[8:10])
cyc = str(hour).zfill(2)
print(year, month, day, hour)

tmmark = str(sys.argv[2])
print('tmmark '+tmmark)

# Forecast valid date/time
itime = ymdh
#vtime = ncepy.ndate(itime,int(fhr))

# Specify plotting domains
domains=['conus']

# colors for difference plots, only need to define once
difcolors = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','white','white','#EEEE00','#EEC900','darkorange','orangered','red']


###################################################
# Read in all variables                           #
###################################################
t1a = time.clock()

# Define the input files - different based on which tmmark you are reading in!
if tmmark == 'tm00':
  vtime = ymdh
  anldata = pygrib.open('/gpfs/dell5/ptmp/emc.campara/fv3lamda/fv3lamda.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f00.grib2')
  gesdata = pygrib.open('/gpfs/dell5/ptmp/emc.campara/fv3lamda/fv3lamda.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f01.tm01.grib2')
elif tmmark == 'tm01':
  vtime = ncepy.ndate(itime,-1)
  anldata = pygrib.open('/gpfs/dell5/ptmp/emc.campara/fv3lamda/fv3lamda.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f00.tm01.grib2')
  gesdata = pygrib.open('/gpfs/dell5/ptmp/emc.campara/fv3lamda/fv3lamda.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f01.tm02.grib2')
elif tmmark == 'tm02':
  vtime = ncepy.ndate(itime,-2)
  anldata = pygrib.open('/gpfs/dell5/ptmp/emc.campara/fv3lamda/fv3lamda.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f00.tm02.grib2')
  gesdata = pygrib.open('/gpfs/dell5/ptmp/emc.campara/fv3lamda/fv3lamda.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f01.tm03.grib2')
elif tmmark == 'tm03':
  vtime = ncepy.ndate(itime,-3)
  anldata = pygrib.open('/gpfs/dell5/ptmp/emc.campara/fv3lamda/fv3lamda.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f00.tm03.grib2')
  gesdata = pygrib.open('/gpfs/dell5/ptmp/emc.campara/fv3lamda/fv3lamda.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f01.tm04.grib2')
elif tmmark == 'tm04':
  vtime = ncepy.ndate(itime,-4)
  anldata = pygrib.open('/gpfs/dell5/ptmp/emc.campara/fv3lamda/fv3lamda.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f00.tm04.grib2')
  gesdata = pygrib.open('/gpfs/dell5/ptmp/emc.campara/fv3lamda/fv3lamda.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f01.tm05.grib2')
elif tmmark == 'tm05':
  vtime = ncepy.ndate(itime,-5)
  anldata = pygrib.open('/gpfs/dell5/ptmp/emc.campara/fv3lamda/fv3lamda.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f00.tm05.grib2')
  gesdata = pygrib.open('/gpfs/dell5/ptmp/emc.campara/fv3lamda/fv3lamda.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f01.tm06.grib2')
elif tmmark == 'tm06':
  vtime = ncepy.ndate(itime,-6)
  anldata = pygrib.open('/gpfs/dell5/ptmp/emc.campara/fv3lamda/fv3lamda.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f00.tm06.grib2')
  gesdata = pygrib.open('/gpfs/dell3/stmp/Benjamin.Blake/increments/'+ymdh+'/gdas.t'+cyc+'z.guess.tm06.grib2')
# For specific humidity fields
#  gesdata2 = pygrib.open('/gpfs/dell3/stmp/Benjamin.Blake/increments/'+ymdh+'/gdas.t'+cyc+'z.guess2.tm06.grib2')

if tmmark != 'tm06':
# lowest model level (hybrid level 1) temperature
  tmphyb_anl = anldata.select(name='Temperature',typeOfLevel='hybrid',level=1)[0].values
  tmphyb_ges = gesdata.select(name='Temperature',typeOfLevel='hybrid',level=1)[0].values
  tmphyb = tmphyb_anl - tmphyb_ges

# lowest model level (hybrid level 1) specific humidity
  spfhhyb_anl = anldata.select(name='Specific humidity',typeOfLevel='hybrid',level=1)[0].values * 1000
  spfhhyb_ges = gesdata.select(name='Specific humidity',typeOfLevel='hybrid',level=1)[0].values * 1000
  spfhhyb = spfhhyb_anl - spfhhyb_ges

# lowest model level wind speed
  uhyb_anl = anldata.select(name='U component of wind',typeOfLevel='hybrid',level=1)[0].values
  vhyb_anl = anldata.select(name='V component of wind',typeOfLevel='hybrid',level=1)[0].values
  uhyb_ges = gesdata.select(name='U component of wind',typeOfLevel='hybrid',level=1)[0].values
  vhyb_ges = gesdata.select(name='V component of wind',typeOfLevel='hybrid',level=1)[0].values
  wspdhyb_anl = np.sqrt(uhyb_anl**2 + vhyb_anl**2)
  wspdhyb_ges = np.sqrt(uhyb_ges**2 + vhyb_ges**2)
  wspdhyb = wspdhyb_anl - wspdhyb_ges

# surface pressure
slp_anl = anldata.select(name='Pressure reduced to MSL')[0].values * 0.01
slp_ges = gesdata.select(name='Pressure reduced to MSL')[0].values * 0.01
slp = slp_anl - slp_ges

# Precipitable water
pw_anl = anldata.select(name='Precipitable water',level=0)[0].values
pw_ges = gesdata.select(name='Precipitable water',level=0)[0].values
pw = pw_anl - pw_ges

# 850 hPa geopotential height
z850_anl = anldata.select(name='Geopotential Height',level=850)[0].values
z850_ges = gesdata.select(name='Geopotential Height',level=850)[0].values
z850 = z850_anl - z850_ges

# 700 hPa geopotential height
z700_anl = anldata.select(name='Geopotential Height',level=700)[0].values
z700_ges = gesdata.select(name='Geopotential Height',level=700)[0].values
z700 = z700_anl - z700_ges

# 500 hPa geopotential height
z500_anl = anldata.select(name='Geopotential Height',level=500)[0].values
z500_ges = gesdata.select(name='Geopotential Height',level=500)[0].values
z500 = z500_anl - z500_ges

# 250 hPa geopotential height
z250_anl = anldata.select(name='Geopotential Height',level=250)[0].values
z250_ges = gesdata.select(name='Geopotential Height',level=250)[0].values
z250 = z250_anl - z250_ges

# 850 hPa temperature
t850_anl = anldata.select(name='Temperature',level=850)[0].values
t850_ges = gesdata.select(name='Temperature',level=850)[0].values
t850 = t850_anl - t850_ges

# 700 hPa temperature
t700_anl = anldata.select(name='Temperature',level=700)[0].values
t700_ges = gesdata.select(name='Temperature',level=700)[0].values
t700 = t700_anl - t700_ges

# 500 hPa temperature
t500_anl = anldata.select(name='Temperature',level=500)[0].values
t500_ges = gesdata.select(name='Temperature',level=500)[0].values
t500 = t500_anl - t500_ges

# 250 hPa temperature
t250_anl = anldata.select(name='Temperature',level=250)[0].values
t250_ges = gesdata.select(name='Temperature',level=250)[0].values
t250 = t250_anl - t250_ges

# 850/700/500 hPa specific humidity
spfh850_anl = anldata.select(name='Specific humidity',level=850)[0].values * 1000
spfh700_anl = anldata.select(name='Specific humidity',level=700)[0].values * 1000
spfh500_anl = anldata.select(name='Specific humidity',level=500)[0].values * 1000
#if tmmark == 'tm06':	# Q fields are in GDAS pgrb2 file at tm06
#  spfh850_ges = gesdata2.select(name='Specific humidity',level=850)[0].values * 1000
#  spfh700_ges = gesdata2.select(name='Specific humidity',level=700)[0].values * 1000
#  spfh500_ges = gesdata2.select(name='Specific humidity',level=500)[0].values * 1000
#else:
spfh850_ges = gesdata.select(name='Specific humidity',level=850)[0].values * 1000
spfh700_ges = gesdata.select(name='Specific humidity',level=700)[0].values * 1000
spfh500_ges = gesdata.select(name='Specific humidity',level=500)[0].values * 1000
spfh850 = spfh850_anl - spfh850_ges
spfh700 = spfh700_anl - spfh700_ges
spfh500 = spfh500_anl - spfh500_ges

# 850 hPa wind speed
u850_anl = anldata.select(name='U component of wind',level=850)[0].values
v850_anl = anldata.select(name='V component of wind',level=850)[0].values
u850_ges = gesdata.select(name='U component of wind',level=850)[0].values
v850_ges = gesdata.select(name='V component of wind',level=850)[0].values
wspd850_anl = np.sqrt(u850_anl**2 + v850_anl**2)
wspd850_ges = np.sqrt(u850_ges**2 + v850_ges**2)
wspd850 = wspd850_anl - wspd850_ges

# 700 hPa wind speed
u700_anl = anldata.select(name='U component of wind',level=700)[0].values
v700_anl = anldata.select(name='V component of wind',level=700)[0].values
u700_ges = gesdata.select(name='U component of wind',level=700)[0].values
v700_ges = gesdata.select(name='V component of wind',level=700)[0].values
wspd700_anl = np.sqrt(u700_anl**2 + v700_anl**2)
wspd700_ges = np.sqrt(u700_ges**2 + v700_ges**2)
wspd700 = wspd700_anl - wspd700_ges

# 500 hPa wind speed
u500_anl = anldata.select(name='U component of wind',level=500)[0].values
v500_anl = anldata.select(name='V component of wind',level=500)[0].values
u500_ges = gesdata.select(name='U component of wind',level=500)[0].values
v500_ges = gesdata.select(name='V component of wind',level=500)[0].values
wspd500_anl = np.sqrt(u500_anl**2 + v500_anl**2)
wspd500_ges = np.sqrt(u500_ges**2 + v500_ges**2)
wspd500 = wspd500_anl - wspd500_ges

# 250 hPa wind speed
u250_anl = anldata.select(name='U component of wind',level=250)[0].values
v250_anl = anldata.select(name='V component of wind',level=250)[0].values
u250_ges = gesdata.select(name='U component of wind',level=250)[0].values
v250_ges = gesdata.select(name='V component of wind',level=250)[0].values
wspd250_anl = np.sqrt(u250_anl**2 + v250_anl**2)
wspd250_ges = np.sqrt(u250_ges**2 + v250_ges**2)
wspd250 = wspd250_anl - wspd250_ges


t2a = time.clock()
t3a = round(t2a-t1a, 3)
print(("%.3f seconds to read all messages") % t3a)

########################################
#    START PLOTTING FOR EACH DOMAIN    #
########################################

def main():

  for dom in domains:
    plot_all(dom)


def plot_all(dom):

  print(('Working on '+dom))

# Get the lats and lons
  grids = [anldata]
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

  Lat0 = anldata[1]['LaDInDegrees']
  Lon0 = anldata[1]['LoVInDegrees']
  print(Lat0)
  print(Lon0)


  # create figure and axes instances
  fig = plt.figure()
  gs = GridSpec(4,4,wspace=0.0,hspace=0.0)
  ax1 = fig.add_subplot(gs[:,:])
  axes = [ax1]
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

  # Create basemap instance and set the dimensions
  for ax in axes:
    if dom == 'conus':
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
    x_shift,y_shift   = m(lon_shift,lat_shift)
 
  # Map/figure has been set up here, save axes instances for use again later
    keep_ax_lst_1 = ax.get_children()[:]

    par += 1
  par = 1


################################
  # Plot lowest model level T
################################
  # Don't plot lowest model level fields for tm06 - not available in gdas files
  if tmmark != 'tm06':

    t1dom = time.clock()
    t1 = time.clock()
    print(('Working on lowest model level T for '+dom))

    units = 'K'
    clevs = [-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,5]
    cm = cmap_difcolors() 
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs1 = m.pcolormesh(x_shift,y_shift,tmphyb,cmap=cm,norm=norm,ax=ax)  
        cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
        cs1.cmap.set_under('darkblue')
        cs1.cmap.set_over('darkred')
        cbar1.set_label(units,fontsize=8)
        cbar1.ax.set_xticklabels([-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,5])
        cbar1.ax.tick_params(labelsize=7)
        ax.text(.5,1.04,'LAMDA Hybrid Level 1 Temperature ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+')',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('lamda_tlvl1_'+dom+'_'+tmmark+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot lowest model level T for: '+dom) % t3)


#################################
  # Plot lowest model level Q
#################################
    t1 = time.clock()
    print(('Working on lowest model level Q for '+dom))

  # Clear off old plottables but keep all the map info
    cbar1.remove()
    clear_plotables(ax1,keep_ax_lst_1,fig)

    units = 'g/kg'
    clevs = [-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1,1.5,2,2.5,3,4,5]
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs1 = m.pcolormesh(x_shift,y_shift,spfhhyb,cmap=cm,norm=norm,ax=ax)
        cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
        cs1.cmap.set_under('darkblue')
        cs1.cmap.set_over('darkred')
        cbar1.set_label(units,fontsize=8)
        cbar1.ax.set_xticklabels([-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1,1.5,2,2.5,3,4,5])
        cbar1.ax.tick_params(labelsize=7)
        ax.text(.5,1.04,'LAMDA Hybrid Level 1 Specific Humidity ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+')',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('lamda_qlvl1_'+dom+'_'+tmmark+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot lowest model level Q for: '+dom) % t3)


#################################
  # Plot lowest model level wind speed
#################################
    t1 = time.clock()
    print(('Working on lowest model level wind speed for '+dom))

  # Clear off old plottables but keep all the map info
    cbar1.remove()
    clear_plotables(ax1,keep_ax_lst_1,fig)

    units = 'm/s'
    skip = 80
    barblength = 4
    clevs = [-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,5]
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs1 = m.pcolormesh(x_shift,y_shift,wspdhyb,cmap=cm,norm=norm,ax=ax)
        cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
        cs1.cmap.set_under('darkblue')
        cs1.cmap.set_over('darkred')
        cbar1.set_label(units,fontsize=8)
        cbar1.ax.set_xticklabels([-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,5])
        cbar1.ax.tick_params(labelsize=7)
        m.barbs(lon[::skip,::skip],lat[::skip,::skip],uhyb_ges[::skip,::skip],vhyb_ges[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='fuchsia',ax=ax)
        m.barbs(lon[::skip,::skip],lat[::skip,::skip],uhyb_anl[::skip,::skip],vhyb_anl[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='black',ax=ax)
        ax.text(.5,1.06,'LAMDA Hybrid Level 1 Wind Speed ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+') \n Analysis(black), First Guess (pink)',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('lamda_wspdlvl1_'+dom+'_'+tmmark+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot lowest model level wind speed for: '+dom) % t3)


#################################
  # Plot surface pressure
#################################
  # Clear off old plottables but keep all the map info
    cbar1.remove()
    clear_plotables(ax1,keep_ax_lst_1,fig)

  # Start of tm06 plots
  t1 = time.clock()
  print(('Working on surface pressure for '+dom))

  units = 'hPa'
  clevs = [-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,5]
  cm = cmap_difcolors() 
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1 = m.pcolormesh(x_shift,y_shift,slp,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cs1.cmap.set_under('darkblue')
      cs1.cmap.set_over('darkred')
      cbar1.set_label(units,fontsize=8)
      cbar1.ax.set_xticklabels([-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,5])
      cbar1.ax.tick_params(labelsize=7)
      ax.text(.5,1.04,'LAMDA Surface Pressure ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+')',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('lamda_slp_'+dom+'_'+tmmark+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot surface pressure for: '+dom) % t3)


#################################
  # Plot precipitable water
#################################
  t1 = time.clock()
  print(('Working on precipitable water for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'mm'
  clevs = [-10,-8,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,8,10]
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1 = m.pcolormesh(x_shift,y_shift,pw,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cs1.cmap.set_under('darkblue')
      cs1.cmap.set_over('darkred')
      cbar1.set_label(units,fontsize=8)
      cbar1.ax.tick_params(labelsize=7)
      ax.text(.5,1.04,'LAMDA Precipitable Water ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+')',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('lamda_pw_'+dom+'_'+tmmark+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot precipitable water for: '+dom) % t3)


#################################
  # Plot geopotential height at 850 hPa
#################################
  t1 = time.clock()
  print(('Working on 850 hPa geopotential height for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'gpm'
  clevs = [-40,-30,-25,-20,-16,-12,-8,-4,0,4,8,12,16,20,25,30,40]
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1 = m.pcolormesh(x_shift,y_shift,z850,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cs1.cmap.set_under('darkblue')
      cs1.cmap.set_over('darkred')
      cbar1.set_label(units,fontsize=8)
      cbar1.ax.set_xticklabels([-40,-30,-25,-20,-16,-12,-8,-4,0,4,8,12,16,20,25,30,40])
      cbar1.ax.tick_params(labelsize=7)
      ax.text(.5,1.04,'LAMDA 850 hPa Geopotential Height ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+')',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('lamda_z850_'+dom+'_'+tmmark+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 850 hPa geopotential height for: '+dom) % t3)


#################################
  # Plot geopotential height at 700 hPa
#################################
  t1 = time.clock()
  print(('Working on 700 hPa geopotential height for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1 = m.pcolormesh(x_shift,y_shift,z700,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cs1.cmap.set_under('darkblue')
      cs1.cmap.set_over('darkred')
      cbar1.set_label(units,fontsize=8)
      cbar1.ax.set_xticklabels([-40,-30,-25,-20,-16,-12,-8,-4,0,4,8,12,16,20,25,30,40])
      cbar1.ax.tick_params(labelsize=7)
      ax.text(.5,1.04,'LAMDA 700 hPa Geopotential Height ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+')',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('lamda_z700_'+dom+'_'+tmmark+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 700 hPa geopotential height for: '+dom) % t3)


#################################
  # Plot geopotential height at 500 hPa
#################################
  t1 = time.clock()
  print(('Working on 500 hPa geopotential height for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1 = m.pcolormesh(x_shift,y_shift,z500,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cs1.cmap.set_under('darkblue')
      cs1.cmap.set_over('darkred')
      cbar1.set_label(units,fontsize=8)
      cbar1.ax.set_xticklabels([-40,-30,-25,-20,-16,-12,-8,-4,0,4,8,12,16,20,25,30,40])
      cbar1.ax.tick_params(labelsize=7)
      ax.text(.5,1.04,'LAMDA 500 hPa Geopotential Height ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+')',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('lamda_z500_'+dom+'_'+tmmark+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 500 hPa geopotential height for: '+dom) % t3)


#################################
  # Plot geopotential height at 250 hPa
#################################
  t1 = time.clock()
  print(('Working on 250 hPa geopotential height for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1 = m.pcolormesh(x_shift,y_shift,z250,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cs1.cmap.set_under('darkblue')
      cs1.cmap.set_over('darkred')
      cbar1.set_label(units,fontsize=8)
      cbar1.ax.set_xticklabels([-40,-30,-25,-20,-16,-12,-8,-4,0,4,8,12,16,20,25,30,40])
      cbar1.ax.tick_params(labelsize=7)
      ax.text(.5,1.04,'LAMDA 250 hPa Geopotential Height ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+')',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('lamda_z250_'+dom+'_'+tmmark+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 250 hPa geopotential height for: '+dom) % t3)


################################
  # Plot 850 hPa T
################################
  t1dom = time.clock()
  t1 = time.clock()
  print(('Working on 850 hPa temperature for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'K'
  clevs = [-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,5]
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1 = m.pcolormesh(x_shift,y_shift,t850,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cs1.cmap.set_under('darkblue')
      cs1.cmap.set_over('darkred')
      cbar1.set_label(units,fontsize=8)
      cbar1.ax.set_xticklabels([-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,5])
      cbar1.ax.tick_params(labelsize=7)
      ax.text(.5,1.04,'LAMDA 850 hPa Temperature ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+')',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('lamda_t850_'+dom+'_'+tmmark+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 850 hPa temperature for: '+dom) % t3)


################################
  # Plot 700 hPa T
################################
  t1dom = time.clock()
  t1 = time.clock()
  print(('Working on 700 hPa temperature for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1 = m.pcolormesh(x_shift,y_shift,t700,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cs1.cmap.set_under('darkblue')
      cs1.cmap.set_over('darkred')
      cbar1.set_label(units,fontsize=8)
      cbar1.ax.set_xticklabels([-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,5])
      cbar1.ax.tick_params(labelsize=7)
      ax.text(.5,1.04,'LAMDA 700 hPa Temperature ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+')',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('lamda_t700_'+dom+'_'+tmmark+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 700 hPa temperature for: '+dom) % t3)


################################
  # Plot 500 hPa T
################################
  t1dom = time.clock()
  t1 = time.clock()
  print(('Working on 500 hPa temperature for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1 = m.pcolormesh(x_shift,y_shift,t500,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cs1.cmap.set_under('darkblue')
      cs1.cmap.set_over('darkred')
      cbar1.set_label(units,fontsize=8)
      cbar1.ax.set_xticklabels([-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,5])
      cbar1.ax.tick_params(labelsize=7)
      ax.text(.5,1.04,'LAMDA 500 hPa Temperature ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+')',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('lamda_t500_'+dom+'_'+tmmark+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 500 hPa temperature for: '+dom) % t3)


################################
  # Plot 250 hPa T
################################
  t1dom = time.clock()
  t1 = time.clock()
  print(('Working on 250 hPa temperature for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1 = m.pcolormesh(x_shift,y_shift,t500,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cs1.cmap.set_under('darkblue')
      cs1.cmap.set_over('darkred')
      cbar1.set_label(units,fontsize=8)
      cbar1.ax.set_xticklabels([-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,5])
      cbar1.ax.tick_params(labelsize=7)
      ax.text(.5,1.04,'LAMDA 250 hPa Temperature ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+')',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('lamda_t250_'+dom+'_'+tmmark+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 250 hPa temperature for: '+dom) % t3)


################################
  # Plot 850 hPa Q
################################
  t1dom = time.clock()
  t1 = time.clock()
  print(('Working on 850 hPa Q for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'g/kg'
  clevs = [-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1,1.5,2,2.5,3,4,5]
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1 = m.pcolormesh(x_shift,y_shift,spfh850,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cs1.cmap.set_under('darkblue')
      cs1.cmap.set_over('darkred')
      cbar1.set_label(units,fontsize=8)
      cbar1.ax.set_xticklabels([-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1,1.5,2,2.5,3,4,5])
      cbar1.ax.tick_params(labelsize=7)
      ax.text(.5,1.04,'LAMDA 850 hPa Specific Humidity ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+')',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('lamda_q850_'+dom+'_'+tmmark+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 850 hPa Q for: '+dom) % t3)


################################
  # Plot 700 hPa Q
################################
  t1dom = time.clock()
  t1 = time.clock()
  print(('Working on 700 hPa Q for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1 = m.pcolormesh(x_shift,y_shift,spfh700,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cs1.cmap.set_under('darkblue')
      cs1.cmap.set_over('darkred')
      cbar1.set_label(units,fontsize=8)
      cbar1.ax.set_xticklabels([-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1,1.5,2,2.5,3,4,5])
      cbar1.ax.tick_params(labelsize=7)
      ax.text(.5,1.04,'LAMDA 700 hPa Specific Humidity ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+')',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('lamda_q700_'+dom+'_'+tmmark+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 700 hPa Q for: '+dom) % t3)


################################
  # Plot 500 hPa Q
################################
  t1dom = time.clock()
  t1 = time.clock()
  print(('Working on 500 hPa Q for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1 = m.pcolormesh(x_shift,y_shift,spfh500,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cs1.cmap.set_under('darkblue')
      cs1.cmap.set_over('darkred')
      cbar1.set_label(units,fontsize=8)
      cbar1.ax.set_xticklabels([-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1,1.5,2,2.5,3,4,5])
      cbar1.ax.tick_params(labelsize=7)
      ax.text(.5,1.04,'LAMDA 500 hPa Specific Humidity ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+')',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('lamda_q500_'+dom+'_'+tmmark+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 500 hPa Q for: '+dom) % t3)


#################################
  # Plot 850 hPa wind speed
#################################
  t1 = time.clock()
  print(('Working on 850 hPa wind speed for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'm/s'
  skip = 80
  barblength = 4
  clevs = [-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,5]
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1 = m.pcolormesh(x_shift,y_shift,wspd850,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cs1.cmap.set_under('darkblue')
      cs1.cmap.set_over('darkred')
      cbar1.set_label(units,fontsize=8)
      cbar1.ax.set_xticklabels([-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,5])
      cbar1.ax.tick_params(labelsize=7)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],u850_ges[::skip,::skip],v850_ges[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='fuchsia',ax=ax)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],u850_anl[::skip,::skip],v850_anl[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='black',ax=ax)
      ax.text(.5,1.06,'LAMDA 850 hPa Wind Speed ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+') \n Analysis(black), First Guess (pink)',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('lamda_wspd850_'+dom+'_'+tmmark+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 850 hPa wind speed for: '+dom) % t3)


#################################
  # Plot 700 hPa wind speed
#################################
  t1 = time.clock()
  print(('Working on 700 hPa wind speed for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1 = m.pcolormesh(x_shift,y_shift,wspd700,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cs1.cmap.set_under('darkblue')
      cs1.cmap.set_over('darkred')
      cbar1.set_label(units,fontsize=8)
      cbar1.ax.set_xticklabels([-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,5])
      cbar1.ax.tick_params(labelsize=7)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],u700_ges[::skip,::skip],v700_ges[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='fuchsia',ax=ax)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],u700_anl[::skip,::skip],v700_anl[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='black',ax=ax)
      ax.text(.5,1.06,'LAMDA 700 hPa Wind Speed ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+') \n Analysis(black), First Guess (pink)',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('lamda_wspd700_'+dom+'_'+tmmark+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 700 hPa wind speed for: '+dom) % t3)


#################################
  # Plot 500 hPa wind speed
#################################
  t1 = time.clock()
  print(('Working on 500 hPa wind speed for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1 = m.pcolormesh(x_shift,y_shift,wspd500,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cs1.cmap.set_under('darkblue')
      cs1.cmap.set_over('darkred')
      cbar1.set_label(units,fontsize=8)
      cbar1.ax.set_xticklabels([-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,5])
      cbar1.ax.tick_params(labelsize=7)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],u500_ges[::skip,::skip],v500_ges[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='fuchsia',ax=ax)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],u500_anl[::skip,::skip],v500_anl[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='black',ax=ax)
      ax.text(.5,1.06,'LAMDA 500 hPa Wind Speed ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+') \n Analysis(black), First Guess (pink)',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('lamda_wspd500_'+dom+'_'+tmmark+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 500 hPa wind speed for: '+dom) % t3)


#################################
  # Plot 250 hPa wind speed
#################################
  t1 = time.clock()
  print(('Working on 250 hPa wind speed for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1 = m.pcolormesh(x_shift,y_shift,wspd250,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cs1.cmap.set_under('darkblue')
      cs1.cmap.set_over('darkred')
      cbar1.set_label(units,fontsize=8)
      cbar1.ax.set_xticklabels([-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,5])
      cbar1.ax.tick_params(labelsize=7)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],u250_ges[::skip,::skip],v250_ges[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='fuchsia',ax=ax)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],u250_anl[::skip,::skip],v250_anl[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='black',ax=ax)
      ax.text(.5,1.06,'LAMDA 250 hPa Wind Speed ('+units+') \n '+itime+' cycle, Valid: '+vtime+' ('+tmmark+') \n Analysis(black), First Guess (pink)',horizontalalignment='center',fontsize=10,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('lamda_wspd250_'+dom+'_'+tmmark+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 250 hPa wind speed for: '+dom) % t3)



  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all variables for: "+dom) % t3dom)
  plt.clf()

######################################################


main()

