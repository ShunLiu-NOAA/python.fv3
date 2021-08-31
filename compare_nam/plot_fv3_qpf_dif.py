import matplotlib
matplotlib.use('Agg')
import io
import matplotlib.pyplot as plt
#from PIL import Image
import matplotlib.image as image
from matplotlib.gridspec import GridSpec
import mpl_toolkits
mpl_toolkits.__path__.append('/gpfs/dell2/emc/modeling/noscrub/gwv/py/lib/python/basemap-1.2.1-py3.6-linux-x86_64.egg/mpl_toolkits/')
from mpl_toolkits.basemap import Basemap
import numpy as np
import pygrib, datetime, os, sys, subprocess
from netCDF4 import Dataset


def clear_plotables(ax,keep_ax_lst,fig):
  #### - step to clear off old plotables but leave the map info - ####
  if len(keep_ax_lst) == 0:
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
#  plt.savefig(ram, format='png', bbox_inches='tight', dpi=150)
  plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
#  ram.seek(0)
#  im = Image.open(ram)
#  im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
#  im2.save(filename, format='PNG')


ymdh = str(sys.argv[1])
dom = str(sys.argv[2])

ymd = ymdh[0:8]
year = int(ymdh[0:4])
month = int(ymdh[4:6])
day = int(ymdh[6:8])
hour = int(ymdh[8:10])
cyc = str(hour).zfill(2)
print(year, month, day, hour)

im = image.imread('/gpfs/dell2/emc/modeling/noscrub/Benjamin.Blake/python.fv3/noaa.png')

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
  xscale=0.14
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

qpf_2 = 0
#fhours = [1,2,3,4,5,6]
fhours = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]
dtime = datetime.datetime(year,month,day,hour,0)
date_list = [dtime + datetime.timedelta(hours=x) for x in fhours]
print(date_list)

# Create the figure
fig = plt.figure()
gs = GridSpec(9,9,wspace=0.0,hspace=0.0)
ax1 = plt.subplot(gs[0:4,0:4])
ax2 = plt.subplot(gs[0:4,5:])
ax3 = plt.subplot(gs[5:,1:8])
axes = [ax1, ax2, ax3]
par = 1

# Setup map corners for plotting.

for ax in axes:
  if dom == 'BN' or dom == 'LA' or dom == 'SF' or dom == 'SP':
    m = Basemap(ax=ax,projection='gnom',lat_0=lat_0,lon_0=lon_0,\
                llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                resolution='h')
  else:
    m = Basemap(ax=ax,projection='gnom',lat_0=lat_0,lon_0=lon_0,\
                llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                resolution='l')
  m.fillcontinents(color='LightGrey',zorder=0)
  m.drawcoastlines(linewidth=0.75)
  m.drawstates(linewidth=0.5)
  m.drawcountries(linewidth=0.5)
#  parallels = np.arange(0.,90.,10.)
#  m.drawparallels(parallels,labels=[1,0,0,0],fontsize=6)
#  meridians = np.arange(180.,360.,10.)
#  m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=6)



  # Map/figure has been set up here, save axes instances for use again later
  if par == 1:
    keep_ax_lst_1 = ax.get_children()[:]
  elif par == 2:
    keep_ax_lst_2 = ax.get_children()[:]
  elif par == 3:
    keep_ax_lst_3 = ax.get_children()[:]

  par += 1
par = 1


for j in range(len(date_list)):

  fhour = str(fhours[j]).zfill(2)
  fhr = int(fhour)
  fhrm1 = fhr - 1
  fhour1 = str(fhrm1).zfill(2)
  print('fhour '+fhour)

  data1 = pygrib.open('/gpfs/dell1/nco/ops/com/nam/prod/nam.'+str(ymd)+'/nam.t'+cyc+'z.conusnest.hiresf'+fhour+'.tm00.grib2')
  data1_m1 = pygrib.open('/gpfs/dell1/nco/ops/com/nam/prod/nam.'+str(ymd)+'/nam.t'+cyc+'z.conusnest.hiresf'+fhour1+'.tm00.grib2')
#  data1 = pygrib.open('/gpfs/dell3/stmp/Benjamin.Blake/fv3nam/'+ymdh+'/'+cyc+'/nam.t'+cyc+'z.conusnest.hiresf'+fhour+'.tm00.grib2')
#  data1_m1 = pygrib.open('/gpfs/dell3/stmp/Benjamin.Blake/fv3nam/'+ymdh+'/'+cyc+'/nam.t'+cyc+'z.conusnest.hiresf'+fhour1+'.tm00.grib2')
  data2 = pygrib.open('/gpfs/dell4/ptmp/emc.campara/fv3lam/fv3lam.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f'+fhour+'.grib2')

  if (fhr <= 3):
    qpf = data1.select(name='Total Precipitation',lengthOfTimeRange=fhr)[0].values * 0.0393701
    qpf_1 = qpf
  elif (fhr > 3) and (fhr % 3 == 1):
    qpf = data1.select(name='Total Precipitation',lengthOfTimeRange=1)[0].values * 0.0393701
    qpf_1 += qpf
  elif (fhr > 3) and (fhr % 3 == 2):
    qpf = data1.select(name='Total Precipitation',lengthOfTimeRange=2)[0].values * 0.0393701
    qpfm1 = data1_m1.select(name='Total Precipitation',lengthOfTimeRange=1)[0].values * 0.0393701
    qpf_1 += (qpf-qpfm1)
  elif (fhr > 3) and (fhr % 3 == 0):    
    qpf = data1.select(name='Total Precipitation',lengthOfTimeRange=3)[0].values * 0.0393701
    qpfm1 = data1_m1.select(name='Total Precipitation',lengthOfTimeRange=2)[0].values * 0.0393701
    qpf_1 += (qpf-qpfm1)
  qpf_2 = data2.select(name='Total Precipitation',lengthOfTimeRange=fhr)[0].values * 0.0393701
  qpf_dif = qpf_2 - qpf_1

  units = 'in'
  lat, lon = data1.select(name='Categorical snow',level=0)[0].latlons()
  if (fhr == 1):
    x,y = m(lon,lat)
    x2,y2 = m(lon,lat)

  clevs = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,7,10,15,20]
  clevsdif = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
  colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','deepskyblue','cyan','mediumpurple','mediumorchid','darkmagenta','darkred','crimson','orangered','darkorange','goldenrod','gold','yellow']
  difcolors = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','white','white','#EEEE00','#EEC900','darkorange','orangered','red']
  itime = dtime.strftime("%m/%d/%Y %Hz")
  vtime = date_list[j].strftime("%m/%d/%Y %Hz") 

  # Clear off old plottables but keep all the map info
  if (fhr > 1):
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)
    clear_plotables(ax3,keep_ax_lst_3,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.contourf(x,y,qpf_1,clevs,colors=colorlist,extend='max',ax=ax)
      cs_1.cmap.set_over('pink')
      if (fhr == 1):
        cbar = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=[0.1,0.5,1,1.5,2,3,5,10,20])
        cbar.set_label(units,fontsize=6)
        cbar.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'NAM Nest '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+str(fhours[j]).zfill(2)+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.contourf(x,y,qpf_2,clevs,colors=colorlist,extend='max',ax=ax)
      cs_2.cmap.set_over('pink')
      if (fhr == 1):
        cbar = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=[0.1,0.5,1,1.5,2,3,5,10,20])
        cbar.set_label(units,fontsize=6)
        cbar.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+str(fhours[j]).zfill(2)+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.contourf(x2,y2,qpf_dif,clevsdif,colors=difcolors,extend='both',ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      if (fhr == 1):
        cbar = m.colorbar(cs,ax=ax,location='bottom',pad=0.05)
        cbar.set_label(units,fontsize=6)
        cbar.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM - NAM Nest '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+str(fhours[j]).zfill(2)+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(boxstyle='square,pad=0.2',facecolor='white',alpha=0.85))    
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareqpf_'+dom+'_f'+fhour+'.png')

plt.close()

