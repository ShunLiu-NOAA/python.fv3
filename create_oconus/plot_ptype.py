import pygrib
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import io
import matplotlib.pyplot as plt
#from PIL import Image
import matplotlib.image as image
import mpl_toolkits
mpl_toolkits.__path__.append('/gpfs/dell2/emc/modeling/noscrub/gwv/py/lib/python/basemap-1.2.1-py3.6-linux-x86_64.egg/mpl_toolkits/')
from mpl_toolkits.basemap import Basemap
import subprocess
from matplotlib.gridspec import GridSpec
import sys, os


def clear_plotables(ax,keep_ax_lst,fig):
  #### - step to clear off old plotables but leave the map info - ####
  if len(keep_ax_lst) == 0:
    print("clear_plotables WARNING keep_ax_lst has length 0. Clearing ALL plottables including map info!")
  cur_ax_children = ax.get_children()[:]
  if len(cur_ax_children) > 0:
    for a in cur_ax_children:
      if a not in keep_ax_lst:
       # if thr artist isn't part of the initial set up, remove it
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


prodhex=['#F00000','#32B2BD','#32B2BD']
parahex=['#F00000','#C15033','#C15033']
snowhex=["#64B3E8", "#5197D7", "#3E7CC6", "#2B60B5", "#1945A4"]
rainhex=["#5EE240", "#4DC534", "#3DA828", "#2D8B1C", "#1D6F11"]
sleethex=["#947EEC", "#7F62CB", "#6B47AB", "#562B8A", "#42106A"]
freezehex=["#E65956", "#DF4A48", "#D93B3A", "#D22C2C", "#CC1E1E"]
mixhex=["#E75FD5", "#D54DBB", "#C33BA2", "#B12989", "#A01870"]

ymdh = str(sys.argv[1])
dom = str(sys.argv[2])

ymd=ymdh[0:8]
year=int(ymdh[0:4])
month=int(ymdh[4:6])
day=int(ymdh[6:8])
hour=int(ymdh[8:10])
cyc = str(hour).zfill(2)
print(year, month , day, hour)

im = image.imread('/gpfs/dell2/emc/modeling/noscrub/Benjamin.Blake/python.fv3/noaa.png')

# Map corners for each domain
if dom == 'ak':
  llcrnrlon = -175.0
  llcrnrlat = 50.0
  urcrnrlon = -110.0
  urcrnrlat = 70.0
  lat_0 = 90.0
  lon_0 = -150.0
  lat_ts = 60.0
  xscale=0.11
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

#fhours = [0,2,24]
fhours = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]
dtime=datetime.datetime(year,month,day,hour,0)
date_list = [dtime + datetime.timedelta(hours=x) for x in fhours]
print(date_list)

# Create the figure
fig = plt.figure()
gs = GridSpec(4,4,wspace=0.0,hspace=0.0)
ax1 = plt.subplot(gs[0:3,:])
axes = [ax1]
par = 1

# Setup map corners for plotting.

for ax in axes:
  if dom == 'ak':
    m = Basemap(ax=ax,projection='stere',lat_ts=lat_ts,lat_0=lat_0,lon_0=lon_0,\
                llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                rsphere=6371229,resolution='i')
  elif dom == 'hi':
    m = Basemap(ax=ax,projection='cyl',\
                llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                rsphere=6371229,resolution='h')
  elif dom == 'pr':
    m = Basemap(ax=ax,projection='cyl',\
                llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                rsphere=6371229,resolution='h')
  elif dom == 'namerica':
    m = Basemap(ax=ax,projection='stere',lat_ts=lat_ts,lat_0=lat_0,lon_0=lon_0,\
                llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,\
                urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,\
                rsphere=6371229,resolution='l')

  m.fillcontinents(color='LightGrey',zorder=0)
  m.drawstates(linewidth=.5,color='k')
  m.drawcoastlines(linewidth=.75, color='k')
  m.drawcountries(linewidth=.5, color='k')

  # Map/figure has been set up here, save axes instances for use again later
  if par == 1:
    keep_ax_lst_1 = ax.get_children()[:]

  par += 1
par = 1

for j in range(len(date_list)):
  ymd=dtime.strftime("%Y%m%d")
  fhour=date_list[j].strftime("%H")
  fhr = str(fhours[j]).zfill(2)
  prodind = pygrib.index('/gpfs/dell4/ptmp/emc.campara/fv3lam/fv3lam.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.'+dom+'.f'+fhr+'.grib2','name','level')
#  paraind = pygrib.index('/gpfs/dell6/ptmp/emc.campara/fv3lamx/fv3lamx.'+str(ymd)+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f'+fhour+'.grib2','name','level')

  refprod=np.asarray(prodind.select(name='Maximum/Composite radar reflectivity',level=0)[0].values)
  rainprod=np.asarray(prodind.select(name='Categorical rain',level=0)[0].values)
  frprod=np.asarray(prodind.select(name='Categorical freezing rain',level=0)[0].values)
  plprod=np.asarray(prodind.select(name='Categorical ice pellets',level=0)[0].values)
  snprod=np.asarray(prodind.select(name='Categorical snow',level=0)[0].values)

  typesprod=np.zeros(frprod.shape)
  typesprod[rainprod==1]=typesprod[rainprod==1]+1
  typesprod[frprod==1]=typesprod[frprod==1]+3
  typesprod[plprod==1]=typesprod[plprod==1]+5
  typesprod[snprod==1]=typesprod[snprod==1]+7
  rain1prod=np.copy(refprod)
  fr1prod=np.copy(refprod)
  pl1prod=np.copy(refprod)
  sn1prod=np.copy(refprod)
  mix1prod=np.copy(refprod)
  rain1prod[typesprod!=1]=-1
  fr1prod[typesprod!=3]=-1
  pl1prod[typesprod!=5]=-1
  sn1prod[typesprod!=7]=-1
  mix1prod[typesprod==0]=-1
  mix1prod[typesprod==1]=-1
  mix1prod[typesprod==3]=-1
  mix1prod[typesprod==5]=-1
  mix1prod[typesprod==7]=-1


  lats,lons=prodind.select(name='Categorical snow',level=0)[0].latlons()
  if (int(fhr) == 0):
    x,y=m(lons,lats)

  clevs=[0,10,20,30,40]
  clevs1=[20,1000]

  # Clear off old plotables but keep all the map info
  if (int(fhr) > 0):
    clear_plotables(ax1,keep_ax_lst_1,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      intime=dtime.strftime("%m/%d/%Y %HZ")
      vtime=date_list[j].strftime("%m/%d/%Y %HZ")
      csrain=m.contourf(x,y,rain1prod,clevs,colors=rainhex,extend='max',ax=ax)
      csmix=m.contourf(x,y,mix1prod,clevs,colors=mixhex,extend='max',ax=ax)
      cssnow=m.contourf(x,y,sn1prod,clevs,colors=snowhex,extend='max',ax=ax)
      cssleet=m.contourf(x,y,pl1prod,clevs,colors=sleethex,extend='max',ax=ax)
      csfrzra=m.contourf(x,y,fr1prod,clevs,colors=freezehex,extend='max',ax=ax)
      ax.text(.5,1.03,'FV3LAM composite reflectivity by ptype \n initialized: '+intime +' valid: '+ vtime + ' (F'+fhr+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)
#    elif par == 2:
#      csrain=m.contourf(x,y,rain1para,clevs,colors=rainhex,extend='max',ax=ax)
#      csmix=m.contourf(x,y,mix1para,clevs,colors=mixhex,extend='max',ax=ax)
#      cssnow=m.contourf(x,y,sn1para,clevs,colors=snowhex,extend='max',ax=ax)
#      cssleet=m.contourf(x,y,pl1para,clevs,colors=sleethex,extend='max',ax=ax)
#      csfrzra=m.contourf(x,y,fr1para,clevs,colors=freezehex,extend='max',ax=ax)
#      ax.text(.5,1.03,'FV3LAM composite reflectivity by ptype \n initialized: '+intime +' valid: '+ vtime + ' (F'+fhr+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=.85,boxstyle='square,pad=0.2'))
#      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      if (int(fhr) == 0):
        caxrain=fig.add_axes([.09,.2,.1,.03])
        cbrain=fig.colorbar(csrain,cax=caxrain,ticks=clevs,orientation='horizontal')
        cbrain.ax.tick_params(labelsize=5)
        cbrain.ax.set_xticklabels(['light rain','','','','heavy rain'])

        caxsnow=fig.add_axes([.27,.2,.1,.03])
        cbsnow=fig.colorbar(cssnow,cax=caxsnow,ticks=clevs,orientation='horizontal')
        cbsnow.ax.tick_params(labelsize=5)
        cbsnow.ax.set_xticklabels(['light snow','','','','heavy snow'])

        caxsleet=fig.add_axes([.45,.2,.1,.03])
        cbsleet=fig.colorbar(cssleet,cax=caxsleet,ticks=clevs,orientation='horizontal')
        cbsleet.ax.tick_params(labelsize=5)
        cbsleet.ax.set_xticklabels(['light sleet','','','','heavy sleet'])

        caxfrzra=fig.add_axes([.63,.2,.1,.03])
        cbfrzra=fig.colorbar(csfrzra,cax=caxfrzra,ticks=clevs,orientation='horizontal')
        cbfrzra.ax.tick_params(labelsize=5)
        cbfrzra.ax.set_xticklabels(['light freezing rain','','','','heavy freezing rain'])

        caxmix=fig.add_axes([.81,.2,.1,.03])
        cbmix=fig.colorbar(csmix,cax=caxmix,ticks=clevs,orientation='horizontal')
        cbmix.ax.tick_params(labelsize=5)
        cbmix.ax.set_xticklabels(['light mix','','','','heavy mix'])
    par += 1
  par = 1

  compress_and_save('comparetype_'+dom+'_f'+fhr+'.png')

plt.close()
