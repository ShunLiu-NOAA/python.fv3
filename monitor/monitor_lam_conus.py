#################### PURPOSE OF SCRIPT ##########################
#This script functions by accessing the "out" file from a regional fv3-cam forecast run and determining the locations, heights, and values where warning flags are being issued.  This script has the capability of determining if the model is failing over regions of convection or at the boundaries

#################### REQUIREMENTS ###############################
#  1.  This script as well as ncepy.py must be located within the directory where the out file is being stored.  the out file, which in the fv3-cam forecast runs is called "out", may be named different from other out files.  It must be the file that includes Warn_K messages.
#  2.  The only way this script will work is if range_warn is set to true inside the input.nml. 
#  3.  This script is set up to run with python 2.  You must use that version since it has Basemap.  Currently, cartopy is not function on platforms like hera and wcoss.  
#  4.  Script currently assumes the use of dynf and phyf files.  In the future, one would want to generalize this step an specify the outfile
#  5.  Script uses the halo files and assumes the location of the file is in directory+'/INPUT'.  
################### FUNCTION ##############################
# If both this script and ncepy.py are copied over to the location of where the outfile is, the outfile is located with the model output, the right version of python is being used, and range_warn is set to true in the input.nml, then one can simply run "python Geographical_Instability_Locator.py /path/to/top/directory DATE outfile" whereupon a .png file with a geographical distribution of warning flags are created.  

# The benefit of this script is that it can identify where (in 3D space), when, and what field is exceeding the expected range.  It is able to determine, for example, if warnings are occurring over convective areas or on/near the boundaries.

################## PYTHON VERSION #######################
# In your .tcshrc (or .bashrc) set the following:
#    module use /contrib/modulefiles
#    module load anaconda/2.7.10
#    
#    Issue module purge after making this change followed by either tcsh or bash

################# DEVELOPMENT HISTORY ##################
# This script was originially developed by Edward Strobach to investigate instability issues in the fv3-cam model
# Script was originally created on 09/2019
# Modified slightly to include user input: 11/12/2020 

######Basic Input #######
import matplotlib
matplotlib.use('Agg')
import re
import matplotlib.pyplot as plt
import mpl_toolkits
mpl_toolkits.__path__.append('/gpfs/dell2/emc/modeling/noscrub/gwv/py/lib/python/basemap-1.2.1-py3.6-linux-x86_64.egg/mpl_toolkits/')
from mpl_toolkits.basemap import Basemap, cm, interp as BasemapInterp
import numpy as np
import ncepy
import os, errno
import sys, glob
from netCDF4 import Dataset
#directory=os.getcwd()
basedir=sys.argv[1]
DATE=sys.argv[2]
#directory=basedir+'/'+DATE
directory=basedir
logdir=basedir
#directory=sys.argv[1]
RES='3359'

#outfile='out'  ## SUBJECTED TO CHANGE
#outfile=sys.argv[3]
for name in glob.glob(basedir+'/OUTPUT*'):
  head,tail = os.path.split(name)
  print(tail)
outfile=tail

######Import Subroutines#######
def String_Search(string,File,out):
        el=string.find('(')
        if el!=-1:
                string=string[0:el]
        ex = []
        for i in re.finditer(string,File):
                ex.append(i.start())
        if out ==0:
                Ex = len(ex)
        elif out == 1:
                Ex = ex
        return Ex

def Failed_Hours(directory):
        Dynfiles=[]
        for j in range(len(os.listdir(directory))):
                if String_Search('dynf',os.listdir(directory)[j],0)==1:
                        Dynfiles.append(os.listdir(directory)[j])
        DynFiles=np.sort(Dynfiles)[-1]
        Failed_hour=int(DynFiles[String_Search('dynf',DynFiles,1)[0]+len('dynf'):String_Search('.nc',DynFiles,1)[0]])
        return Failed_hour

def Range_Warns(directory,datdir):
    data=open(directory+'/'+outfile,'r').readlines()
    pfull=Dataset(datdir+'/dynf000.nc').variables['pfull'][:]
    vertind=[]
    lon1=[]
    lon2=[]
    lon3=[]
    lon4=[]
    lat1=[]
    lat2=[]
    lat3=[]
    lat4=[]
    val=[]
    longname=[]
    fieldtype=[]
    n=0
    while n<len(data)-1:
        if String_Search('Warn_K',data[n],0)==1:
            spaceloc=String_Search(' ',data[n],1) 
            m=0
            while spaceloc[m+1]-spaceloc[m]==1:
                m=m+1
            vertind.append(pfull[int(data[n][spaceloc[m]:spaceloc[m+1]])-1])
            Test=data[n][String_Search('lat',data[n],1)[0]+5::]  
            spaces=String_Search(' ',Test,1)
#            lon.append(float(Test[0:spaces[0]]))
#            lat.append(float(Test[spaces[np.where(np.diff(spaces)>1)[0][0]]:spaces[np.where(np.diff(spaces)>1)[0][0]+1]]))
            if String_Search(' VA ',data[n],0)==1:
                val.append(float(data[n][String_Search(' VA ',data[n],1)[0]+6:len(data[n])-1]))
                longname.append('Meridional Wind (m/s)')
                fieldtype.append('VA')
                lon4.append(float(Test[0:spaces[0]]))
                lat4.append(float(Test[spaces[np.where(np.diff(spaces)>1)[0][0]]:spaces[np.where(np.diff(spaces)>1)[0][0]+1]]))

            elif String_Search(' UA ',data[n],0)==1:
                val.append(float(data[n][String_Search(' UA ',data[n],1)[0]+6:len(data[n])-1]))
                longname.append('Zonal Wind (m/s)')
                fieldtype.append('UA')
                lon3.append(float(Test[0:spaces[0]]))
                lat3.append(float(Test[spaces[np.where(np.diff(spaces)>1)[0][0]]:spaces[np.where(np.diff(spaces)>1)[0][0]+1]]))

            elif String_Search(' TA ',data[n],0)==1:
                val.append(float(data[n][String_Search(' TA ',data[n],1)[0]+6:len(data[n])-1]))
                longname.append('Temperature (K)')
                fieldtype.append('TA')
                lon1.append(float(Test[0:spaces[0]]))
                lat1.append(float(Test[spaces[np.where(np.diff(spaces)>1)[0][0]]:spaces[np.where(np.diff(spaces)>1)[0][0]+1]]))

            elif String_Search('W_dyn',data[n],0)==1 or String_Search('TA_dyn',data[n],0)==1:
                val.append(float(data[n][String_Search('_dyn',data[n],1)[0]+6:len(data[n])-1]))
                if String_Search('W_dyn',data[n],0)==1:
                    longname.append('Vertical Velocity (m/s)')
                    fieldtype.append('W')
                    lon2.append(float(Test[0:spaces[0]]))
                    lat2.append(float(Test[spaces[np.where(np.diff(spaces)>1)[0][0]]:spaces[np.where(np.diff(spaces)>1)[0][0]+1]]))

                elif String_Search('TA_dyn',data[n],0)==1:
                    longname.append('Temperature (K)')
                    fieldtype.append('TA')
                    lon1.append(float(Test[0:spaces[0]]))
                    lat1.append(float(Test[spaces[np.where(np.diff(spaces)>1)[0][0]]:spaces[np.where(np.diff(spaces)>1)[0][0]+1]]))

            else:
                val.append(float('nan'))
                longname.append(float('nan'))
                fieldtype.append(float('nan'))
        n=n+1
    return np.array(lon1),np.array(lon2),np.array(lon3),np.array(lon4),np.array(lat1),np.array(lat2),np.array(lat3),np.array(lat4),np.array(vertind),np.array(val),np.array(longname),np.array(fieldtype)

max_hour=Failed_Hours(directory)
if String_Search('Warn_K',open(logdir+'/'+outfile,'r').read(),0)>0:
        lon1,lon2,lon3,lon4,lat1,lat2,lat3,lat4,pres,val,longname,fieldtype=Range_Warns(logdir,directory)
        griddir=directory+'/INPUT'
        xx=Dataset(griddir+'/C'+RES+'_grid.tile7.halo3.nc').variables['x'][:]
        yy=Dataset(griddir+'/C'+RES+'_grid.tile7.halo3.nc').variables['y'][:]
#        llcrnrlon,llcrnrlat,urcrnrlon,urcrnrlat,res=ncepy._default_corners_res('CONUS2')
        llcrnrlon=-123.5
        llcrnrlat=21.0
        urcrnrlon=-61.0
        urcrnrlat=48.0
        lat_0=35.4
        lon_0=-97.6
#        lat_ts=30.0
#        lon_0=np.mean([llcrnrlon,urcrnrlon])
#        lat_0=np.mean([llcrnrlat,urcrnrlat])
        ### GEOGRAPHICAL OVERLAYS ###
        fig=plt.figure(figsize=(11,8))
        ax=fig.add_subplot(111)
        m = Basemap(ax=ax,projection='gnom',lat_0=lat_0,lon_0=lon_0,\
                    llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                    llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                    resolution='l')
        m.drawmapboundary()
        m.fillcontinents(color='LightGrey')
        dx=10
        parallels = np.arange(-80.,90.,dx)
        meridians = np.arange(0.,360.,dx)
        m.drawcoastlines(linewidth=1.25)
        m.drawstates(linewidth=1.25)
        m.drawcountries(linewidth=1.25)
        m.drawparallels(parallels,labels=[1,0,0,0],fontsize=16)
        m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=16)
        xt_n,yt_n=m(xx[:,:],yy[:,:])
        m.plot(xt_n[:,0],yt_n[:,0],color='k',linewidth=3)
        m.plot(xt_n[:,-1],yt_n[:,-1],color='k',linewidth=3)
        m.plot(xt_n[0,:],yt_n[0,:],color='k',linewidth=3)
        m.plot(xt_n[-1,:],yt_n[-1,:],color='k',linewidth=3)

# Temperature warnings - fieldtype='TA'
        XTN1,YTN1=m(lon1[:],lat1[:])
        count1 = len(XTN1)
        print(count1)
        marker1, = m.plot(XTN1,YTN1,color='r',marker='o',linestyle='none',alpha=0.65)

# Vertical velocity warnings - fieldtype='W'
        XTN2,YTN2=m(lon2[:],lat2[:])
        count2 = len(XTN2)
        print(count2)
        marker2, = m.plot(XTN2,YTN2,color='k',marker='o',linestyle='none',alpha=0.65)

# Zonal wind warnings - fieldtype='UA'
        XTN3,YTN3=m(lon3[:],lat3[:])
        count3 = len(XTN3)
        print(count3)
        marker3, = m.plot(XTN3,YTN3,color='b',marker='o',linestyle='none',alpha=0.65)

# Meridional wind warnings - fieldtype='VA'
        XTN4,YTN4=m(lon4[:],lat4[:])
        count4 = len(XTN4)
        print(count4)
        marker4, = m.plot(XTN4,YTN4,color='g',marker='o',linestyle='none',alpha=0.65)

        plt.title(DATE,loc='left')
        plt.legend((marker1,marker2,marker3,marker4),('Temperature ('+str(count1)+')','Vertical Velocity ('+str(count2)+')','Zonal Wind ('+str(count3)+')','Meridional Wind ('+str(count4)+')'),loc='lower left')
        ax.text(.5,1.01,'Geographical Locations of Numerical Instabilities',horizontalalignment='center',fontsize=12,transform=ax.transAxes)
        plt.savefig('Locations_lam.png', format='png')
#        plt.savefig('/home/Edward.Strobach/Geographical_Locations_of_Numerical_Instabilities.png')
        plt.close()
        ### HEIGHT LOCATIONS/VALUE WARNINGS
        unique_fields=np.unique(fieldtype)
        unique_longname=np.unique(longname)
        for i in range(len(unique_fields)):
            if unique_fields[i]!='nan':
                inds=np.where(fieldtype==unique_fields[i])[0]
                fig=plt.figure(figsize=(11,9))
                ax=plt.gca()
                plt.scatter(val[inds],pres[inds],color='b')
                ax.invert_yaxis()
                if unique_fields[i]=='TA':
                    longname='Temperature (K)'
                elif unique_fields[i]=='UA':
                    longname='Zonal Wind (m/s)'
                elif unique_fields[i]=='VA':
                    longname='Meridional Wind (m/s)'
                elif unique_fields[i]=='W':
                    longname='Vertical Velocity (m/s)'
                plt.xlabel(longname)
                plt.ylabel('Pressure (mb)')
                plt.title(DATE,loc='left')
                plt.savefig(unique_fields[i]+'_lam.png', format='png')
                plt.close()        

else:
        print("Unable to determine if numerical instability occurred based on 'out' file")
