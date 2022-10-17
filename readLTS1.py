"""readLTS0.m

Decode and plot a PBOS syslog file for one sensor. 
Optionally, download the file from PBOS computer if it isn't in the
current directory; 
"""

from dataclasses import dataclass, field
import pandas as pd 
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns  
import pysftp
from tidal import *
from datetime import timedelta

# Seaborn plotting style                                                                                                               
sns.set_style("darkgrid")

@dataclass
class PBOSdata :
    fdir : str = "."
    fname : str = "lts_yyyy_mm_dd"
    df : pd.DataFrame = pd.DataFrame([])


    def getOneDay(self, yyyy,mm,dd, force_download=False) :

        # Create syslog filename from yyyy, mm, dd
        # Retrieve it via sftp if it isn't already been downloaded
        self.fname = f"lts_{yyyy:4d}_{mm:02d}_{dd:02d}.log"
        fpath = self.fdir + '/' + self.fname
        print(self.fname)
        if ((glob.glob(fpath) == []) or force_download):
            print(f'Getting file {self.fname} from PBOS')
            host = '107.190.208.42'
            # host = '10.123.123.28'
            port = 10022
            username='rnd20220701'
            password = 'uvaf2022again'
            remotePath = './'+self.fname
            localPath = remotePath
            cnopts = pysftp.CnOpts()
            cnopts.hostkeys = None 
            print(remotePath)
            with pysftp.Connection(host=host, port=port, username=username, password=password, cnopts=cnopts) as sftp:
                print("Connection succesfully established ... ")
                sftp.get(remotePath, localPath)

        # Read the CSV file.  Note use of \s+ to accept both tab and blank separators. (Skip first line if ntp syslog entry)
        skiprows = int('ntp' in next(open(fpath)))
        self.df = pd.read_csv(fpath, sep='\s+', header=None, engine='python', on_bad_lines='warn', skiprows=skiprows)

        # For some reason, all columns are categorical.  Only first 5 columns [0:4] should be categorical.
        # Others columsn [5:] should be numeric.
        cat_columns = self.df.select_dtypes(['object']).columns[5:]
        self.df[cat_columns] = self.df[cat_columns].apply(lambda x: pd.to_numeric(x, errors='coerce'))

        return self 

    def getDays(self, start_date, end_date, force_download=False) :

        dfall = pd.DataFrame([])

        startdt = pd.to_datetime(start_date)
        enddt = pd.to_datetime(end_date)
        day_count = (enddt - startdt).days

        for singledt in (startdt + timedelta(n) for n in range(day_count)):


            self = self.getOneDay(singledt.year, singledt.month, singledt.day, force_download)
            dfall = pd.concat([dfall, self.df])
            print(self.df.shape)
        
        self.df = dfall

        print()
        return self


    def getSensor(self, tag) :

        match tag:
            case 'FLBBFLRT2K-7341' :
                # Sometimes the syslog entry has more than 12 columns. Drop those columns.
                Ncols = self.df.shape[1]
                if (Ncols > 12) :
                    self.df.drop(range(12,Ncols), axis=1, inplace=True)
                
                # Named columns for this sensor tag
                self.df.columns = ['date', 'time', 'host', 'port', 'tag', 'em1', 'chl470', 'em2', 'bb700', 'em3', 'chl435', 'temp']
                
            case _:
                print(f'Unsupported tag: {tag}')
        
        # Add datetime = date + time and sort
        self.df['dt'] = pd.to_datetime(self.df.date + ' ' + self.df.time, utc=False)
        self.df.sort_values(by=['dt'], inplace=True)

        # Extract data for the desired sensor tag
        dfsensor = self.df[self.df.tag == tag].reset_index(drop=True)

        return dfsensor

    

def main() :

    # Sensor tag
    tag = 'FLBBFLRT2K-7341'

    # Instantiate a PBOSdata object
    pbd = PBOSdata('.')
    # dfsensor = pbd.getOneDay(2022,7,15,force_download=False).getSensor(tag)

    start_date = '20220720'
    end_date = '20220721'
    dfsensor = pbd.getDays(start_date, end_date, force_download=False).getSensor(tag)

    # print(dfsensor.chl470.shape)
    # print(np.min(dfsensor.chl470))
    # print(np.max(dfsensor.chl470))
    # print(pbd.df.tail())
    # print(dfsensor.tail())

   
    # plot it
    window=31
    hf = plt.figure(1)
    sns.scatterplot(data=dfsensor, x='dt', y='chl435', marker='.', edgecolor='none', s=3, color='blue')
    sns.scatterplot(data=dfsensor, x='dt', y='chl470', marker='.', edgecolor='none', s=3, color='green')
    sns.scatterplot(data=dfsensor, x='dt', y='bb700',  marker='.', edgecolor='none', s=3, color='red')
    
    sns.lineplot(x=dfsensor.dt, y=dfsensor.chl435.rolling(window).median().rolling(window).mean(), color='black')
    sns.lineplot(x=dfsensor.dt, y=dfsensor.chl470.rolling(window).median().rolling(window).mean(), color='black')
    sns.lineplot(x=dfsensor.dt, y=dfsensor.bb700.rolling(window).median().rolling(window).mean(), color='black')
    hl = plt.legend(['chl435', 'chl470', 'bb700'], markerscale=5.)
    plt.title(tag)
    plt.ylabel('Counts')
    plt.xlabel('Date Hr')
    plt.draw()


     # Get Tidal (Water Level) data from South Beach
    station = '9435380'
    startdt = dfsensor.dt.iloc[0]
    enddt = dfsensor.dt.iloc[-1]
    begin_date = startdt.strftime('%Y%m%d')
    end_date   = enddt.strftime('%Y%m%d')
    scaleWaterLevel = 1000

    dftide = tidalGetWaterLevel(station, begin_date, end_date)
    dftide.WaterLevel = dftide.WaterLevel * scaleWaterLevel

    sns.lineplot(data= dftide, x='datetime', y='WaterLevel', color='orange')
    plt.draw()

    # Show the plots
    plt.show()


if (__name__ == "__main__") :
    main()

