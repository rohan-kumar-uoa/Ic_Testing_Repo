import numpy as np
import matplotlib.pyplot as plt
# import spekpy as sp
import os
import pandas as pd
# from datetime import date, time, datetime
from scipy.interpolate import interp1d
import time
# import datetime
from nptdms import TdmsFile
from uncertainties import unumpy as unp
from uncertainties import ufloat

from numpy import array, unique, log
from scipy.optimize import curve_fit
from uncertainties.unumpy import nominal_values, std_devs
# from matplotlib.dates import date2num
# from copy import deepcopy
from os.path import join as pathjoin
import datetime
from scipy.optimize import OptimizeWarning

import warnings
warnings.filterwarnings("error")
from PIL import Image
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid
from multimethod import multimethod
from typing import Union

def nom(arr):
    return nominal_values(arr)

def err(arr):
    return std_devs(arr)

prop_cycle = plt.rcParams['axes.prop_cycle']
color_sequence = prop_cycle.by_key()['color']
def get_colourlist(n):
    return color_sequence[:n]

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def powerlaw(x,n,ic):
    return (x/ic)**n

def extendedpowerlaw(i,n,ic, R, V0):
    return (i/ic)**n + i*R + V0
   
@multimethod
def split_and_average(arr:np.ndarray, secsize = 1):
    newarr = np.split(arr, len(arr)/secsize)
    return np.mean(newarr, axis = 1)

class power_fit():
    def __init__(self):
        self.n = np.array([])
        self.ic = np.array([])

    @classmethod
    def populate(cls, coffs):
        obj = cls()
        obj.n = coffs[0,:]
        obj.ic = coffs[1,:]
        return obj

    def __repr__(self):
        return "Power Fit Data\n" + f"ic: {self.ic}\n" + f"n: {self.n}"
    
    def to_array(self):
        return np.array([self.n, self.ic])
    
    def split_and_average(self, secsize = 1):
        arr = self.to_array()
        nrows, ncols = arr.shape
        splitted = np.array(np.split(arr, ncols/secsize, axis = 1))
        reduced = np.mean(splitted, axis = 2).T
        return self.__class__.populate(reduced)
    
    def crop_end(self, l = 1):
        arr = self.to_array()[:,:-1]
        new = self.populate(arr)
        return new

class extended_fit(power_fit):
    def __init__(self):
        super().__init__()
        self.R = np.array([])
        self.V0 = np.array([])

    def __repr__(self):
        return "Power Fit Data\n" + f"ic: {self.ic}\n" + f"n: {self.n}\n" + f"R: {self.R}\n" + f"V0: {self.V0}"

    @classmethod
    def populate(cls, coffs):
        obj = super(extended_fit, cls).populate(coffs)
        obj.R = coffs[2,:]
        obj.V0 = coffs[3,:]
        return obj

    def to_array(self):
        return np.array([self.n, self.ic, self.R, self.V0])

    def split_and_average(self, secsize = 1):
        arr = self.to_array()
        nrows, ncols = arr.shape
        splitted = np.array(np.split(arr, ncols/secsize, axis = 1))
        reduced = np.mean(splitted, axis = 2).T
        return self.__class__.populate(reduced)
    
    def crop_end(self, l = 1):
        arr = self.to_array()[:,:-1]
        new = self.populate(arr)
        return new
    
def solve_ic(foldername, ic_guess = 18.5):
    tdms_file = TdmsFile.read(foldername + "\\Experiment_DataFile.tdms")
    measured_data = tdms_file['Measured Data'].as_dataframe()
    measured_data['Run Number'] = measured_data['Run Number'].astype(float)
    measured_data['Current (A)'] = measured_data['Current (A)'].astype(float)
    measured_data['Voltage (uV/cm)'] = measured_data['Voltage (uV/cm)'].astype(float)

    t = measured_data['Date'] + ' ' + measured_data['Time']
    run_times = pd.to_datetime(t).dt.tz_localize("Pacific/Auckland")
    measured_data['run_times'] = run_times
    AbsTime = measured_data.groupby('Run Number')['run_times'].max()
    RelTime = np.array((AbsTime-AbsTime.iloc[0]).dt.total_seconds())
    AbsTime = np.array(AbsTime)

    power_params = []
    extended_params = []
    for name, df in tqdm(measured_data[['Run Number', 'Voltage (uV/cm)', 'Current (A)']].groupby('Run Number'), desc = f'Fitting Ic curves in {foldername}'):
        if len(df['Current (A)']) < 2:
            newcoffs = np.full(2, ufloat(np.nan, np.nan))
            power_params.append(newcoffs)
            newcoffs = np.full(4, ufloat(np.nan, np.nan))
            extended_params.append(newcoffs)
            continue
        
        p0 = [ic_guess,19.5]
        try:
            popt, pcov = curve_fit(powerlaw, df['Current (A)'], df['Voltage (uV/cm)'], p0 = p0)
            perr = np.sqrt(np.diag(pcov))
            newcoffs = unp.uarray(popt,perr)
        except OptimizeWarning:
            newcoffs = np.full(2, ufloat(np.nan, np.nan))
        power_params.append(newcoffs)

        param_bounds = ([-np.inf, - np.inf, 0, -np.inf],[np.inf, np.inf, np.inf, np.inf])
        try:
            popt, pcov = curve_fit(extendedpowerlaw, df['Current (A)'], df['Voltage (uV/cm)'], p0 = np.concat((p0,[0,0])), bounds = param_bounds)
            perr = np.sqrt(np.diag(pcov))
            newcoffs = unp.uarray(popt,perr)
        except OptimizeWarning:
            newcoffs = np.full(4, ufloat(np.nan, np.nan))
        extended_params.append(newcoffs)

    power_params = np.array(power_params).transpose()
    extended_params = np.array(extended_params).transpose()
    return AbsTime, RelTime, power_params, extended_params

def populate_raw_data(foldername):
    tdms_file = TdmsFile.read(foldername + "\\Experiment_DataFile.tdms")
    measured_data = tdms_file['Measured Data'].as_dataframe()
    measured_data['Run Number'] = measured_data['Run Number'].astype(int)
    measured_data['Current (A)'] = measured_data['Current (A)'].astype(float)
    measured_data['Voltage (uV/cm)'] = measured_data['Voltage (uV/cm)'].astype(float)

    t = measured_data['Date'] + ' ' + measured_data['Time']
    run_times = pd.to_datetime(t).dt.tz_localize("Pacific/Auckland")
    measured_data['run_times'] = run_times
    AbsTime = measured_data.groupby('Run Number')['run_times'].max()
    RelTime = np.array((AbsTime-AbsTime.iloc[0]).dt.total_seconds())
    AbsTime = np.array(AbsTime)

    raw_current = []
    raw_voltage = []
    for name, df in measured_data[['Run Number', 'Voltage (uV/cm)', 'Current (A)']].groupby('Run Number'):
        raw_current.append(df['Current (A)'].to_numpy())
        raw_voltage.append(df['Voltage (uV/cm)'].to_numpy())
        pass

    return raw_current, raw_voltage

def concat_dict(d1,d2):
    if set(d1.keys()) != set(d2.keys()):
        raise ValueError("Dictionaries must have the same keys")

    combined = {}
    for k in d1.keys():
        combined[k] = np.concat([d1[k], d2[k]])

    return combined

def concat_fit_result(fit1 : power_fit, fit2 : power_fit):
    if type(fit1) != type(fit2):
        raise ValueError("power_fit instances must be same subclass")
    fittype = type(fit1)

    
    arr1 = fit1.to_array()
    arr2 = fit2.to_array()
    combined = np.concat([arr1, arr2], axis = 1)
    fitobj = fittype.populate(combined)
    return fitobj

def AbsTime_to_RelTime(abstimearr):
    """Convert an array of absolute timestamps into relative times
    
    Parameters:
        abstimearr (array[pd.Timestamp]): Array of absolute timestamps stored in pd.Timestamp format

    Returns:
        reltimearr (array[float]): Array of relative times as floats.    
        
    """
    if isinstance(abstimearr, np.ndarray):
        abstimearr = pd.Series(abstimearr)

    return np.array((abstimearr-abstimearr.iloc[0]).dt.total_seconds())

def RelTime_to_AbsTime(first_time, times):
    """Converts an array of relative times and a start time into absolute timestamps
    
    Parameters:
        first_time (pd.Timestamp): Absolute timestamp of first element
        times (array[float]): array of relative times

    Returns:
        tarr (numpy.ndarray[pd.Timestamp]): Array of absolute times  
        
    """
    tarr = pd.to_datetime(first_time.tz_localize(None).timestamp() + times,unit = 's')
    tarr = [t.tz_localize('Pacific/Auckland') for t in tarr]
    return np.array(tarr)

def dosage_from_tdms(actions = [], timings = [], fname = "", plot_bool = False):
    tdms_file = TdmsFile.read(fname + "\\Experiment_DataFile.tdms")
    measured_data = tdms_file['Measured Data']
    aggregated_data = tdms_file['Aggregated Data']
    temperature_data = tdms_file['Temperature Data']

    exp_start_time = pd.Timestamp(measured_data['Date'][0] + " " + measured_data['Time'][0][0:-8])
    exp_end_time = pd.Timestamp(measured_data['Date'][-1] + " " + measured_data['Time'][-1][0:-8])
    exp_dur = exp_end_time-exp_start_time
    df = pd.DataFrame({'time':[exp_start_time], 'V': [0.], 'i':[0.], 'duration':[0.], 'Pratio': [0.]})

    for i in range(len(actions)):
        if actions[i] == "monthly seasoning":
            sdf = pd.read_csv("monthly_seasoning.csv")
            seasoning_starttime = pd.Timestamp(timings[i][0])
            
            new_row = pd.DataFrame({'time': seasoning_starttime, "V": [0.], 'i':[0.], 'duration':[0.],'Pratio':[0.]})
            df = pd.concat([df, new_row], ignore_index=True)

            seasontimes = [seasoning_starttime + pd.Timedelta(seconds = sdf['runtime'][i]) for i in range(0,len(sdf['duration']))]
            seasondf = pd.DataFrame({'time': seasontimes, 'V': sdf['V'], 'i' : sdf['i'], 'Pratio' : sdf['Prat'], 'duration':sdf['duration']})

            new_row = pd.DataFrame({'time': seasoning_starttime, "V": [sdf['V'][0]], 'i':[sdf['i'][0]], 'duration':[0.],'Pratio':[sdf['Prat'][0]]})
            df = pd.concat([df, new_row], ignore_index=True)
            df = pd.concat([df, seasondf], ignore_index=True)


            season_endtime = df['time'][len(df)-1]
            new_row = pd.DataFrame({'time': season_endtime, "V": [0.], 'i':[0.], 'duration':[0.],'Pratio':[0.]})
            df = pd.concat([df, new_row], ignore_index=True)
        elif actions[i] == "constant":
            xray_start_time, xray_end_time = [pd.Timestamp(timings[i][j]) for j in range(len(timings[i]))]
            
            xraydf = pd.DataFrame({'time': [xray_start_time,xray_start_time,xray_end_time, xray_end_time], "V": [0.,320e3,320e3,0], 'i':[0., 10e-3,10e-3,0.], 'duration':[0.,0.,0.,0.],'Pratio':[0.,1.,1.,0.]})
            df = pd.concat([df, xraydf], ignore_index=True)

    new_row = pd.DataFrame({'time': exp_end_time, "V": [0.], 'i':[0.], 'duration':[0.],'Pratio':[0.]})
    df = pd.concat([df, new_row], ignore_index=True)

    timedifs = [df['time'][i+1]-df['time'][i] for i in range(len(df)-1)]
    timedifs = [timedifs[i].total_seconds() for i in range(len(timedifs))]
    durations = np.concat([[0.], timedifs])
    df['duration'] = durations

    df['runtime'] = np.cumsum(df['duration'])
    df['cumsum_Pratio'] = cumulative_trapezoid(df['Pratio'], df['runtime'], initial = 0)
    # plt.style.use('dark_background')

    if plot_bool:
        fig, ax1 = plt.subplots()
        ax1.plot(df['runtime']/3600, df['Pratio'], label = 'Instantaneous Dose')
        ax1.set(xlabel = 'Time (Hours)', ylabel = 'Fraction of Maximum Power')
        ax2 = ax1.twinx()
        ax2.plot(df['runtime']/3600, df['cumsum_Pratio'], 'r', label = 'Cumulative Dose')
        ax2.set(ylabel = "Cumulative Power Ratio")
        ax1.legend()
        ax2.legend()
        fig.tight_layout()


    df.to_csv(fname + "\\zdosagedata.csv", index = False)

    return df

class ic_run():
    def __init__(self):
        self.foldername = ""
        self.id = ""
        self.first_time = pd.Timestamp.now().tz_localize("Pacific/Auckland")
        self.times = np.array([])
        self.power = power_fit()
        self.extended = extended_fit()
        self.crittimes = []
        self.numrums = 0
        self.splitindex = []

    @classmethod
    def load_from_folder(cls, foldername, ic_guess = 18.5):
        obj = cls()
        obj.foldername = foldername
        obj.id = os.path.split(foldername)[1]
        AbsTime, RelTime, power_params, extended_params = solve_ic(foldername, ic_guess=ic_guess)
        obj.first_time, obj.times = (AbsTime[0], RelTime)
        obj.power = power_fit.populate(power_params)
        obj.extended = extended_fit.populate(extended_params)

        obj.crittimes = []
        obj.numruns = 1
        obj.splitindex = []
        print(f"Analysed Data at {obj.id}")
        return obj

    def __repr__(self):
        return f"Ic Test from {self.foldername}"

    def setid(self, newid):
        self.id = newid

    def __add__(self, other):
        combined = ic_run()
        combined.foldername = np.array([self.foldername, other.foldername])

        combined.id = (self.id, other.id)
        combined.times = np.concat([self.times, other.times+self.times[-1]])
        combined.first_time = min(self.first_time, other.first_time)

        combined.power = concat_fit_result(self.power, other.power)
        combined.extended = concat_fit_result(self.extended, other.extended)

        combined.crittimes = (self.crittimes, other.crittimes)
        combined.numruns = 2
        combined.splitindex = [len(self.times)]

        return combined

    def defaultplot(self, plot_type = "extended"):
        if plot_type == "extended":
            ic_array = self.extended.ic
        elif plot_type == "power":
            ic_array = self.power.ic

        plt.figure(figsize=1.5*np.array([4,3]))
        if self.numruns == 1:
            plt.errorbar(self.times/3600, nom(ic_array),yerr = err(ic_array),fmt = 'o-',markersize = 2,capsize=3,label = self.id)
        else:
            spl = lambda x: np.split(x, self.splitindex)
            for i,(time, ic) in enumerate(zip(spl(self.times), spl(ic_array))):
                plt.errorbar(time/3600, nom(ic),yerr = err(ic),fmt = 'o-',markersize = 2,capsize=3,label = self.id[i])

        plt.legend(loc = 'upper center',bbox_to_anchor=(0.67, 1.01)).get_frame().set_alpha(.9)
        plt.gcf().autofmt_xdate()
        plt.xlabel("Time (hours)")
        plt.ylabel("Ic (A)")
        return

    def load_raw_data(self):
        raw_current, raw_voltage = populate_raw_data(self.foldername)
        self.raw_current = raw_current
        self.raw_voltage = raw_voltage

# class ic_run_array(np.ndarray, ic_run):
#     def __init__(self, foldername=None):
#         super().__init__(foldername)

class data():
    """A class to store data recorded at different times.

    Attributes:
        vals (numpy.ndarray): array storing data values
        times (numpy.ndarray): array storing relative timestamps
        first_time (pd.Timestamp): absolute timestamp reference
    """
    def __init__(self, vals = np.array([]),
                  times = np.array([]),
                    first_time = pd.Timestamp.now()) -> None:
        """Initialises the data class

        Attributes:
            vals (numpy.ndarray): array storing data values
            times (numpy.ndarray): array storing relative timestamps
            first_time (pd.Timestamp): absolute timestamp reference
        """
        self.vals = vals
        self.times = times
        self.first_time = first_time
        pass

    def __repr__(self):
        return "Data Cluster"

class xray_ic_run(ic_run):
    def __init__(self):
        super().__init__()
        self.refillingtimes = []
        

    @classmethod
    def load_from_folder(cls, foldername, dosage_rate = 23.6, ic_guess = 18.5):
        obj = super(xray_ic_run,cls).load_from_folder(foldername, ic_guess = ic_guess)
        obj.set_doserate(dosage_rate)
        return obj
    
    def average_repeats(self,numrepeats = 1):
        self.power = self.power.split_and_average(numrepeats)
        self.power = self.power.crop_end()
        self.extended = self.extended.split_and_average(numrepeats)
        self.extended = self.extended.crop_end()
        self.times = self.times[2::3][:-1]

    def add_crittimes(self, timepairs, mode = 'absolute'):
        if mode == 'relative':
            self.crittimes =np.array(timepairs)
        elif mode == 'absolute':
            crittimes = []
            for tA, tB in timepairs:
                tA_reltime = pd.Timestamp(tA).tz_localize('Pacific/Auckland') - self.first_time
                tB_reltime = pd.Timestamp(tB).tz_localize('Pacific/Auckland') - self.first_time

                relative_timepair = (tA_reltime.total_seconds(),tB_reltime.total_seconds())
                crittimes.append(relative_timepair)

            self.crittimes = np.array(crittimes)

    def add_refillingtimes(self, timepairs, mode = 'absolute'):
        if mode == 'relative':
            self.refillingtimes = np.array(timepairs)
        elif mode == 'absolute':
            refillingtimes = []
            for t in timepairs:
                tA_reltime = pd.Timestamp(t).tz_localize('Pacific/Auckland') - self.first_time
                tA_seconds = (tA_reltime.total_seconds())
                refillingtimes.append(tA_seconds)

            self.refillingtimes = np.array(refillingtimes)

    def defaultplot(self, plot_type = "extended"):
        ic_array = getattr(self, plot_type).ic

        plt.figure(figsize=1.5*np.array([4,3]))
        plt.errorbar(self.times/3600, nom(ic_array),yerr = err(ic_array),fmt = 'go-',markersize = 2,capsize=3,label = self.id)

        if len(self.crittimes)>0:
            for i,(beg,end) in enumerate(self.crittimes):
                if i > 0:
                    xlab = None
                else:
                    xlab = 'Source On'
                plt.axvspan(beg/3600, end/3600, alpha=.3, color='red', label = xlab)

        xmin,xmax,ymin, ymax = plt.axis()
        plt.axis((xmin,xmax,ymin, ymax))

        if len(self.refillingtimes)>0:
            plt.vlines(self.refillingtimes/3600,ymin = ymin, ymax = ymax, label = "LN2 Refilled")

        plt.legend(loc = 'upper center',bbox_to_anchor=(0.67, 1.01)).get_frame().set_alpha(.9)
        plt.gcf().autofmt_xdate()
        plt.xlabel("Time (hours)")
        plt.ylabel("Ic (A)")
        return
    
    def set_doserate(self, dose = 23.7):
        self.dosrate = dose

    def load_dosages(self):
        try:
            df = pd.read_csv(os.path.join(self.foldername, 'zdosagedata.csv'))

            AbsTime = np.array(pd.to_datetime(df['time']).dt.tz_localize("Pacific/Auckland"))
            first_time = AbsTime[0]
            times = AbsTime_to_RelTime(AbsTime)
            cumul_doses = np.array(df['cumsum_Pratio'])*self.dosrate
            self.cumuldose = data(vals = cumul_doses, times = times, first_time=first_time)
            # print(first_time)
            # print(self.AbsTime[0])
            if self.first_time > first_time:
                tdelta = (self.first_time - first_time).total_seconds()
            else:
                tdelta = (first_time - self.first_time).total_seconds()
            # print(tdelta)
            dose_interp = interp1d(times, cumul_doses, fill_value = 'extrapolate')
            interped_doses = dose_interp(self.times + tdelta)
            self.interp_cumuldose = data(vals = interped_doses, times = self.times, first_time = first_time)
            assert len(self.interp_cumuldose.vals) == len(self.interp_cumuldose.times)
        except FileNotFoundError:
            print("Dosages have not been populated for this")

    def __add__(self, other):
        combined = xray_ic_run()
        combined.foldername = np.array([self.foldername, other.foldername])

        combined.id = (self.id, other.id)
        combined.times = np.concat([self.times, other.times+self.times[-1]])
        combined.first_time = min(self.first_time, other.first_time)

        combined.power = concat_fit_result(self.power, other.power)
        combined.extended = concat_fit_result(self.extended, other.extended)

        combined.crittimes = (self.crittimes, other.crittimes)
        combined.numruns = 2
        combined.splitindex = [len(self.times)]

        cumuldose = np.concat([self.cumuldose.vals, other.cumuldose.vals + self.cumuldose.vals[-1]])
        cumuldose_times = np.concat([self.cumuldose.times, other.cumuldose.times + self.cumuldose.times[-1]])
        cumuldose_first_time = self.cumuldose.first_time

        combined.cumuldose = data(vals = cumuldose, times = cumuldose_times, first_time=cumuldose_first_time)

        interp_cumuldose = np.concat([self.interp_cumuldose.vals, other.interp_cumuldose.vals + self.interp_cumuldose.vals[-1]])
        interp_cumuldose_times = np.concat([self.interp_cumuldose.times, other.interp_cumuldose.times + self.interp_cumuldose.times[-1]])
        interp_cumuldose_first_time = self.interp_cumuldose.first_time

        combined.interp_cumuldose = data(vals = interp_cumuldose, times = interp_cumuldose_times, first_time=interp_cumuldose_first_time)

        return combined

    def plotvdose(self, type = "power"):
        plt.figure()
        ic_arr = getattr(self, type).ic
        doses = []
        ic_vals = []

        for tA,tB in self.crittimes:
            for i in range(len(self.interp_cumuldose.times)):
                if tA <= self.interp_cumuldose.times[i] <= tB:
                    doses.append(self.interp_cumuldose.vals[i])
                    ic_vals.append(ic_arr[i])
        
        plt.errorbar(np.array(doses)/1e3, nom(ic_vals), yerr = err(ic_vals), label = self.id, fmt = 'o-',markersize = 2,capsize=3,)
        plt.legend()
        plt.xlabel("Dose (kGy)")
        plt.ylabel("Ic (A)") 
        plt.tight_layout()


def get_uncertain_mean(uarr):
    arr = [x for x in uarr if ~np.isnan(x.n)]
    # n = len(arr)
    # mn = sum([x.n/x.s**2 for x in arr])
    # err = np.sqrt(n/(sum([x.s**2])))
    return sum(arr)/len(arr)

def get_uncertain_mean2(uarr):
    arr = [x for x in uarr if ~np.isnan(x.n)]
    n = len(arr)
    mn = sum([x.n/x.s**2 for x in arr])/sum((1/x.s)**2 for x in arr)
    err = np.sqrt(n/(sum([(1/x.s)**2 for x in arr])))
    return ufloat(mn,err)