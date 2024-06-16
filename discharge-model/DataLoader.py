import os
import re
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt

from util import *

class DataLoader:
    """

    Convention: column names: t | vbl xx 1.0 yy 0.5 | t | vbl xx 1.0 yy 0.6 ...
    Names can be changed with init_column_names

    """
    def __init__(self, start_time, load_period, data_path=os.getcwd()) -> None:
        self.start_time = start_time
        self.fit_period = load_period
        self.data_path  = data_path

        self.init_column_names()
        self.init_data_frames()

        print("\nData loader working in %s" % self.data_path)

    def init_column_names(self, t="t", vbl="vbl", vwl="vwl", vdd="vdd", temp="temp", e_wr="e_wr", e_disch="e_disch", data="data") -> None:
        self.time_name = t
        self.vbl_name  = vbl
        self.vwl_name  = vwl
        self.vdd_name  = vdd
        self.temp_name = temp
        self.e_wr_name = e_wr
        self.w_disch_name = e_disch
        self.data_name = data

    def init_data_frames(self) -> None:
        self.df_bit_line       = pd.DataFrame()
        self.df_temperature    = pd.DataFrame()
        self.df_mismatch       = pd.DataFrame()
        self.df_mismatch_sigma = pd.DataFrame()
        self.df_power          = pd.DataFrame()

    def read_discharge(self, file, only_nominal_vdd=False, nominal_vdd=1.0, interpolate=False, interpolation_points=INTERPOLATION_POINTS):
        file_name = os.path.join(self.data_path, file)
        if not os.path.exists(file_name):
            print("Cannot read file %s: does not exist" % file_name)
            return
        df = pd.read_csv(file_name, na_values=['', ' '], dtype=float)
        df_fit = pd.DataFrame()

        for col in df:
            if not self.vbl_name in col:  # time column
                filter = (df[col] >= self.start_time) & (df[col] <  (self.start_time + self.fit_period))
                t = df[col].loc[filter].add(-self.start_time).to_numpy().reshape(-1, 1)
            else:  # column with vbl data (has to be after corresponding time column)
                float_start = col.index(' ', col.index(self.vwl_name))
                floats = re.findall(r"[-+]?(?:\d*\.*\d+)", col[float_start:])
                word_line = float(floats[0])

                float_start = col.index(' ', col.index(self.vdd_name))
                floats = re.findall(r"[-+]?(?:\d*\.*\d+)", col[float_start:])
                supply = float(floats[0])

                if word_line > supply:
                    continue

                if only_nominal_vdd and ((supply < nominal_vdd-0.001) or (supply > nominal_vdd+0.001)):
                    continue

                vbl = df[col].loc[filter].to_numpy().reshape(-1, 1)

                if interpolate:
                    t_interpolate = np.linspace(0, self.fit_period, interpolation_points)
                    vbl = np.interp(t_interpolate, t.flatten(), vbl.flatten()).reshape(-1, 1)
                    t = t_interpolate.reshape(-1, 1)

                vwl = word_line * np.ones(t.shape)
                vdd = supply * np.ones(t.shape)

                array = np.concatenate((t, vwl, vdd, vbl), axis=1)
                df_fit = pd.concat([df_fit, pd.DataFrame(array, columns=["t", "vwl", "vdd", "vbl"])], ignore_index=True)

        self.df_bit_line = df_fit
        print("Read %d discharge data points" % len(df_fit.index))

        t   = self.df_bit_line["t"].to_numpy()
        vwl = self.df_bit_line["vwl"].to_numpy()
        vdd = self.df_bit_line["vdd"].to_numpy()
        vbl = self.df_bit_line["vbl"].to_numpy()
        return t, vwl, vdd, vbl

    def read_temperature(self, file, nominal_vdd=1.0, interpolate=True, wl=True, dd=False):
        file_name = os.path.join(self.data_path, file)
        if not os.path.exists(file_name):
            print("Cannot read file %s: does not exist" % file_name)
            return
        df = pd.read_csv(file_name, na_values=['', ' '], dtype=float)
        df_fit = pd.DataFrame()

        for col in df:
            if not self.vbl_name in col:
                filter = (df[col] >= self.start_time) & (df[col] <  (self.start_time + self.fit_period))
                t = df[col].loc[filter].add(-self.start_time).to_numpy().reshape(-1, 1)
            else:  # column with vbl data (has to be after corresponding time column)
                if wl:
                    float_start = col.index(' ', col.index(self.vwl_name))
                    floats = re.findall(r"[-+]?(?:\d*\.*\d+)", col[float_start:])
                    word_line = float(floats[0])
                    if word_line > nominal_vdd:
                        continue

                float_start = col.index(' ', col.index(self.temp_name))
                floats = re.findall(r"[-+]?(?:\d*\.*\d+)", col[float_start:])
                temperature = float(floats[0])

                if dd:
                    float_start = col.index(' ', col.index(self.vdd_name))
                    floats = re.findall(r"[-+]?(?:\d*\.*\d+)", col[float_start:])
                    supply = float(floats[0])

                vbl = df[col].loc[filter].to_numpy().reshape(-1, 1)

                if interpolate:
                    t_interpolate = np.linspace(0, self.fit_period, INTERPOLATION_POINTS)
                    vbl = np.interp(t_interpolate, t.flatten(), vbl.flatten()).reshape(-1, 1)
                    t = t_interpolate.reshape(-1, 1)

                if wl:
                    vwl = word_line * np.ones(t.shape)
                else:
                    vwl = np.zeros(t.shape)
                temp = temperature * np.ones(t.shape)

                if dd:
                    vdd = supply * np.ones(t.shape)
                else:
                    vdd = np.ones(t.shape) * NOMINAL_VDD

                array = np.concatenate((t, vwl, temp, vbl, vdd), axis=1)
                df_fit = pd.concat([df_fit, pd.DataFrame(array, columns=["t", "vwl", "temp", "vbl", "vdd"])], ignore_index=True)
                # print(array)

        self.df_temperature = df_fit
        print("Read %d temperature data points" % len(df_fit.index))
        t   =  self.df_temperature["t"].to_numpy()
        vwl =  self.df_temperature["vwl"].to_numpy()
        temp = self.df_temperature["temp"].to_numpy()
        vbl =  self.df_temperature["vbl"].to_numpy()
        vdd =  self.df_temperature["vdd"].to_numpy()
        return t, vwl, vdd, vbl, temp

    def read_mismatch(self, dir, hist_step=0.025e-9, no_bins=21, min_hist_samples=120, debug=False, savepath=None):
        dir_name = os.path.join(self.data_path, dir)
        if not os.path.exists(dir_name):
            print("Cannot open dir %s: does not exist" % dir_name)
            return

        df_fit = pd.DataFrame(columns=["t", "vwl", "vbl"])

        files = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
        for fn in files:
            if debug:
                print(fn)
            floats = re.findall(r"[-+]?(?:\d*\.*\d+)", fn)
            word_line = np.abs(float(floats[0]))

            file_name = os.path.join(dir_name, fn)
            df = pd.read_csv(file_name, na_values=['', ' '], dtype=float)

            df_wl = pd.DataFrame(columns=["t", "vwl", "vbl"])

            df_t = df.filter(like='time').stack().reset_index()
            df_t.columns = ['a','b','t']
            df_v = df.filter(like='V_blb').stack().reset_index()
            df_v.columns = ['a','b','v']

            df_wl['t']= df_t['t']
            df_wl['vbl'] = df_v['v']

            if debug:
                print(df_wl.describe())
                print(df_t.describe())

            filter = (df_wl['t'] >= self.start_time) & (df_wl['t'] <  (self.start_time + self.fit_period))
            df_wl = df_wl.loc[filter]
            df_wl['t'] = df_wl['t'].add(-1.0 * self.start_time)

            t = df_wl['t'].to_numpy().reshape(-1, 1)
            vbl = df_wl['vbl'].to_numpy().reshape(-1, 1)

            vwl = word_line * np.ones(t.shape)
            array = np.concatenate((t, vwl, vbl), axis=1)

            if debug:
                print(array.shape)

            df_fit = pd.concat([df_fit, pd.DataFrame(array, columns=["t", "vwl", "vbl"])], ignore_index=True)

            if debug:
                print(len(df_fit.index))

        self.df_mismatch = df_fit
        print("Read %d mismatch data points from %d files" % (len(self.df_mismatch.index), len(files)))

        no_hists = int(np.floor(self.fit_period/hist_step))
        sampling_times = np.linspace(0.0, self.fit_period, no_hists)

        df_fit_sigma = pd.DataFrame()
        df_wl = pd.DataFrame()

        for vwl in self.df_mismatch["vwl"].unique():
            filter = (self.df_mismatch['vwl'] > vwl - 0.001)& (self.df_mismatch['vwl'] < vwl + 0.001)
            df_wl = self.df_mismatch.loc[filter]
            array = np.zeros([0, 3])
            for sampling_time in sampling_times:
                filter = (df_wl['t'] >= sampling_time) & (df_wl['t'] < sampling_time+hist_step)
                df_filtered = df_wl.loc[filter]
                sample = df_filtered['vbl'].to_numpy()

                if sample.size < min_hist_samples: # Filter out small samples
                    continue

                hist, bins = np.histogram(sample, no_bins)
                BIN_SPACING = (np.max(bins)-np.min(bins))/no_bins
                stddev = np.std(sample)

                if(savepath):
                    if(vwl == 1.0 and sampling_time > 1e-9 and sampling_time < 1.04e-9):
                        plt.figure(figsize=(9*CM, 8.0*CM))
                        plt.stairs(hist, edges=1000*bins)
                        plt.xlabel("Bit-line voltage $\sigma$ [mV]")
                        plt.ylabel("Number of occurrences")
                        plt.grid()
                        plt.savefig(savepath + "-normal.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
                        plt.show()

                    if(vwl == 0.3 and sampling_time > 1e-10 and sampling_time < 1.4e-10):
                        plt.figure(figsize=(9*CM, 8.0*CM))
                        plt.stairs(hist, edges=1000*bins)
                        plt.xlabel("Bit-line voltage $\sigma$ [mV]")
                        plt.ylabel("Number of occurrences")
                        plt.grid()
                        plt.savefig(savepath + "-border.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
                        plt.show()

                if(np.mean(sample) >= 0.95):
                    if debug:
                        print("vwl=%.2f, t=%.3e: avg=%.3f, hist max=%.3f, samples %d" % (vwl, sampling_time, np.mean(sample), np.mean(bins[np.argwhere(hist == np.amax(hist))]+ 0.5*BIN_SPACING) , len(sample)))
                    hist_max = np.mean(bins[np.argwhere(hist == np.amax(hist))]+ 0.5*BIN_SPACING)
                    sigma = np.sqrt(np.mean((sample[sample <= hist_max] - hist_max)**2))

                t = sampling_time + 0.5*hist_step
                sigma = stddev
                vwl = vwl

                array = np.append(array, np.array([[t, sigma, vwl]]), axis=0)

            df_fit_sigma = pd.concat([pd.DataFrame(array, columns=["t", "sigma", "vwl"]), df_fit_sigma], ignore_index=True)

        self.df_mismatch_sigma = df_fit_sigma
        print("Created %d histograms" % len(df_fit_sigma.index))
        t     =  self.df_mismatch_sigma["t"].to_numpy()
        vwl   =  self.df_mismatch_sigma["vwl"].to_numpy()
        sigma =  self.df_mismatch_sigma["sigma"].to_numpy()
        return t, vwl, sigma

    def read_energy(self, file_wr, file_disch, debug=False):
        file_name_wr = os.path.join(self.data_path, file_wr)
        if not os.path.exists(file_name_wr):
            print("Cannot read file %s: does not exist" % file_name_wr)
            return
        df_wr = pd.read_csv(file_name_wr, na_values=['', ' '], dtype=float)

        if debug:
            print(df_wr)

        file_name_disch = os.path.join(self.data_path, file_disch)
        if not os.path.exists(file_name_disch):
            print("Cannot read file %s: does not exist" % file_name_disch)
            return
        df_disch = pd.read_csv(file_name_disch, na_values=['', ' '], dtype=float)

        if debug:
            print(df_disch)

        df_fit = pd.DataFrame()

        for col in df_disch:
            if not self.vdd_name in col:
                float_start = col.index(' ', col.index(self.vwl_name))
                floats = re.findall(r"[-+]?(?:\d*\.*\d+)", col[float_start:])
                vwl = float(floats[0])

                float_start = col.index(' ', col.index(self.temp_name))
                floats = re.findall(r"[-+]?(?:\d*\.*\d+)", col[float_start:])
                temp = float(floats[0])

                float_start = col.index(' ', col.index(self.data_name))
                floats = re.findall(r"[-+]?(?:\d*\.*\d+)", col[float_start:])
                data = float(floats[0])

                for idx, val in df_disch[col].items():
                    vdd = df_disch.filter(like=self.vdd_name, axis=1).iloc[[idx]].to_numpy().item()
                    if vwl > vdd:
                        continue

                    e_disch = val
                    array = np.array([[data, vdd, vwl, temp, 0.0, e_disch]])
                    if debug:
                        print(array)
                    df_fit = pd.concat([df_fit, pd.DataFrame(array, columns=["data", "vdd", "vwl", "temp", "e_wr", "e_dc"])], ignore_index=True)

        for col in df_wr:
            if not self.vdd_name in col:
                float_start = col.index(' ', col.index(self.temp_name))
                floats = re.findall(r"[-+]?(?:\d*\.*\d+)", col[float_start:])
                temp = float(floats[0])

                float_start = col.index(' ', col.index(self.data_name))
                floats = re.findall(r"[-+]?(?:\d*\.*\d+)", col[float_start:])
                data = float(floats[0])

                for idx, val in df_wr[col].items():
                    vdd = df_disch.filter(like=self.vdd_name, axis=1).iloc[[idx]].to_numpy().item()
                    e_wr = val
                    filter = (df_fit['data']==data) & (df_fit['vdd']==vdd) & (df_fit['temp']==temp)

                    df_fit.loc[filter, 'e_wr'] = e_wr

        print("Read %d energy data points" % len(df_fit.index))
        self.df_power = df_fit

        data   = self.df_power["data"].to_numpy()
        vwl = self.df_power["vwl"].to_numpy()
        vdd = self.df_power["vdd"].to_numpy()
        temp = self.df_power["temp"].to_numpy()
        e_wr = self.df_power["e_wr"].to_numpy()
        e_dc = self.df_power["e_dc"].to_numpy()
        return data, vdd, vwl, temp, e_wr, e_dc
