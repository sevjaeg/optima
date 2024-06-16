##!/usr/bin/python3

import math
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt

from util import *

DEBUG = False
PLOT  = True

DISCHARGE_START = 9.6e-9  # s after simulation start
FIT_TIME        = 2.4e-9  # s after discharge start

if PLOT:
	plt.rcParams.update({
		"text.usetex": True,
		"font.family": "libertine",
		"font.size":  11
	})

###############################################################################

df = pd.read_csv("spice-data/discharge-basic.csv", na_values=['', ' '], dtype=float)
df_voltages = pd.DataFrame()

t = df['t'].to_numpy().reshape(-1, 1)
vwl  = df['vwl'].to_numpy().reshape(-1, 1)
vbl  = df['vbl'].to_numpy().reshape(-1, 1)
vblb = df['vblb'].to_numpy().reshape(-1, 1)

array = np.concatenate((t, vwl, vbl, vblb), axis=1)
df_voltages = pd.concat([df_voltages, pd.DataFrame(array, columns=["t", "vwl", "vbl", "vblb"])], ignore_index=True)

if PLOT:
	plt.figure(figsize=(FIG_WIDTH_WIDE*CM, FIG_HEIGHT*CM))
	t_plot= df_voltages["t"].to_numpy()*1e9
	vwl_plot = df_voltages["vwl"].to_numpy()
	vbl_plot = df_voltages["vbl"].to_numpy()
	vblb_plot = df_voltages["vblb"].to_numpy()
	plt.plot(t_plot, vwl_plot, linestyle='--', marker='None', markersize=1, label="$V_{WL}$")
	# plt.plot(t_plot, vbl_plot, linestyle='-', marker='None', markersize=1, label="$V_{BL}$")
	plt.plot(t_plot, vblb_plot, linestyle='-', marker='None', markersize=1, label="$V_{\overline{BL}}$")

	plt.xlim(left=0.0, right=12.0)  # smooth end ()
	plt.xlabel(r"\bf$\bf t$ [ns]")
	plt.ylabel(r"\bf$\bf V$ [V]")
	plt.legend(loc='lower right')
	plt.grid()
	plt.show()

###############################################################################

df = pd.read_csv("spice-data/discharge-vdd.csv", na_values=['', ' '], dtype=float)

wl_voltages = np.array([0.3, 0.34, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.78, 0.82, 0.86, 0.9, 0.94, 0.98, 1., 1.02, 1.06, 1.1])
supply_voltages = np.array([0.9 , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.  , 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1 ])

print("Word line voltages: ", wl_voltages)
print("Supply voltages:    ", supply_voltages)

df_fit = pd.DataFrame()

for wl_voltage in wl_voltages:

	if math.isclose(wl_voltage, 0.3) or math.isclose(wl_voltage, 0.5) or math.isclose(wl_voltage, 0.7) or math.isclose(wl_voltage, 0.9) or math.isclose(wl_voltage, 1.1):
		wl_voltage_str = str(np.round(wl_voltage, 2))
	elif  math.isclose(wl_voltage, 1.0):
		wl_voltage_str = str(int(wl_voltage))
	else:
		wl_voltage_str = str(np.round(wl_voltage, 3))

	for supply_voltage in supply_voltages:
		if wl_voltage > supply_voltage:
			continue  # only consider word line voltages up to the supply voltage, other combinations are not relevant in practice

		if DEBUG:
			print("vdd: ", supply_voltage, " vwl:", wl_voltage)

		if math.isclose(supply_voltage, 0.9) or math.isclose(supply_voltage, 1.1) :
			supply_voltage_str = str(np.round(supply_voltage, 2))
		elif  math.isclose(supply_voltage, 1.0):
			supply_voltage_str = str(int(supply_voltage))
		else:
			supply_voltage_str = str(np.round(supply_voltage, 3))

		col_bl = "vbl vwl " + wl_voltage_str + " vdd " + supply_voltage_str
		idx_bl = df.columns.get_loc(col_bl)

		df_filtered = df.iloc[:, [idx_bl-1, idx_bl]]
		time = df_filtered.columns[0]
		df_filtered = df_filtered.loc[(df_filtered.loc[:,time] >= DISCHARGE_START-0.3)& (df_filtered.loc[:,time] <  (DISCHARGE_START + FIT_TIME))]

		t = df_filtered[time].add(-DISCHARGE_START).to_numpy().reshape(-1, 1)
		vwl = wl_voltage * np.ones(t.shape)
		vdd = supply_voltage * np.ones(t.shape)
		vbl = df_filtered[col_bl].to_numpy().reshape(-1, 1)

		array = np.concatenate((t, vwl, vdd, vbl), axis=1)
		if DEBUG:
			print(array)

		df_fit = pd.concat([df_fit, pd.DataFrame(array, columns=["t", "vwl", "vdd", "vbl"])], ignore_index=True)
		if DEBUG:
			print("df_fit.size:", df_fit.size)

if DEBUG:
	print(df_fit.head)
	print(df_fit.describe)

###############################################################################

df = pd.read_csv("spice-data/discharge-temp.csv", na_values=['', ' '], dtype=float)

wl_voltages = np.array([0.3, 0.34, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.78, 0.82, 0.86, 0.9, 0.94, 0.98, 1.0])
temperatures = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]).astype(float)

print("Word line voltages: ", wl_voltages)
print("Temperatures:       ", temperatures)

df_fit_temp = pd.DataFrame()

for wl_voltage in wl_voltages:

	if math.isclose(wl_voltage, 0.3) or math.isclose(wl_voltage, 0.5) or math.isclose(wl_voltage, 0.7) or math.isclose(wl_voltage, 0.9) or math.isclose(wl_voltage, 1.1):
			wl_voltage_str = str(np.round(wl_voltage, 2))
	elif  math.isclose(wl_voltage, 1.0):
			wl_voltage_str = str(int(wl_voltage))
	else:
		wl_voltage_str = str(np.round(wl_voltage, 3))

	for temperature in temperatures:

		# print("T: ", temperature, " vwl:", wl_voltage)
		temperature_str = str(int(temperature))

		col_bl = "vbl vwl " + wl_voltage_str + " temp " + temperature_str
		idx_bl = df.columns.get_loc(col_bl)

		df_filtered = df.iloc[:, [idx_bl-1, idx_bl]]
		time = df_filtered.columns[0]
		df_filtered = df_filtered.loc[(df_filtered.loc[:,time] >= DISCHARGE_START-0.3)& (df_filtered.loc[:,time] < (DISCHARGE_START + FIT_TIME))]

		t = df_filtered[time].add(-DISCHARGE_START).to_numpy().reshape(-1, 1)
		vwl = wl_voltage * np.ones(t.shape)
		temp = temperature * np.ones(t.shape)
		vbl = df_filtered[col_bl].to_numpy().reshape(-1, 1)

		array = np.concatenate((t, vwl, temp, vbl), axis=1)
		# print(array)

		df_fit_temp = pd.concat([df_fit_temp, pd.DataFrame(array, columns=["t", "vwl", "temp", "vbl"])], ignore_index=True)

###############################################################################

if PLOT:
	vwl_plot = np.array([0.3, 0.5, 0.7, 0.9, 1.0])

	plt.figure(figsize=(FIG_WIDTH_WIDE*CM, FIG_HEIGHT*CM))
	for vwl_sample in vwl_plot:
		df_sample = df_fit.loc[(df_fit.loc[:,'vdd'] >= 0.999) & (df_fit.loc[:,'vdd'] <= 1.001) & (df_fit.loc[:,'vwl'] >= (vwl_sample - 0.01)) & (df_fit.loc[:,'vwl'] <  (vwl_sample + 0.01))]
		print(vwl_sample, df_sample.size)
		t_plot= df_sample["t"].to_numpy()*1e9
		vbl_plot = df_sample["vbl"].to_numpy()
		t_interpolate = np.linspace(0, FIT_TIME*1e9, 500)
		vbl_plot = np.interp(t_interpolate, t_plot.flatten(), vbl_plot.flatten()).reshape(-1, 1)
		t_plot = t_interpolate.reshape(-1, 1)
		filter = vbl_plot > vwl_sample - 0.3
		filter_nonlin = (vbl_plot <= vwl_sample - 0.3) & (t_plot> DISCHARGE_START)
		plt.plot(t_plot[filter], vbl_plot[filter], linestyle='-', marker='None', markersize=1, label="$V_{WL}=%.1f$\,V" %(vwl_sample))
		plt.plot(t_plot[filter_nonlin], vbl_plot[filter_nonlin], linestyle='dotted', marker='None', markersize=1, color='grey')

	plt.xlim(left=0.0, right=2.2)  # smooth end ()
	plt.xlabel(r"\bf$\bf t$ [ns]")
	plt.ylabel(r"\bf$\bf V_{\overline{BL}}$ [V]")
	plt.legend()
	plt.grid()
	plt.show()

if PLOT:
	plt.figure(figsize=(FIG_WIDTH_WIDE*CM, FIG_HEIGHT*CM))

	capture_times = np.array([0.2, 0.4, 0.8, 1.6])

	for capture_time in capture_times:
		df_sample = df_fit.loc[(df_fit.loc[:,'vdd'] >= 0.999) & (df_fit.loc[:,'vdd'] <= 1.001) & (df_fit.loc[:,'t'] >= (capture_time*1e-9 - 0.01e-9)) & (df_fit.loc[:,'t'] <  (capture_time*1e-9 + 0.01e-9))]

		vbl_plot = df_sample["vbl"].to_numpy()
		vwl_plot = df_sample["vwl"].to_numpy()
		plt.plot(vwl_plot, vbl_plot, linestyle='-', marker='None', markersize=1, label="$\\tau_{0}=%.1f$\,ns" %(capture_time))

	plt.xlim(left=0.3, right=1.0)
	plt.xlabel(r"\bf$\bf V_{WL}$ [V]")
	plt.ylabel(r"\bf$\bf V_{\overline{BL}}$ [V]")
	plt.legend()
	plt.grid()
	plt.show()


if PLOT:
	plt.figure(figsize=(FIG_WIDTH_WIDE*CM, FIG_HEIGHT*CM))

	temperatures = np.array([0, 25, 50, 75])

	for temperature in temperatures:
		df_sample = df_fit_temp.loc[(df_fit_temp.loc[:,'vwl'] >= (1.0 - 0.01)) & (df_fit_temp.loc[:,'vwl'] <  (1.0 + 0.01)) & (df_fit_temp.loc[:,'temp'] >= (temperature - 0.01)) & (df_fit_temp.loc[:,'temp'] <  (temperature + 0.01))]

		print(temperature, df_sample.size)

		vbl_plot = df_sample["vbl"].to_numpy()
		t_plot = df_sample["t"].to_numpy()*1e9

		plt.plot(t_plot, vbl_plot, linestyle='-', marker='None', markersize=1, label="$T=%.0f$\,°C" %temperature)

	plt.xlim(left=0.0, right=2.2)
	plt.xlabel(r"\bf$\bf t$ [ns]")
	plt.ylabel(r"\bf$\bf V_{\overline{BL}}$ [V]")
	plt.legend()
	plt.grid()
	plt.show()

if PLOT:
	plt.figure(figsize=(FIG_WIDTH_WIDE*CM, FIG_HEIGHT*CM))

	voltages = np.array([0.9, 1.0, 1.1])

	for voltage in voltages:
		df_sample = df_fit.loc[(df_fit.loc[:,'vdd'] >= (voltage - 0.001)) & (df_fit.loc[:,'vdd'] <= (voltage + 0.001)) & (df_fit.loc[:,'vwl'] >= (0.9 - 0.001)) & (df_fit.loc[:,'vwl'] <  (0.9 + 0.001))]

		print(voltage, df_sample.size)

		vbl_plot = df_sample["vbl"].to_numpy()
		t_plot = df_sample["t"].to_numpy()*1e9

		plt.plot(t_plot, vbl_plot, linestyle='-', marker='None', markersize=1, label="$V_{DD}=%.1f$\,V" % voltage)

	plt.xlim(left=0.0, right=2.2)
	plt.xlabel(r"\bf$\bf t$ [ns]")
	plt.ylabel(r"\bf$\bf V_{\overline{BL}}$ [V]")
	plt.legend()
	plt.grid()
	plt.show()

df = pd.read_csv("spice-data/discharge-process.csv", na_values=['', ' '], dtype=float)

df_fit_proc = pd.DataFrame()

corners = ['best', 'typ', 'worst']

for corner in corners:

	col_bl = corner
	idx_bl = df.columns.get_loc(col_bl)

	df_filtered = df.iloc[:, [idx_bl-1, idx_bl]]
	time = df_filtered.columns[0]
	df_filtered = df_filtered.loc[(df_filtered.loc[:,time] >= DISCHARGE_START-0.3)& (df_filtered.loc[:,time] < (DISCHARGE_START + FIT_TIME))]

	t = df_filtered[time].add(-DISCHARGE_START).to_numpy().reshape(-1, 1)
	c = np.repeat(np.array([[corner]]), t.size, axis=0)
	vbl = df_filtered[col_bl].to_numpy().reshape(-1, 1)

	array = np.concatenate((t, c, vbl), axis=1)

	df_fit_proc = pd.concat([df_fit_proc, pd.DataFrame(array, columns=["t", "corner", "vbl"])], ignore_index=True)

if PLOT:
	plt.figure(figsize=(FIG_WIDTH_WIDE*CM, FIG_HEIGHT*CM))

	df_sample = df_fit_proc.loc[(df_fit_proc.loc[:,'corner'] == 'best')]

	vbl_plot = df_sample["vbl"].astype(float).to_numpy()
	t_plot = df_sample["t"].astype(float).to_numpy()*1e9

	plt.plot(t_plot, vbl_plot, linestyle='-', marker='None', markersize=1, label="fast")

	df_sample = df_fit_proc.loc[(df_fit_proc.loc[:,'corner'] == 'typ')]

	vbl_plot = df_sample["vbl"].astype(float).to_numpy()
	t_plot = df_sample["t"].astype(float).to_numpy()*1e9

	plt.plot(t_plot, vbl_plot, linestyle='-', marker='None', markersize=1, label="nominal")

	df_sample = df_fit_proc.loc[(df_fit_proc.loc[:,'corner'] == 'worst')]

	vbl_plot = df_sample["vbl"].astype(float).to_numpy()
	t_plot = df_sample["t"].astype(float).to_numpy()*1e9

	plt.plot(t_plot, vbl_plot, linestyle='-', marker='None', markersize=1, label="slow")

	plt.xlim(left=0.0, right=2.2)
	plt.xlabel(r"\bf$\bf t$ [ns]")
	plt.ylabel(r"\bf$\bf V_{\overline{BL}}$ [V]")
	plt.legend()
	plt.grid()
	plt.show()

## Monte Carlo samples

DISCHARGE_START = 9.0e-9

wl_voltages = np.array([0.3, 0.34, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.78, 0.82, 0.9, 0.94, 0.98, 1.0])
samples = np.arange(1000) + 1

df_fit_mc = pd.DataFrame()
df_fit_sigma = pd.DataFrame()

for wl_voltage in wl_voltages:
	df = pd.read_csv("spice-data/mismatch2/discharge_%.2f.csv" % (wl_voltage), na_values=['', ' '], dtype=float)
	df_fit_mc_run = pd.DataFrame()

	for sample in samples:
		if DEBUG:
			print("vwl:", str(wl_voltage), "sample: ", sample)
		col_bl = "vbl mcparamset %d" % sample
		idx_bl = df.columns.get_loc(col_bl)

		df_filtered = df.iloc[:, [idx_bl-1, idx_bl]]
		time = df_filtered.columns[0]
		df_filtered = df_filtered.loc[(df_filtered.loc[:,time] >= DISCHARGE_START)& (df_filtered.loc[:,time] < (DISCHARGE_START + FIT_TIME))]

		t = df_filtered[time].add(-DISCHARGE_START).to_numpy().reshape(-1, 1)
		vbl = df_filtered[col_bl].to_numpy().reshape(-1, 1)
		vwl = wl_voltage * np.ones(t.shape)

		array = np.concatenate((t, vbl, vwl), axis=1)
		if DEBUG:
			print(array)

		df_fit_mc_run = pd.concat([pd.DataFrame(array, columns=["t", "vbl", "vwl"]),df_fit_mc_run], ignore_index=True)
		if DEBUG:
			print(df_fit_mc_run.size)

	TIME_STEP = 0.025e-9
	sampling_times = np.linspace(0.0, FIT_TIME, int(np.floor(FIT_TIME/TIME_STEP)))

	array = np.zeros([0, 3])
	for sampling_time in sampling_times:
		df_filtered = df_fit_mc_run.loc[(df_fit_mc_run['t'] >= sampling_time)& (df_fit_mc_run['t'] < sampling_time+TIME_STEP)]
		sample = df_filtered['vbl'].to_numpy()

		# Filter out if number of elements insufficient
		if sample.size < 130:
			continue
		sample = sample - np.mean(sample)
		stddev = np.std(sample)
		NO_BINS = 65

		hist, bins = np.histogram(sample, NO_BINS)
		BIN_SPACING = (np.max(bins)-np.min(bins))/NO_BINS
		if DEBUG:
			print(sample.size, stddev)
			plt.plot(sample, color ='red', marker='x', markersize=2.5, linestyle='None')
			plt.grid()
			plt.show()
			print(hist)
			print(bins)
			print("hist max at", bins[np.argmax(hist)]+0.5*BIN_SPACING)
			plt.stairs(hist, edges=bins)
			plt.show()

		t = sampling_time + 0.5*TIME_STEP
		sigma = stddev
		vwl = wl_voltage
		if DEBUG:
			print(np.array([t, sigma, vwl]))
		array = np.append(array, np.array([[t, sigma, vwl]]), axis=0)

	df_fit_sigma = pd.concat([pd.DataFrame(array, columns=["t", "sigma", "vwl"]), df_fit_sigma], ignore_index=True)

	df_fit_mc = pd.concat([df_fit_mc,df_fit_mc_run], ignore_index=True)

if PLOT:
	plt.figure(figsize=(FIG_WIDTH_WIDE*CM, FIG_HEIGHT*CM))

	df_sample = df_fit_mc.loc[(df_fit_mc.loc[:,'vwl'] <= 0.31)]

	print(df_sample.size)

	vbl_plot = df_sample["vbl"].to_numpy()
	t_plot = df_sample["t"].to_numpy()*1e9

	plt.plot(t_plot, vbl_plot, linestyle='-', marker='None', markersize=1, label="$V_{WL}=0.3\,V$")

	df_sample = df_fit_mc.loc[(df_fit_mc.loc[:,'vwl'] >= 0.69) & (df_fit_mc.loc[:,'vwl'] <= 0.71)]

	print(df_sample.size)
	vbl_plot = df_sample["vbl"].to_numpy()
	t_plot = df_sample["t"].to_numpy()*1e9

	plt.plot(t_plot, vbl_plot, linestyle='-', marker='None', markersize=1, label="$V_{WL}=0.7\,V$")

	df_sample = df_fit_mc.loc[(df_fit_mc.loc[:,'vwl'] >= 0.99) & (df_fit_mc.loc[:,'vwl'] <= 1.01)]

	print(df_sample.size)
	vbl_plot = df_sample["vbl"].to_numpy()
	t_plot = df_sample["t"].to_numpy()*1e9

	plt.plot(t_plot, vbl_plot, linestyle='-', marker='None', markersize=1, label="$V_{WL}=1.0\,V$")

	plt.xlim(left=0.0, right=2.2)
	plt.xlabel(r"\bf$\bf t$ [ns]")
	plt.ylabel(r"\bf$\bf V_{\overline{BL}}$ [V]")
	plt.legend()
	plt.grid()
	plt.show()

if PLOT:
	exit(0)
