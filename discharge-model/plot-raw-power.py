##!/usr/bin/python3

import math
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt

from DataLoader import DataLoader
from util import *

DEBUG = False

plt.rcParams.update({
	"text.usetex": True,
	"font.family": "libertine",
	"font.size":  11
})

loader = DataLoader(0.0, 0.0)

###############################################################################

df = pd.read_csv("spice-data/power-vwl-0.8.csv", na_values=['', ' '], dtype=float)

t0 = df['t0'].to_numpy().reshape(-1, 1)
p0  = df['p0'].to_numpy().reshape(-1, 1)
t1 = df['t1'].to_numpy().reshape(-1, 1)
p1  = df['p1'].to_numpy().reshape(-1, 1)

fig, ax = plt.subplots(figsize=(FIG_WIDTH_WIDE*CM, FIG_HEIGHT*CM))
ax.axvspan(0, 2.4, facecolor=RED)
ax.axvspan(6, 9.6, facecolor=RED)
ax.axvspan(12, 15.6, facecolor=RED)
ax.axvspan(2.4, 6, facecolor=YELLOW)
ax.axvspan(9.6, 12, facecolor=GREEN)
plt.plot(t0*1e9, 1e6*p0, color ='red', marker='None', markersize=2.5, label ="$d=0$", linestyle='-')
plt.plot(t1*1e9, 1e6*p1, color ='blue', marker='None', markersize=2.5, label ="$d=1$", linestyle=':')
plt.xlabel("$t$ [ns]")
plt.ylabel("$P$ [µW]")
plt.xlim(0, 15.6)
plt.ylim(0, 300)
plt.grid()
plt.legend(loc='upper right')
plt.show()

###############################################################################

data, vdd, vwl, temp, e_wr, e_dc = loader.read_energy("spice-data/write-energy.csv", "spice-data/discharge-energy.csv", debug=False)

filter = np.argwhere((data == 0.0) & (vwl == 0.7) & (temp == 25))
vdd0 = vdd[filter].flatten()
e_wr0 = e_wr[filter].flatten()
e_dc0 = e_dc[filter].flatten()

filter = np.argwhere((data == 1.0) & (vwl == 0.7) & (temp == 25))
vdd1 = vdd[filter].flatten()
e_wr1 = e_wr[filter].flatten()
e_dc1 = e_dc[filter].flatten()

fig, ax = plt.subplots(figsize=(9*CM, 8.0*CM))
plt.plot(vdd0, 1e15*e_wr0, color ='blue', marker='None', markersize=2.5, label ="$d=0$", linestyle='-')
plt.plot(vdd1, 1e15*e_wr1, color ='blue', marker='None', markersize=2.5, label ="$d=1$", linestyle='-')
plt.xlabel("$V_{DD}$ [V]")
plt.ylabel("$E_{\mathrm{wr}}$ [fJ]")
plt.xlim(0.9, 1.1)
plt.grid()
# plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(FIG_WIDTH_SMALL*CM, FIG_HEIGHT*CM))
plt.plot(vdd0, 1e15*e_dc0, color ='red', marker='None', markersize=2.5, label ="$d=0$", linestyle='-')
plt.plot(vdd1, 1e15*e_dc1, color ='blue', marker='None', markersize=2.5, label ="$d=1$", linestyle='-')
plt.xlabel("$V_{DD}$ [V]")
plt.ylabel("$E_{\mathrm{disch}}$ [fJ]")
plt.xlim(0.9, 1.1)
plt.grid()
plt.legend()
plt.show()

filter = np.argwhere((data == 0.0) & (vdd == 1.1) & (temp == 25))
vwl0 = vwl[filter].flatten()
e_wr0 = e_wr[filter].flatten()
e_dc0 = e_dc[filter].flatten()

filter = np.argwhere((data == 1.0) & (vdd == 1.1) & (temp == 25))
vwl1 = vwl[filter].flatten()
e_wr1 = e_wr[filter].flatten()
e_dc1 = e_dc[filter].flatten()

fig, ax = plt.subplots(figsize=(FIG_WIDTH_SMALL*CM, FIG_HEIGHT*CM))
plt.plot(vwl0, 1e15*e_dc0, color ='red', marker='None', markersize=2.5, label ="$d=0$", linestyle='-')
plt.plot(vwl1, 1e15*e_dc1, color ='blue', marker='None', markersize=2.5, label ="$d=1$", linestyle='-')
plt.xlabel("$V_{WL}$ [V]")
plt.ylabel("$E_{\mathrm{disch}}$ [fJ]")
plt.xlim(0.3, 1.1)
plt.grid()
plt.legend()
plt.show()
