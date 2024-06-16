##!/usr/bin/python3

import math
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt

from util import *
from DataLoader import DataLoader
from SramFitter import SramFitter

PLOT             = True
PLOT_ERR_DETAILS = False

if PLOT_ERR_DETAILS:
    PLOT = True

DISCHARGE_START = 9.6e-9  # s after simulation start
FIT_TIME        = 2.0e-9  # s after discharge start

loader = DataLoader(start_time=DISCHARGE_START, load_period=FIT_TIME+0.2E-9)   # time in loader in  s
fitter = SramFitter(fit_time=FIT_TIME)  # time in fitter in ns

if PLOT:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "libertine",
        "font.size":  11,
    })

###################################################################################################
if __name__ == "__main__":
    print("\n______________________________________________________________________\n" + \
        "  Bitline voltage fitting (time, word line voltage)\n")

    t, vwl, vdd, vbl = loader.read_discharge("spice-data/discharge-vdd.csv", only_nominal_vdd=True, nominal_vdd=1.0, interpolate=True)

    fitter.fit_bit_line(vbl, t, vwl, vdd)
    fitter.print_params()

    print("------------ Validation ------------")
    t, vwl, vdd, vbl = loader.read_discharge("spice-data/discharge-vdd.csv", only_nominal_vdd=True, nominal_vdd=1.0, interpolate=False)
    fitter.validate_model(t, vwl, vdd, vbl, NOMINAL_TEMP, plot=PLOT, plot_err_details=PLOT_ERR_DETAILS, savepath=None)

    fitter.clear_fit()

    ###################################################################################################
    print("\n______________________________________________________________________\n" + \
        "  Bitline voltage fitting including supply voltage\n")

    t, vwl, vdd, vbl = loader.read_discharge("spice-data/discharge-vdd.csv", interpolate=True)
    fitter.fit_bit_line(vbl, t, vwl, vdd)
    fitter.print_params()

    print("------------ Validation ------------")
    t, vwl, vdd, vbl = loader.read_discharge("spice-data/discharge-vdd.csv", interpolate=False)
    fitter.validate_model(t, vwl, vdd, vbl, NOMINAL_TEMP, plot=PLOT, plot_err_details=PLOT_ERR_DETAILS, savepath=None)

    ###################################################################################################
    print("\n______________________________________________________________________\n" + \
        "  Temperature Modelling\n")

    t, vwl, _, vbl, temp = loader.read_temperature("spice-data/discharge-temp.csv", interpolate=True)
    fitter.fit_temperature(vbl, t, vwl, temp)
    fitter.print_params()

    print("------------ Validation ------------")
    # fitter.validate_temperature_model(t, vbl, vwl, temp, plot=PLOT, savepath=None)
    t, vwl, vdd, vbl, temp = loader.read_temperature("spice-data/discharge-temp.csv", interpolate=False)
    fitter.validate_model(t, vwl, vdd, vbl, temp, mismatch=False, plot=PLOT, plot_err_details=PLOT_ERR_DETAILS, savepath=None)

    ###################################################################################################
    print("\n______________________________________________________________________\n" + \
        "  Energy Modelling\n")

    data, vdd, vwl, temp, e_wr, e_dc = loader.read_energy("spice-data/write-energy.csv", "spice-data/discharge-energy.csv", debug=False)
    fitter.fit_write_energy(data=data, vdd=vdd, temp=temp, energy=e_wr)
    fitter.fit_discharge_energy(data=data, vdd=vdd, temp=temp, vwl=vwl, energy=e_dc)
    fitter.print_params()

    print("------------ Validation ------------")
    fitter.validate_energy_model(data=data, vwl=vwl, vdd=vdd, temp=temp, e_wr=e_wr, e_disch=e_dc, plot=PLOT, plot_err_details=PLOT_ERR_DETAILS, savepath=None)

    ###################################################################################################
    print("\n______________________________________________________________________\n" + \
        "  Mismatch Modelling\n")

    t, vwl, sigma = loader.read_mismatch("spice-data/mismatch", debug=False, savepath=None)

    fitter.fit_mismatch(t, vwl, sigma)
    fitter.print_params()
    fitter.export_params("../system-model/include", "sram_params.v")

    print("------------ Validation ------------")
    fitter.validate_mismatch_model(t, sigma, vwl, plot=PLOT, plot_err_details=PLOT_ERR_DETAILS, savepath=None)

    ###################################################################################################
    print("\n______________________________________________________________________\n" + \
        "  Evaluation\n")

    t_test = np.linspace(0.0, 2e-9, 201)

    if PLOT:
        plt.figure(figsize=(FIG_WIDTH_WIDE*CM, FIG_HEIGHT*CM))

    for i in range(500):
        vbl_sample = fitter.sample_discharge(t_test, 1.0, 1.1, 0, mismatch=True)
        if PLOT:
            if i==0:
                plt.plot(t_test, vbl_sample, color ='red', marker='x', markersize=2.5, linestyle='None', label="$V_{WL} = 1.0\,V$, $V_{DD} = 1.1\,V$, $T=0\,^{\circ}\mathrm{C}$")
            else:
                plt.plot(t_test, vbl_sample, color ='red', marker='x', markersize=2.5, linestyle='None')

    for i in range(500):
        vbl_sample = fitter.sample_discharge(t_test, 0.7, 1.0, 25, mismatch=True)
        if PLOT:
            if i==0:
                plt.plot(t_test, vbl_sample, color ='blue', marker='x', markersize=2.5, linestyle='None', label="$V_{WL} = 0.7\,V$, $V_{DD} = 1.0\,V$, $T=25\,^{\circ}\mathrm{C}$")
            else:
                plt.plot(t_test, vbl_sample, color ='blue', marker='x', markersize=2.5, linestyle='None')

    for i in range(500):
        vbl_sample = fitter.sample_discharge(t_test, 0.5, 0.9, 75, mismatch=True)
        if PLOT:
            if i==0:
                plt.plot(t_test, vbl_sample, color ='green', marker='x', markersize=2.5, linestyle='None', label="$V_{WL} = 0.5\,V$, $V_{DD} = 0.9\,V$, $T=75\,^{\circ}\mathrm{C}$")
            else:
                plt.plot(t_test, vbl_sample, color ='green', marker='x', markersize=2.5, linestyle='None')

    if PLOT:
        plt.xlabel("$t$ [ns]")
        plt.ylabel("$V_{\overline{BL}}$ [V]")
        plt.legend()
        plt.grid()
        plt.show()
