import os
import re
import numpy as np
import pandas as pd

CM = 1.0/2.54  # 1 cm in inches
FIG_HEIGHT = 8.0
FIG_WIDTH_SMALL = 8.0
FIG_WIDTH_WIDE = 13.0

INTERPOLATION_POINTS = 15

NOMINAL_VDD = 1.0
NOMINAL_TEMP = 25

RED =   (248/255,206/255,204/255)
YELLOW =(255/255,242/255,204/255)
GREEN = (213/255,232/255,212/255)

def get_validation_data(path, start_time, end_time, vwl=False):
    df = pd.read_csv(path, na_values=['', ' '], dtype=float)
    df = df.loc[(df['t'] >= start_time) & (df["t"] <= end_time)]
    t = np.add(df['t'].to_numpy(), -start_time)
    vbl = df['vbl'].to_numpy()
    if vwl:
        vwl = df['vwl'].to_numpy()
    else:
        vwl = None
    vdd = None
    temp = None
    return t, vwl, vbl
