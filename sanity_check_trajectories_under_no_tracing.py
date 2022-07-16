# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:49:03 2022

@author: bitsi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sanity check for epidemic trajectories when there is no tracing: pct=0 or eff=0

file_dir = "mct_dct_data\\"

cur_eff = 0.0
pop_size, eff_mct, pct_dct, trace_type  = 100e3, cur_eff, 0, 'mct'
filename = f'{pop_size:.0f}_' + trace_type + f'_{eff_mct:.1f}' + '.csv'
df_mct = pd.read_csv(file_dir + filename)

cur_pct = 0.0
pop_size, eff_mct, pct_dct, trace_type  = 100e3, 1, cur_pct, 'dct'
filename = f'{pop_size:.0f}_' + trace_type + f'_{pct_dct * 100:.0f}%' + '.csv'
df_dct = pd.read_csv(file_dir + filename)

for col in df_mct.columns.to_list():
    ax = df_mct.plot(y=col, label='mct', color='k')
    df_dct.plot(y=col, ax=ax, label='dct', color='r')
plt.close('all')    

tmp = df_mct - df_dct

# the trajectories are exactly the same, the difference in fitted parameters come from the fitting process