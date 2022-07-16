# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 13:00:42 2022

@author: bitsi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# processing of fitted parameters on grid values (both for mct and dct)
# provide visualization

#%% for DCT
file_dir = "mct_dct_data\\SEIIRD\\"

fitted_params = []
for cur_pct in np.arange(0, 0.9, 0.1):
    pop_size, eff_mct, pct_dct, trace_type  = 100e3, 1, cur_pct, 'dct'
    file_name = f'{pop_size:.0f}_' + trace_type + f'_{pct_dct * 100:.0f}%_' + 'estimated_parameters' + '.csv'
    cur_df = pd.read_csv(file_dir + file_name)
    fitted_params.append(cur_df)

# visualization for each parameters

df_lamda = []
for df in fitted_params:
    df_lamda.append(df.lamda)

df_lamda_all = pd.concat(df_lamda, axis=1)
df_lamda_all.columns = [f'{cur_pct * 100:.0f}%' for cur_pct in np.arange(0, 0.9, 0.1)]
df_lamda_all.plot( cmap='coolwarm')

# save figures for each parameters : DCT

for col_name in ["beta_1", "beta_2", "alpha", "gamma", "lamda"]:
    df_list = []
    for df in fitted_params:
        df_list.append(df[col_name])
    df = pd.concat(df_list, axis=1)
    df.columns = [f'{cur_pct * 100:.0f}%' for cur_pct in np.arange(0, 0.9, 0.1)]
    df.plot( cmap='coolwarm', title=col_name, xlabel='Days')
    plt.savefig(file_dir + trace_type + '_' + col_name + '.png')
    
    
#%% for MCT
file_dir = "mct_dct_data\\SEIIRD\\"

fitted_params = []
for cur_eff in np.arange(0, 1.1, 0.1):
    pop_size, eff_mct, pct_dct, trace_type  = 100e3, cur_eff, 0, 'mct'
    file_name = f'{pop_size:.0f}_' + trace_type + f'_{eff_mct:.1f}_' + 'estimated_parameters' + '.csv'
    cur_df = pd.read_csv(file_dir + file_name)
    fitted_params.append(cur_df)

# visualization for each parameters

df_lamda = []
for df in fitted_params:
    df_lamda.append(df.lamda)

df_lamda_all = pd.concat(df_lamda, axis=1)
df_lamda_all.columns = [f'{eff_mct:.1f}' for eff_mct in np.arange(0, 1.1, 0.1)]
df_lamda_all.plot( cmap='coolwarm')

# save figures for each parameters : DCT

for col_name in ["beta_1", "beta_2", "alpha", "gamma", "lamda"]:
    df_list = []
    for df in fitted_params:
        df_list.append(df[col_name])
    df = pd.concat(df_list, axis=1)
    df.columns = [f'{eff_mct:.1f}' for eff_mct in np.arange(0, 1.1, 0.1)]
    df.plot( cmap='coolwarm', title=col_name, xlabel='Days')
    plt.savefig(file_dir + trace_type + '_' + col_name + '.png')        