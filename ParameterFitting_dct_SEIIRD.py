# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:34:05 2022

@author: bitsi
"""

import pandas as pd
import numpy as np
import covsirphy as cs
import matplotlib.pyplot as plt

# parameter fitting for mct
file_dir = "mct_dct_data\\"
for cur_pct in np.arange(0, 0.9, 0.1):
    pop_size, eff_mct, pct_dct, trace_type  = 100e3, 1, cur_pct, 'dct'
    filename = f'{pop_size:.0f}_' + trace_type + f'_{pct_dct * 100:.0f}%' + '.csv'
    loader = cs.DataLoader(update_interval=None)
    # loader.read_csv("mct_0.9_tw_2_20000.csv") # , parse_dates=["date"], dayfirst=False
    loader.read_csv(file_dir + filename)
    
    # loader.read_csv("no_intervention_100000.csv")
    loader._local_df.rename(columns={"n_susceptible": "Susceptible", "n_exposed": "Exposed", "n_infectious": "Infectious",
                                     "n_diagnosed": "Isolated", "n_recovered": "Recovered", "n_dead": "Dead"}, inplace=True)
    loader.assign(
        country="Synthetic",
        Population=pop_size,
        date= pd.date_range(start="3/1/2020", periods=loader.local.shape[0])
    )
    # lock user supplied data first
    loader.lock_extend(date="date", country="country", province="province")
    # initiate an instance of jhu_data
    jhu_data = loader.jhu_extend()
    # Specify country and province (optinal) names
    snl = cs.Scenario(country="Synthetic", province=None, tau=1440, auto_complement=False)
    # Register datasets
    snl.register(jhu_data)
    # snl.trend(min_size=7)
    snl.trend_SA(min_size=7, aggregate_cols = ["Isolated", "Recovered", "Dead"]) # "Isolated", "Recovered",
    
    snl.summary()
    model = cs.SEIIRD
    model.population = pop_size
    
    snl.estimate(model=model, metric="MSE", check_dict = {"timeout": 360, "timeout_iteration": 1, "tail_n": 20, "allowance": (0.99, 1.01)},
                 metric_cols = ["Exposed", "Isolated", "Infectious"], metric_cols_weights = [0.33, 0.33, 0.34]
                 )
    fig_name = "mct_dct_data\\SEIIRD\\" + f'{pop_size:.0f}_' + trace_type + f'_{pct_dct * 100:.0f}%_'
    snl.history("Susceptible", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
    plt.savefig(fig_name + "Susceptible" + '.png')
    
    snl.history("Exposed", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
    plt.savefig(fig_name + "Exposed" + '.png')
    
    snl.history("Infectious", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
    plt.savefig(fig_name + "Infectious" + '.png')
    
    snl.history("Isolated", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
    plt.savefig(fig_name + "Isolated" + '.png')
    
    snl.history("Recovered", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
    plt.savefig(fig_name + "Recovered" + '.png')
    
    snl.history("Dead", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
    plt.savefig(fig_name + "Dead" + '.png')
    
    plt.close('all')
    
    fitted_df = snl._summary()
    sub_df = fitted_df[["Start", "End"] + cs.SEIIRD_Q.PARAMETERS]
    sub_df['Date'] = sub_df.apply(lambda d: pd.date_range(d['Start'],
                                                        d['End'], 
                                                        freq='D'),
                                  axis=1)
    sub_df = sub_df.explode('Date')
    
    filename_save = "mct_dct_data\\SEIIRD\\" + f'{pop_size:.0f}_' + trace_type + f'_{pct_dct * 100:.0f}%_' + 'estimated_parameters' + '.csv'
    sub_df.to_csv(filename_save, index=False)
    