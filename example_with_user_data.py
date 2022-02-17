# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 09:28:42 2022

@author: bitsi
"""
# import os
# os.chdir('C:\Xin\work\github\covid19-sir')
# os.getcwd()

import pandas as pd
# from pprint import pprint
import covsirphy as cs
# from covsirphy.cleaning.jhu_extend import JHUData_extend
# from covsirphy.cleaning.jhu_data import JHUData


loader = cs.DataLoader(update_interval=None)
# loader.read_csv("mct_0.9_tw_2_20000.csv") # , parse_dates=["date"], dayfirst=False
loader.read_csv("mct_0.9_tw_2_100000.csv")
loader._local_df.rename(columns={"n_susceptible": "Susceptible", "n_exposed": "Exposed", "n_infectious": "Infectious",
                                 "n_diagnosed": "Isolated", "n_recovered": "Recovered", "n_dead": "Dead"}, inplace=True)
loader.assign(
    country="Synthetic",
    Population=100_000,
    date= pd.date_range(start="2/1/2020", periods=loader.local.shape[0])
)
loader._local_df["Infectious"] -= loader._local_df["Isolated"]
loader.local
# lock user supplied data first
loader.lock_extend(date="date", country="country", province="province")
# initiate an instance of jhu_data
jhu_data = loader.jhu_extend()
#%%

# Specify country and province (optinal) names
snl = cs.Scenario(country="Synthetic", province=None, tau=1440, auto_complement=False)
# Register datasets
snl.register(jhu_data)
# snl.trend(min_size=7)
snl.trend_SA(min_size=7, aggregate_cols = ["Isolated", "Recovered", "Dead"])

snl.summary()
model = cs.SEIIRD
model.population = 100_000

if __name__ == '__main__':
    snl.estimate(model=model, metric="RMSLE", check_dict = {"timeout": 360, "timeout_iteration": 1, "tail_n": 20, "allowance": (0.99, 1.01)},
                 metric_cols = ["Exposed", "Infectious"], metric_cols_weights = [1, 0]
                 )

#%%
snl.history("Susceptible", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
snl.history("Exposed", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
snl.history("Infectious", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
snl.history("Isolated", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
snl.history("Recovered", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
snl.history("Dead", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])

#%%
snl.simulate(variables="all", name="Main")
snl.summary(cs.SEIIRD.PARAMETERS)
snl.summary(["Start", "End"])

raw_data = snl["Main"]._track_df[cs.SEIIRD.VARIABLES]
cs.line_plot(raw_data, "Total number of cases over time")
# if __name__ == '__main__':
#     loader = cs.DataLoader("../input", update_interval= 24)
#     jhu_data = loader.jhu()
    
    
#     # data = pd.read_csv()
#     loader.read_csv("dct_0.9_tw_2.csv")
    
#     # Specify country and province (optinal) names
#     snl = cs.Scenario(country="Synthetic", province=None)
#     # Register datasets
#     snl.register(jhu_data)
    
#     snl.trend().summary()
#     # Estimate the tau value and parameter values of SIR-F model
#     snl.estimate(cs.SIRF)
#     # Show the summary of parameter estimation
#     snl.summary()