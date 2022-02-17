# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:06:02 2022

@author: bitsi
"""

# # change working directory
# import os
# os.chdir('C:\Xin\work\github\covid19-sir')
# os.getcwd()

from pprint import pprint
import covsirphy as cs


if __name__ == '__main__':
    loader = cs.DataLoader("../input", update_interval=48)
    jhu_data = loader.jhu()
    
    # Specify country and province (optinal) names
    snl = cs.Scenario(country="Japan", province=None)
    # Register datasets
    snl.register(jhu_data)
    
    snl.trend().summary()
    # Estimate the tau value and parameter values of SIR-F model
    snl.estimate(cs.SIRF, metric="MAE",)
    # Show the summary of parameter estimation
    snl.summary()

#%% 
snl.history("Infected")
snl.history("Fatal")
snl.history("Recovered")
snl.simulate(variables="all", name="Main")
