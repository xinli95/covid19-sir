# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:08:33 2022

@author: bitsi
"""

# import os
# os.chdir('C:\Xin\work\github\covid19-sir')
# os.getcwd()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
from scipy.stats.qmc import LatinHypercube
import statsmodels.api as sm
from datetime import timedelta
import copy
import covsirphy as cs
#### generate a synthetic trajectory of diagnosed compartment

# read in trajectories from offline data and compute a randomly weighted one
df_list = []
file_dir = "mct_dct_data\\"
for cur_eff in np.arange(0, 1.1, 0.1):
    pop_size, eff_mct, pct_dct, trace_type  = 100e3, cur_eff, 0, 'mct'
    filename = f'{pop_size:.0f}_' + trace_type + f'_{eff_mct:.1f}' + '.csv'
    tmp_df = pd.read_csv(file_dir + filename)
    df_list.append(tmp_df.n_diagnosed)
df_mct = pd.concat(df_list, axis=1)
df_mct.columns = [f'{eff_mct:.1f}' for eff_mct in np.arange(0, 1.1, 0.1)]
df_mct.plot( cmap='coolwarm', title='n_diagnosed')
# plt.savefig(file_dir + 'n_diagnosed_mct.png')


df_list = []
for cur_pct in np.arange(0, 0.9, 0.1):
    pop_size, eff_mct, pct_dct, trace_type  = 100e3, 1, cur_pct, 'dct'
    filename = f'{pop_size:.0f}_' + trace_type + f'_{pct_dct * 100:.0f}%' + '.csv'
    tmp_df = pd.read_csv(file_dir + filename)
    df_list.append(tmp_df.n_diagnosed)
df_dct = pd.concat(df_list, axis=1)
df_dct.columns = [f'{pct_dct * 100:.0f}%' for pct_dct in np.arange(0, 0.9, 0.1)]
df_dct.plot( cmap='coolwarm', title='n_diagnosed')
# plt.savefig(file_dir + 'n_diagnosed_dct.png')

df_merged = pd.concat([df_mct, df_dct], axis=1)

# np.random.seed(100)
rsamples = np.random.random(df_merged.shape[1])
rsamples /= rsamples.sum()
rsamples
df_synthetic = df_merged.mul(rsamples, axis=1).sum(axis=1)
noises = np.random.randn(df_synthetic.shape[0])
noises = df_synthetic * 0.1 * noises
df_synthetic = df_synthetic + noises
ax = df_synthetic.plot(c='k',linestyle='dashed', label='Synthetic')
df_merged.iloc[:45,:].plot( cmap='coolwarm', title='n_diagnosed', ax = ax)
ax.axvline(x=45, c='g')
plt.legend(['Observed'])

# #### given the synthetic trajectory, carry out multi-period resource allocation
# n_periods = (df_synthetic.shape[0] - 30) // 7 # first 30 days have no variation, 7 days/period
# decisions = [] # 'lamda'
# variables = [] # variables in the compartmental model: 'beta_1', 'beta_2', 'alpha', 'gamma'
# states = [] # states of the compartment models; initial values at the beginning of each period

# cur_decision = 0 # no tracing initially
# initial_exposed = 20
# pop_size = 100e3
# cur_states = {'n_susceptible': pop_size - initial_exposed,
#               'n_exposed': initial_exposed,
#               'n_infectious': 0,
#               'n_diagnosed': 0,
#               'n_recovered': 0,
#               'n_dead': 0}


#%% data preparation for compartments except "n_diagnosed"
def collect_data(col_name):
    ## data from mct
    df_list = []
    file_dir = "mct_dct_data\\"
    for cur_eff in np.arange(0, 1.1, 0.1):
        pop_size, eff_mct, pct_dct, trace_type  = 100e3, cur_eff, 0, 'mct'
        filename = f'{pop_size:.0f}_' + trace_type + f'_{eff_mct:.1f}' + '.csv'
        tmp_df = pd.read_csv(file_dir + filename)
        df_list.append(tmp_df[col_name])
    df_mct = pd.concat(df_list, axis=1)
    df_mct.columns = [f'{eff_mct:.1f}' for eff_mct in np.arange(0, 1.1, 0.1)]
    
    ## data from dct
    df_list = []
    for cur_pct in np.arange(0, 0.9, 0.1):
        pop_size, eff_mct, pct_dct, trace_type  = 100e3, 1, cur_pct, 'dct'
        filename = f'{pop_size:.0f}_' + trace_type + f'_{pct_dct * 100:.0f}%' + '.csv'
        tmp_df = pd.read_csv(file_dir + filename)
        df_list.append(tmp_df[col_name])
    df_dct = pd.concat(df_list, axis=1)
    df_dct.columns = [f'{pct_dct * 100:.0f}%' for pct_dct in np.arange(0, 0.9, 0.1)]
    df_merged = pd.concat([df_mct, df_dct], axis=1)
    return df_merged

compartment_names = ['n_susceptible', 'n_exposed', 'n_infectious', 'n_diagnosed', 'n_recovered', 'n_dead']
offline_data = {}
for col in compartment_names:
    tmp = collect_data(col)
    tmp.index = pd.date_range(start="3/1/2020", periods=tmp.shape[0])
    offline_data[col] = tmp

#%% utility functions

def initialize_dataloader(df, pop_size):
    # pop_size = 100_000
    loader = cs.DataLoader(update_interval=None)
    loader.read_dataframe(df)
    loader._local_df.rename(columns={"n_susceptible": "Susceptible", "n_exposed": "Exposed", "n_infectious": "Infectious",
                                     "n_diagnosed": "Isolated", "n_recovered": "Recovered", "n_dead": "Dead"}, inplace=True)
    loader.assign(
        country="Synthetic",
        Population=pop_size,
        date= pd.date_range(start="3/1/2020", periods=loader.local.shape[0])
    )
    loader.local
    # lock user supplied data first
    loader.lock_extend(date="date", country="country", province="province")
    # initiate an instance of jhu_data
    jhu_data = loader.jhu_extend()
    return jhu_data
#%% get the data for each compartment of the synthetic one
rsamples = np.random.random(df_merged.shape[1])
rsamples /= rsamples.sum()

df_synthetic_dict = {}
compartment_names = ['n_susceptible', 'n_exposed', 'n_infectious', 'n_diagnosed', 'n_recovered', 'n_dead']
for col_name in compartment_names:
    df_merged = collect_data(col_name)
    df_synthetic_dict[col_name] =  df_merged.mul(rsamples, axis=1).sum(axis=1)


#%% get the estimated parameters for the bases

def collect_bases_parameters(col_name):
# col_name = 'beta_1'
    ## data from mct
    df_list = []
    file_dir = "mct_dct_data\\SEIIRD\\"
    for cur_eff in np.arange(0, 1.1, 0.1):
        pop_size, eff_mct, pct_dct, trace_type  = 100e3, cur_eff, 0, 'mct'
        filename = f'{pop_size:.0f}_' + trace_type + f'_{eff_mct:.1f}' + '_estimated_parameters' + '.csv'
        tmp_df = pd.read_csv(file_dir + filename)
        df_list.append(tmp_df[col_name])
    df_mct = pd.concat(df_list, axis=1)
    df_mct.columns = [f'{eff_mct:.1f}' for eff_mct in np.arange(0, 1.1, 0.1)]
    
    ## data from dct
    df_list = []
    for cur_pct in np.arange(0, 0.9, 0.1):
        pop_size, eff_mct, pct_dct, trace_type  = 100e3, 1, cur_pct, 'dct'
        filename = f'{pop_size:.0f}_' + trace_type + f'_{pct_dct * 100:.0f}%' + '_estimated_parameters' + '.csv'
        tmp_df = pd.read_csv(file_dir + filename)
        df_list.append(tmp_df[col_name])
    df_dct = pd.concat(df_list, axis=1)
    df_dct.columns = [f'{pct_dct * 100:.0f}%' for pct_dct in np.arange(0, 0.9, 0.1)]
    df_merged = pd.concat([df_mct, df_dct], axis=1)
    
    # # mct 0.0 should be equivalent to dct 0%
    # df_merged['0%'] = df_merged['0.0']
    
    # update the index
    df_merged.index = pd.date_range(start="3/1/2020", periods=df_merged.shape[0])
    return df_merged

parameter_names = ['beta_1', 'beta_2', 'alpha', 'gamma', 'lamda']
offline_parameters = {}
for col_name in parameter_names:
    offline_parameters[col_name] = collect_bases_parameters(col_name)
    
#### delete mct 0.0, since it is equivalent to dct 0%
offline_data = {key: df.iloc[:,1:] for key, df in offline_data.items()}
offline_parameters = {key: df.iloc[:, 1:] for key, df in offline_parameters.items()}

#%% estimate the phase-wise parameters for the synthetic one
df_synthetic = pd.concat(df_synthetic_dict.values(), axis=1)
df_synthetic.columns = compartment_names

# pop_size = 100e3
# jhu_data = initialize_dataloader(df_synthetic, pop_size)
# snl = cs.Scenario(country="Synthetic", province=None, tau=1440, auto_complement=False)
# # Register datasets
# snl.register(jhu_data)
# # snl.trend(min_size=7)
# snl.trend_SA(min_size=7, aggregate_cols = ["Isolated", "Recovered", "Dead"]) # "Isolated", "Recovered",

# snl.summary()
# model = cs.SEIIRD
# model.population = pop_size

# snl.estimate(model=model, metric="MSE", check_dict = {"timeout": 360, "timeout_iteration": 1, "tail_n": 20, "allowance": (0.99, 1.01)},
#              metric_cols = ["Exposed", "Isolated", "Infectious"], metric_cols_weights = [0.33, 0.33, 0.34]
#              )

# snl.history("Susceptible", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
# snl.history("Exposed", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
# snl.history("Infectious", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
# snl.history("Isolated", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
# snl.history("Recovered", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
# snl.history("Dead", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])

# fitted_df = snl._summary()
# sub_df = fitted_df[["Start", "End"] + cs.SEIIRD.PARAMETERS]
# sub_df['Date'] = sub_df.apply(lambda d: pd.date_range(d['Start'],
#                                                     d['End'], 
#                                                     freq='D'),
#                               axis=1)
# heuristic_df = sub_df.explode('Date')

#%% constraint function on lamda
def constraint_lamda(t, const_type):
    
    # get the time dependent constraint on lamda
    max_lamda = 0.3
    min_lamda = 0.1
    start_t = 31
    rampup_t = 20
    if const_type == 'linear':
        if t - start_t <= rampup_t:
            return min_lamda + (t - start_t) * (max_lamda - min_lamda) / rampup_t
        else:
            return max_lamda
    
    elif const_type == 'exponential':
        if t - start_t <= rampup_t:
            return min_lamda + np.exp(t - rampup_t + start_t) * (max_lamda - min_lamda)
        else:
            return max_lamda
        
    # elif const_type == 'log':
    #     if t - start_t <= rampup_t:
    #         return min_lamda + np.exp(t - rampup_t + start_t) * (max_lamda - min_lamda)
    #     else:
    #         return max_lamda
        
    else:
        print('Not Implemented')
        return None
    
def get_budget_lamda(t, const_type):
    
    # get the budget lamda for current period
    discount = 0.7
    cur_max_lamda = constraint_lamda(t, const_type)
    
    return len_period * cur_max_lamda * discount
    
#%% main loop



# from numpy.random import default_rng
SEIIRD_col_names = ['n_susceptible', 'n_exposed', 'n_infectious', 'n_diagnosed', 'n_recovered', 'n_dead']
pop_size = 100e3
model = cs.SEIIRD
model.population = pop_size
# rng = np.random.default_rng(12345)
np.random.seed(12345)
n_scenarios = 20
# pre generate random samples for each scenario
rsamples = np.random.randn(n_scenarios)

# take out observation of n_diagnosed
df_synthetic_diagnosed = df_synthetic[['n_diagnosed']]  # observed signal of n_diagnosed
df_synthetic_diagnosed.index = pd.date_range(start="3/1/2020", periods=df_synthetic_diagnosed.shape[0])

# add random noises to the observation
noises = np.random.randn(*df_synthetic_diagnosed.shape)
noises = df_synthetic_diagnosed * 0.1 * noises
df_synthetic_diagnosed = df_synthetic_diagnosed + noises

df_bases = df_merged.copy()
len_period0 = 30 # len of period 0
len_period = 7
const_type = 'linear'
cur_end = df_synthetic_diagnosed.index.min()
start_date = df_synthetic_diagnosed.index.min()
while cur_end < df_synthetic_diagnosed.index.max():
    
    # cur_Theta: parameters determined from last period
    # cur_lamda: decision variable determined from last period: not implemented here !!!!
    # 1. with the initial cur_Theta, cur_lamda, simulate one period forward for each scenario
    # 2. Update the estimate on cur_Theta based on the bases with the observations until the end of current period
    # 3. Update the decision variable by solving the optimization model
     
    if cur_end == df_synthetic_diagnosed.index.min(): # first period when there is no intervention
        df_observed = df_synthetic_diagnosed[:len_period0]  
        df_other_compartment_dict = {col: df.iloc[:len_period0,:].mean(axis=1) for col, df in offline_data.items()}
        df_other_compartment_dict.pop('n_diagnosed')
        df_others = pd.DataFrame(df_other_compartment_dict)
        df_all = df_others.copy()
        df_all["n_diagnosed"] = df_observed.to_numpy()
        df_all = df_all[SEIIRD_col_names]
        
        # estimate period0 for once, make copies of scenarios
        jhu_data = initialize_dataloader(df_all, pop_size)
        snl = cs.Scenario(country="Synthetic", province=None, tau=1440, auto_complement=False)
        # Register datasets
        snl.register(jhu_data)
        snl.trend_SA(min_size=7, aggregate_cols = ["Isolated", "Recovered", "Dead"]) # "Isolated", "Recovered",
        snl.estimate(model=model, metric="MSE", check_dict = {"timeout": 360, "timeout_iteration": 1, "tail_n": 20, "allowance": (0.99, 1.01)},
                 metric_cols = ["Exposed","Isolated", "Infectious"], metric_cols_weights = []
                 )
        saved_track_df = snl._tracker_dict["Main"]._track_df
        
        ### all scenarios start from the same observations in the first period; parameters are estimated only once

        scenarios_list = []
        for scenario_idx in range(n_scenarios):
            snl = cs.Scenario(country="Synthetic", province=None, tau=1440, auto_complement=False)
            # Register datasets
            snl.register(jhu_data)
            snl.trend_SA(min_size=7, aggregate_cols = ["Isolated", "Recovered", "Dead"])
            snl._tracker_dict["Main"]._track_df = saved_track_df.copy()
            snl._model = model
            scenarios_list.append(snl)
            plt.close('all')
        
        #### update cur_Theta, which excludes lamda
        cur_start, cur_end = df_observed.index.min(), df_observed.index.max()
        df_bases = offline_data['n_diagnosed'].loc[cur_start:cur_end]
        
        # linear regression with bases
        X = df_bases.copy()
        model = sm.OLS(df_observed, X)
        model_fitted = model.fit()
        cur_beta = model_fitted.params
        cur_bse = model_fitted.bse
        df_predicted = X @ model_fitted.params
        # df_predicted_ub = X @ (cur_beta + 2 * cur_bse)
        # df_predicted_lb = X @ (cur_beta - 2 * cur_bse)
        # # visualize
        # fig, ax = plt.subplots()
        # df_observed.loc[cur_start:cur_end].plot(ax=ax, style = 'ro-', label = 'New observation')
        # df_predicted.loc[cur_start:cur_end].plot(ax=ax, style = 'b*-', label = 'Observation recovery')
        # # df_predicted_ub.loc[cur_start:cur_end].plot(ax=ax, style = 'g--', label = 'CI')
        # # df_predicted_lb.loc[cur_start:cur_end].plot(ax=ax, style = 'g--', label = 'CI')
        # bases_ax = df_bases.plot(ax=ax, style='k-', alpha=0.5, label=None)
        # plt.axvline(x=cur_start)
        # # plt.setp(bases_ax, label="_")
        # plt.legend()
        
        # calculate cur_Theta based on offline_parameters
        cur_Theta_dict = {}
        cur_Theta_sd_dict = {}
        for param_name in parameter_names:
            if param_name == 'lamda':
                df_tmp = offline_parameters[param_name]
                # get the parameters for the next period
                df_tmp = df_tmp.loc[cur_end + timedelta(days=1):cur_end + timedelta(days=len_period)]
                best_lamda = (df_tmp @ cur_beta).mean()
                best_lamda = np.ones(len_period) * best_lamda
            else:
                df_tmp = offline_parameters[param_name]
                # get the parameters for the next period
                df_tmp = df_tmp.loc[cur_end + timedelta(days=1):cur_end + timedelta(days=len_period)]
                cur_Theta_dict[param_name] = (df_tmp @ cur_beta).mean()
                cur_Theta_sd_dict[param_name] = (df_tmp @ cur_beta).std()
        
        next_start = cur_end + timedelta(days = 1)
        next_end = cur_end + timedelta(days=len_period)
    else:
        #### simulate one period forward for each scenario, with the updated parameters and decision from last period
        
        cur_start, cur_end = next_start, next_end
        df_states = []
        for idx, cur_scenario in enumerate(scenarios_list):
            
            # cur_Theta_dict['lamda'] = Lamda
            # apply perturbation around cur_Theta_dict
            cur_noise = rsamples[idx]
            cur_Theta_dict_perturbed = {key: max(val + cur_Theta_sd_dict[key] * cur_noise, 0) for key, val in cur_Theta_dict.items()}
            # # add decision variable best_lamda
            # cur_Theta_dict_perturbed['lamda'] = best_lamda
            
            # add best_lamda day by day
            for cur_lamda in best_lamda:
                cur_Theta_dict_perturbed['lamda'] = cur_lamda
                cur_scenario.add(days=1, **cur_Theta_dict_perturbed)
            
            # cur_scenario.add(days=len_period, **cur_Theta_dict_perturbed)
            # snl_summary = cur_scenario._summary()
            # cur_phase = snl_summary.index[-1]
            # cur_start, cur_end = snl_summary.iloc[-1].loc[['Start', 'End']]
            sim_df = cur_scenario.simulate(variables=['Susceptible', 'Exposed', 'Infectious', 'Isolated', 'Recovered', 'Dead'], name="Main")
            df_states.append(sim_df.set_index('Date').loc[:cur_end])
            # update the phase tracker with sim_df
            track_df = cur_scenario._tracker_dict["Main"]._track_df
            track_df.update(sim_df.set_index(['Date']).loc[cur_start:cur_end])
            cur_scenario._tracker_dict["Main"]._track_df = track_df.copy()
            # cur_scenario.clear()
        print(f'current period start and end time: {cur_start}, {cur_end}')
        
        #### treat simulated diagnosed compartment as scenarios, offline data as bases
        df_observed = df_synthetic_diagnosed.loc[cur_start:cur_end] # add 7 days/period
        # df_bases = pd.concat([tmp_df.Isolated.loc[cur_start:cur_end] for tmp_df in df_simulated], axis = 1)
        # df_bases.columns = range(df_bases.shape[1])
        df_bases = offline_data['n_diagnosed'].loc[cur_start:cur_end]
        
        # ## standardize X
        # scaler = StandardScaler()
        # df_bases_standardized = scaler.fit_transform(df_bases)
        # # df_observed_standardized = scaler.transform(df_observed.T)
        # X = sm.add_constant(df_bases_standardized)
        X = df_bases.copy()
        
        # X = df_bases_standardized.copy()
        model = sm.OLS(df_observed, X)
        model_fitted = model.fit()
        cur_beta = model_fitted.params
        # cur_bse = model_fitted.bse
        df_predicted = X @ model_fitted.params
        df_predicted = pd.DataFrame(df_predicted, index=df_observed.index)
        # df_predicted_ub = X @ (cur_beta + 2 * cur_bse)
        # df_predicted_lb = X @ (cur_beta - 2 * cur_bse)
        
        # visualize
        fig, ax = plt.subplots()
        df_observed.loc[cur_start:cur_end].plot(ax=ax, style = 'ro-', label = 'New observation')
        df_predicted.loc[cur_start:cur_end].plot(ax=ax, style = 'b*-', label = 'Observation recovery')
        # df_predicted_ub.loc[cur_start:cur_end].plot(ax=ax, style = 'g--', label = 'CI')
        # df_predicted_lb.loc[cur_start:cur_end].plot(ax=ax, style = 'g--', label = 'CI')
        bases_ax = df_bases.plot(ax=ax, style='k-', alpha=0.5, label=None)
        plt.axvline(x=cur_start)
        # plt.setp(bases_ax, label="_")
        plt.legend()
        
        #### use the estimated coefficients to inform the parameters(except lamda) for future period, value range for lamda
        # calculate cur_Theta based on offline_parameters
        cur_Theta_dict = {}
        cur_Theta_sd_dict = {}
        for param_name in parameter_names:
            if param_name == 'lamda':
                df_tmp = offline_parameters[param_name]
                df_tmp = df_tmp.loc[cur_end + timedelta(days=1):cur_end + timedelta(days=len_period)]
                lamda_mean = (df_tmp @ cur_beta).mean() #.to_numpy()[0]
                lamda_sd = (df_tmp @ cur_beta).std() #.to_numpy()
                lamda_sum = (df_tmp @ cur_beta).sum()
                print(lamda_mean, lamda_sd)
                lamda_lb, lamda_ub = max(0, lamda_mean - 2 * lamda_sd), max(lamda_mean + 2 * lamda_sd, 0)
            else:
                df_tmp = offline_parameters[param_name]
                # get the parameters for the next period
                df_tmp = df_tmp.loc[cur_end + timedelta(days=1):cur_end + timedelta(days=len_period)]
                cur_Theta_dict[param_name] = (df_tmp @ cur_beta).mean()
                cur_Theta_sd_dict[param_name] = (df_tmp @ cur_beta).std()
        
        # when there is not enough data, break
        if cur_end > df_synthetic_diagnosed.index.max():
            break
        
        # #### optimizing over lamda
        #### get the value of fitness function for each scenario, assign weights
        mse_errors = []
        for scenario_idx in range(n_scenarios):
            df_simulated = df_states[scenario_idx][['Isolated']]
            mse_errors.append(np.sum(abs(df_simulated[cur_start:cur_end].to_numpy() - df_observed.to_numpy())))
        reciprocals = [1./val for val in mse_errors]
        weights = np.array(reciprocals) / sum(reciprocals)
        # print(weights)
        
        # search for the best lamda in [lb, ub]
        # for each lamda, calculate the expected # of death in the future period
        #### determine search space for lamda
        next_start = cur_end + timedelta(days = 1)
        next_end = cur_end + timedelta(days=len_period)
        cur_t = (next_end - start_date).days
        cur_max_lamda = constraint_lamda(cur_t, const_type)
        cur_budget = get_budget_lamda(cur_t, const_type)
        cur_budget = min(cur_budget, lamda_sum) if lamda_sum > 0 else cur_budget
        
        ## get candidates with sum equals cur_budget
        engine = LatinHypercube(d=len_period)
        lhs_sample = engine.random(n=20)
        lhs_sample /= lhs_sample.sum(axis=1, keepdims=True)
        lamda_candidates = lhs_sample * cur_budget
        if lamda_lb >= lamda_ub:
            lamda_candidates = np.ones((1,len_period)) * lamda_lb
        print(f'cur_max_lamda: {cur_max_lamda}; cur_budget: {cur_budget}; num_candidates: {lamda_candidates.shape[0]}')
        
        # if lamda_lb < lamda_ub:
        #     lamda_candidates = np.linspace(lamda_lb, lamda_ub, 10)
        # else:
        #     lamda_candidates = [lamda_lb]
        
        
        weighted_future_val = []
        for lamda_idx, cur_lamda_array in enumerate(lamda_candidates):
            scenarios_list_copy = copy.deepcopy(scenarios_list)
            future_val_list = []
            for scenario_idx, cur_scenario in enumerate(scenarios_list_copy):
                # add daily lamda, one by one
                for cur_lamda in cur_lamda_array:
                    cur_scenario.add(days=1, **{'lamda': cur_lamda})
                # snl_summary = cur_scenario._summary()
                # cur_phase = snl_summary.index[-1]
                # tmp_start, tmp_end = snl_summary.iloc[-1].loc[['Start', 'End']]
                sim_df = cur_scenario.simulate(variables=['Susceptible', 'Exposed', 'Infectious', 'Isolated', 'Recovered', 'Dead'], name="Main")
                df_simulated = sim_df.set_index('Date').loc[next_start:next_end][['Susceptible']]
                future_val_list.append(df_simulated) 
            future_val_df = pd.concat(future_val_list, axis=1)
            cur_weighted_res = (future_val_df @ weights).mean() # T x S s x 1
            weighted_future_val.append(cur_weighted_res)
        # print(weighted_future_val)
        best_idx = weighted_future_val.index(max(weighted_future_val))
        
        best_lamda = lamda_candidates[best_idx]
        print(f'best_lamda: {best_lamda}')
        
            
#%% compare the simulated n_diagnosed with the observed one

    
fig, ax = plt.subplots()
for df in df_states:
    df.Isolated.plot(ax = ax, style='b*-', label='Simulated')
df_synthetic_diagnosed.plot(ax = ax, style='ro-', label='Observed')
ax.legend()
ax.set_title('Isolated')
plt.show()
# plt.savefig('opt_' + str(n_scenarios) + '_Isolated.png')    

fig, ax = plt.subplots()
for df in df_states:
    df.Susceptible.plot(ax = ax, style='b*-', label='Simulated')
# df_synthetic_diagnosed.plot(ax = ax, style='ro-', label='Observed')
ax.legend()
ax.set_title('Susceptible')
plt.show()
# plt.savefig('opt_' + str(n_scenarios) + '_Susceptible.png')   

fig, ax = plt.subplots()
for df in df_states:
    df.Exposed.plot(ax = ax, style='b*-', label='Simulated')
# df_synthetic_diagnosed.plot(ax = ax, style='ro-', label='Observed')
ax.legend()
ax.set_title('Exposed')
plt.show()

fig, ax = plt.subplots()
for df in df_states:
    df.Infectious.plot(ax = ax, style='b*-', label='Simulated')
ax.legend()
ax.set_title('Infectious')
plt.show()

fig, ax = plt.subplots()
for df in df_states:
    df.Recovered.plot(ax = ax, style='b*-', label='Simulated')
ax.legend()
ax.set_title('Recovered')
plt.show()


fig, ax = plt.subplots()
for df in df_states:
    df.Dead.plot(ax = ax, style='b*-', label='Simulated')
ax.legend()
ax.set_title('Dead')
plt.show()
# plt.savefig('opt_' + str(n_scenarios) + '_Dead.png') 




#%% enumerate numpy array
tmp_array = lhs_sample.copy()
for idx, tmp in enumerate(tmp_array):
    print(idx, tmp)



