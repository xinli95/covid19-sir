# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:08:33 2022

@author: bitsi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
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

ax = df_synthetic.plot(c='k',linestyle='dashed', label='Synthetic')
df_merged.iloc[:45,:].plot( cmap='coolwarm', title='n_diagnosed', ax = ax)
ax.axvline(x=45, c='g')
plt.legend(['Observed'])

#### given the synthetic trajectory, carry out multi-period resource allocation
n_periods = (df_synthetic.shape[0] - 30) // 7 # first 30 days have no variation, 7 days/period
decisions = [] # 'lamda'
variables = [] # variables in the compartmental model: 'beta_1', 'beta_2', 'alpha', 'gamma'
states = [] # states of the compartment models; initial values at the beginning of each period

cur_decision = 0 # no tracing initially
initial_exposed = 20
pop_size = 100e3
cur_states = {'n_susceptible': pop_size - initial_exposed,
              'n_exposed': initial_exposed,
              'n_infectious': 0,
              'n_diagnosed': 0,
              'n_recovered': 0,
              'n_dead': 0}


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
data = {}
for col in compartment_names:
    tmp = collect_data(col)
    tmp.index = pd.date_range(start="3/1/2020", periods=tmp.shape[0])
    data[col] = tmp

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


#%% estimate the phase-wise parameters for the synthetic one
df_synthetic = pd.concat(df_synthetic_dict.values(), axis=1)
df_synthetic.columns = compartment_names

pop_size = 100e3
jhu_data = initialize_dataloader(df_synthetic, pop_size)
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

snl.history("Susceptible", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
snl.history("Exposed", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
snl.history("Infectious", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
snl.history("Isolated", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
snl.history("Recovered", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])
snl.history("Dead", VALUE_COLUMNS=['Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"])

fitted_df = snl._summary()
sub_df = fitted_df[["Start", "End"] + cs.SEIIRD.PARAMETERS]
sub_df['Date'] = sub_df.apply(lambda d: pd.date_range(d['Start'],
                                                    d['End'], 
                                                    freq='D'),
                              axis=1)
heuristic_df = sub_df.explode('Date')
#%% main loop
# from numpy.random import default_rng
SEIIRD_col_names = ['n_susceptible', 'n_exposed', 'n_infectious', 'n_diagnosed', 'n_recovered', 'n_dead']
pop_size = 100e3
model = cs.SEIIRD_Q
model.population = pop_size
rng = np.random.default_rng(12345)
n_samples = 10
df_synthetic_diagnosed = df_synthetic[['n_diagnosed']]
df_synthetic_diagnosed.index = pd.date_range(start="3/1/2020", periods=df_synthetic_diagnosed.shape[0])
df_bases = df_merged.copy()
for i in range(n_periods):
    if i == 0:
        df_observed = df_synthetic_diagnosed[:30] # first period 
        df_others_dict = {col: df.iloc[:30,:].mean(axis=1) for col, df in data.items()}
        df_others_dict.pop('n_diagnosed')
        df_others = pd.DataFrame(df_others_dict)
        df_all = df_others.copy()
        df_all["n_diagnosed"] = df_observed.to_numpy()
        df_all = df_all[SEIIRD_col_names]
        jhu_data = initialize_dataloader(df_all, pop_size)
        snl = cs.Scenario(country="Synthetic", province=None, tau=1440, auto_complement=False)
        # Register datasets
        snl.register(jhu_data)
        snl.trend_SA(min_size=7, aggregate_cols = ["Isolated", "Recovered", "Dead"]) # "Isolated", "Recovered",
        snl.estimate(model=model, metric="MSE", check_dict = {"timeout": 360, "timeout_iteration": 1, "tail_n": 20, "allowance": (0.99, 1.01)},
                 metric_cols = ["Exposed","Isolated", "Infectious"], metric_cols_weights = []
                 )
        # get the last period's parameters as the mean value
        Theta_mean = snl.summary(cs.SEIIRD_Q.PARAMETERS).iloc[-1, :-1]
        Theta_sd = 0.05 * np.ones(Theta_mean.shape[0]) * Theta_mean
        Theta_samples = rng.multivariate_normal(mean=Theta_mean, cov=np.diag(Theta_sd**2), size=n_samples)
        Theta_names = Theta_mean.index.to_list()
        
        # initial decision follows the last period
        Lamda = snl.summary(cs.SEIIRD_Q.PARAMETERS).iloc[-1, -1]

    else:
        # simulate with the initial condition and uncertainty around \Theta (parameters except \lamda)
        
        df_simulated = []
        for Theta_idx in range(n_samples):
            cur_Theta = Theta_samples[Theta_idx,:]
            cur_Theta_dict = {key:val for key, val in zip(Theta_names, cur_Theta)}
            cur_Theta_dict['lamda'] = Lamda
            snl.add(days=7, **cur_Theta_dict)
            snl_summary = snl._summary()
            cur_phase = snl_summary.index[-1]
            cur_start, cur_end = snl_summary.iloc[-1].loc[['Start', 'End']]
            sim_df = snl.simulate(variables=['Susceptible', 'Exposed', 'Infectious', 'Isolated', 'Recovered', 'Dead'], name="Main")
            df_simulated.append(sim_df.set_index('Date').loc[:cur_end])
            snl.clear()
        
        ## treat simulated diagnosed compartment as scenarios, pre-generated cases as bases
        
        df_observed = df_synthetic_diagnosed.loc[cur_start:cur_end] # add 7 days/period
        # df_bases = pd.concat([tmp_df.Isolated.loc[cur_start:cur_end] for tmp_df in df_simulated], axis = 1)
        # df_bases.columns = range(df_bases.shape[1])
        df_bases = data['n_diagnosed'].loc[cur_start:cur_end]
        # # apply standardization
        # scaler = StandardScaler()
        # df_bases_scaled = scaler.fit_transform(df_bases.T)
        # df_observed_scaled = scaler.transform(df_observed.to_numpy().reshape(1,-1))
        # X = df_bases_scaled.T.copy()# sm.add_constant(df_bases_scaled.T)
        # model = sm.OLS(df_observed_scaled.T, X)
        # model_fitted = model.fit()
        # cur_beta = model_fitted.params
        # cur_bse = model_fitted.bse
        # df_predicted_scaled = X @ model_fitted.params
        # df_predicted_ub_scaled = X @ (cur_beta + 0.001 * cur_bse)
        # df_predicted_lb_scaled = X @ (cur_beta - 0.001 * cur_bse)
        # df_predicted = scaler.inverse_transform(df_predicted_scaled.reshape(1,-1))
        # df_predicted_ub = scaler.inverse_transform(df_predicted_ub_scaled.reshape(1,-1))
        
        # X = sm.add_constant(df_bases)
        X = df_bases.copy()
        model = sm.OLS(df_observed, X)
        model_fitted = model.fit()
        cur_beta = model_fitted.params
        cur_bse = model_fitted.bse
        df_predicted = X @ model_fitted.params
        df_predicted_ub = X @ (cur_beta + 1 * cur_bse)
        df_predicted_lb = X @ (cur_beta - 1 * cur_bse)
        
        fig, ax = plt.subplots()
        df_observed.loc[cur_start:cur_end].plot(ax=ax, style = 'ro-', label = 'New observation')
        df_predicted.loc[cur_start:cur_end].plot(ax=ax, style = 'b*-', label = 'Observation recovery')
        df_predicted_ub.loc[cur_start:cur_end].plot(ax=ax, style = 'g--', label = 'CI')
        df_predicted_lb.loc[cur_start:cur_end].plot(ax=ax, style = 'g--', label = 'CI')
        bases_ax = df_bases.plot(ax=ax, style='k-', alpha=0.5, label=None)
        plt.axvline(x=cur_start)
        # plt.setp(bases_ax, label="_")
        plt.legend()
        
        # use the estimated coefficients to inform the parameters(except lamda) for future period
        
        
        # ## find the weight parameter for each basis
        # obs_len = df_observed.shape[0]
        # x_bases = df_bases.iloc[:obs_len,:]
        # X = sm.add_constant(x_bases)
        # model = sm.OLS(df_observed, X, missing='drop')
        # model_fitted = model.fit()
        # cur_beta = model_fitted.params
        # cur_bse = model_fitted.bse
        
        # ## forecast the trajectory of diagnosed compartment based on the weights and its uncertainty
        # ## and the trajectory of other compartments based on the ODE with parameters from last period
        # # generate weight vector from normal distribution
        # if max(cur_bse) < 0.01:
        #     n_samples = 1
        # else:
        #     n_samples = 10
        # weight_samples = rng.multivariate_normal(mean=cur_beta, cov=np.diag(cur_bse**2), size=n_samples)
        # diagnosed_compartment = weight_samples @ df_merged
    
            
    
    



















