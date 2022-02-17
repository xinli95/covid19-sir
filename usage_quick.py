
# change working directory
import os
os.chdir('C:\Xin\work\github\covid19-sir')
os.getcwd()

from pprint import pprint
import covsirphy as cs

loader = cs.DataLoader("../input", update_interval= 24)
# The number of cases and population values
jhu_data = loader.jhu()
# Government Response Tracker (OxCGRT)
oxcgrt_data = loader.oxcgrt()
# The number of tests
pcr_data = loader.pcr()
# The number of vaccinations
vaccine_data = loader.vaccine()
# Mobility data
mobility_data = loader.mobility()

data_dict = loader.collect()
snl = cs.Scenario(country="Japan", province=None)
snl.register(**data_dict)

# Specify country and province (optinal) names
snl = cs.Scenario(country="Japan", province=None)
# Register datasets
snl.register(jhu_data, extras=[oxcgrt_data, pcr_data, vaccine_data])

snl.records(variables="CFR").tail()
# This is the same as
# snl.records(variables=["Confirmed", "Fatal", "Recovered"])

_ = snl.records(variables="I")
# This is the same as
# snl.records(variables=["Infected"])

df = snl.records(variables="all", show_figure=False)
pprint(df.set_index("Date").columns.tolist(), compact=True)
snl.records(variables=["Vaccinations"]).tail()
# Acceptable variables are the same as Scenario.records()
_ = snl.records_diff(variables="C", window=7)
# Show the details of complement
snl.show_complement()

###################### S-R trend analysis ###############
snl.trend().summary()
# Estimate the tau value and parameter values of SIR-F model
snl.estimate(cs.SIRF)
# Show the summary of parameter estimation
snl.summary()
# Show RMSLE scores with the number of optimization trials and runtime for phases
snl.summary(columns=["Start", "End", "RMSLE", "Trials", "Runtime"])
# Visualize the accuracy for the 2nd phase
snl.estimate_accuracy(phase="2nd")
# phase="last" means the last phases
# snl.estimate_accuracy(phase="last")
# Get total score
# snl.score(metrics="RMSLE")
metrics_list = ["MAE", "MSE", "MSLE", "RMSE", "RMSLE", "MAPE"]
for metrics in metrics_list:
    metrics_name = metrics.rjust(len(max(metrics_list, key=len)))
    print(f"{metrics_name}: {snl.score(metrics=metrics):.3f}")

# Get parameter values
snl.get("Rt", phase="4th")
# phase="last" means the last phases
snl.get("Rt", phase="last")
# Get the parameter values as a dataframe
snl.summary(columns=[*cs.SIRF.PARAMETERS, "Rt"])

# Compare the actual values and the main scenario
_ = snl.history("Infected")
_ = snl.simulate(name="Main")

###################### Scenario analysis ################
snl.clear(name="Main")
# Add one future phase 30 days with the parameter set of the last past phase
snl.add(days=30, name="Main")
# Add one future phase until 28Feb2022 with the same parameter set
snl.add(end_date="28Feb2022", name="Main")
# Simulate the number of cases
snl.simulate(name="Main").tail()

# Calcuate the current sigma value of the last phase
sigma_current = snl.get("sigma", name="Main", phase="last")
sigma_current
# Sigma value will be double
sigma_new = sigma_current * 1.2
sigma_new
# Initialize "Medicine" scenario (with the same past phases as that of Main scenario)
snl.clear(name="Medicine")
# Add 30 days as a new future phases with the same parameter set
snl.add(name="Medicine", days=30, sigma=sigma_current)
# Add a phase with doubled sigma value and the same end date with main date
snl.add(name="Medicine", end_date="28Feb2022", sigma=sigma_new)
df = snl.summary()
df.loc[df["Type"] == "Future"]
_ = snl.simulate(name="Medicine").tail()

#################### short-term prediction of parameter values ##########
# Create Forecast scenario (copy Main scenario and delete future phases)
snl.clear(name="Forecast", template="Main")
# Fitting with linear regression model (Elastic Net regression)
fit_dict = snl.fit(
    name="Forecast", metric="MAPE", regressors=None,
    removed_cols=["Stringency_index", "Tests", "Tests_diff", "Vaccinations"]
)
# Short-term prediction
snl.predict(name="Forecast")
# We can select list of days to predict optionally
# snl.predict(days=[1, 4], name="Forecast")
df = snl.summary()
df.loc[df["Type"] == "Future"]
# Or, when you do not need 'fit_dict',
# snl.fit_predict(name="Forecast").summary(name="Forecast")
# Adjust the last end dates
snl.adjust_end()
# Show the last phases of all scenarios
all_df = snl.summary().reset_index()
for name in all_df["Scenario"].unique():
    df = snl.summary(name=name)
    last_end_date = df.loc[df.index[-1], "End"]
    print(f"{name} scenario: to {last_end_date}")
_ = snl.simulate(variables="CFR", name="Forecast").tail()
_ = snl.history("Infected")

# Get the minimum value (from today to future) to set lower limit of y-axis
lower_limit = snl.history("Infected", dates=(snl.today, None), show_figure=False).min().min()
# From today to future (no limitation regarding end date)
_ = snl.history("Infected", dates=(snl.today, None), ylim=(lower_limit, None))
_ = snl.history("Infected", past_days=20)
_ = snl.history("Infected", phases=["3rd", "4th", "5th"])

snl.describe()
_ = snl.history(target="Infected")
