#----------------- change working directory --------------------
import os
os.chdir('C:\Xin\work\github\covid19-sir')
os.getcwd()

#----------------- import --------------------
from pprint import pprint
import covsirphy as cs
cs.__version__

#---------------- simple SIR model ------------
# % SIR example
example_data = cs.ExampleData(tau=1440, start_date="01Jan2020")
# Check records
example_data.cleaned()
# Model name
print(cs.SIR.NAME)
# Example parameter values
pprint(cs.SIR.EXAMPLE, compact=True)

model = cs.SIR
area = {"country": "Full", "province": model.NAME}
# Add records with SIR model
example_data.add(model, **area)

# Records with model variables
df = example_data.specialized(model, **area)
df.head()

#%% SIRF example
print(cs.SIRF.NAME)
# Example parameter values
pprint(cs.SIRF.EXAMPLE, compact=True)

model = cs.SIRF
area = {"country": "Full", "province": model.NAME}
# Add records with SIR model
example_data.add(model, **area)
# Records with model variables
df = example_data.specialized(model, **area)
cs.line_plot(df.set_index("Date"), title=f"Example data of {model.NAME} model", y_integer=True)


#%%%%%%% scenarios comparison

# Preset of SIR-F parameters and initial values
preset_dict = cs.SIRF.EXAMPLE["param_dict"]
preset_dict
area = {"country": "Theoretical"}
# Create dataset from 01Jan2020 to 31Jan2020
example_data.add(cs.SIRF, step_n=30, **area)
# Create Scenario instance
snl = cs.Scenario(tau=1440, **area)
snl.register(example_data)
# Show records with Scenario instance
record_df = snl.records()
record_df.head()
record_df.tail()

# Set 0th phase from 02Jan2020 to 31Jan2020 with preset parameter values
snl.clear(include_past=True)
snl.add(end_date="31Jan2020", model=cs.SIRF, **preset_dict)
# Show summary
snl.summary()

# Add main scenario
snl.add(end_date="31Dec2020", name="Main")
snl.summary()

# Add lockdown scenario
snl.clear(name="Lockdown")
# Get rho value of the 0th phase and halve it
rho_lock = snl.get("rho", phase="0th") * 0.5
# Add th 1st phase with the calculated rho value
snl.add(end_date="31Dec2020", name="Lockdown", rho=rho_lock)

# Add medicine scenario
snl.clear(name="Medicine")
kappa_med = snl.get("kappa", phase="0th") * 0.5
sigma_med = snl.get("sigma", phase="0th") * 2
snl.add(end_date="31Dec2020", name="Medicine", kappa=kappa_med, sigma=sigma_med)

# Add vaccine scenario
snl.clear(name="Vaccine")
rho_vac = snl.get("rho", phase="0th") * 0.8
kappa_vac = snl.get("kappa", phase="0th") * 0.6
sigma_vac = snl.get("sigma", phase="0th") * 1.2
snl.add(end_date="31Dec2020", name="Vaccine",  rho=rho_vac, kappa=kappa_vac, sigma=sigma_vac)

# Show summary
snl.summary()
# Show the history of rho as a dataframe and a figure
# we can set theta/kappa/rho/sigma for SIR-F model
snl.history(target="rho").head()
# Describe the scenarios
snl.describe()

# The number of infected cases
_ = snl.history(target="Infected")
# The number of fatal cases
_ = snl.history(target="Fatal")

# Main scenario
_ = snl.simulate(name="Main")
# Lockdown scenario
_ = snl.simulate(name="Lockdown")
# Medicine scenario
_ = snl.simulate(name="Medicine")
# Vaccine scenario
_ = snl.simulate(name="Vaccine")