from pprint import pprint
import covsirphy as cs

# Standard users and developers
data_loader = cs.DataLoader("../input")
# The number of cases and population values
jhu_data = data_loader.jhu()
snl = cs.Scenario(country="Italy", province=None)
snl.register(jhu_data)
df = snl.records()
df.tail()

_ = snl.trend()
snl.summary()
# Default value of timeout is 180 sec
snl.estimate(cs.SIRF, timeout=30)

_ = snl.history(target="Rt")
_ = snl.history_rate()
# Add a phase with 30 days from the date of the last record
snl.add(days=30)
_ = snl.simulate()
