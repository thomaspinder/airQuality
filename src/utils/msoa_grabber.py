import pandas as pd

household_data = pd.read_excel('../data/msoa_household.xls', sheet_name='Total annual income', skiprows=4)
household_data.to_csv('../data/msoa_household.csv', index=False)
