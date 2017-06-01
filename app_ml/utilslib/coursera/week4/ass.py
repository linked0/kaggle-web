import pandas as pd
import numpy as np

data = pd.read_csv('user-row.csv')
data_corr = data.corr()