import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import numpy as np
import pandas as pd
import os

directory = 'ContinuousV1/Results'

headers = ['Episode', 'Reward']

single_file = True
only_file = "agent[32, 64]_0.001_1"

if single_file:
    f = os.path.join(directory, only_file)
    df = pd.read_csv(f, index_col=0)
    df.plot()
else:
    figure, axis = plt.subplots(len(os.listdir(directory)), sharex=True, sharey=True)

    for i, file in enumerate(os.listdir(directory)):
        f = os.path.join(directory, file)
        df = pd.read_csv(f, index_col=0)
        df.rename(columns = {'Rewards':file}, inplace = True)
        df.plot(ax=axis[i], subplots=True, label=file, legend=False)
        axis[i].legend(loc=4)

plt.show()