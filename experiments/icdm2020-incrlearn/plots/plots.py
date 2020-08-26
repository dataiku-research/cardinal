import dataset
import pandas as pd
from matplotlib import pyplot as plt
import os
import matplotlib
import numpy as np
import matplotlib.ticker as ticker

import sys
sys.path.append("../")
from icdm2020 import init_figure, plot_by_method, save_figure, iters

font = {'size': 16}

matplotlib.rc('font', **font)

ds = os.path.basename(os.getcwd())

db = dataset.connect('sqlite:///database.db')

for name in ['accuracy', 'top_exploration', 'hard_contradiction', 'batch_agreement', 'batch_difficulty']:

    if db[name].count() == 0:
        continue

    df = pd.DataFrame(list(db[name].find(method={'not': 'experimental'})))

    if name == 'hard_contradiction':
        df['value'] = 1 - df['value']
    
    init_figure()
    
    plot_by_method(df, log=(name == 'top_exploration'))
    
    legend_kwargs = {}
    if name == "accuracy":
        legend_kwargs['loc'] = 4

    if name in ['accuracy', 'hard_contradiction', 'batch_agreement', 'batch_difficulty']:
        formatter = ticker.FormatStrFormatter('%.2f')
        plt.gca().yaxis.set_major_formatter(formatter)
    
    # if name in ['batch_difficulty']:
    #     legend_kwargs = None           

    save_figure(ds, name, legend_kwargs=legend_kwargs)
    
    plt.close()
