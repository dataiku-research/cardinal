import dataset
import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
from matplotlib import cm
import matplotlib


names = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "mnist": "MNIST",
    "fashion": "Fashion",
    "nomao": "NOMAO",
    "phishing": "Phishing",
    "news": "20 NG",
    "ldpa": "LDPA",
    "wall_robot": "Robot",
    "uncertainty": "Confidence",
    "margin": "Margin",
    "entropy": "Entropy",
    "kmeans": "KMeans",
    "wkmeans": "WKMeans",
    "random": "Random",
    "accuracy": "Accuracy",
    "hard_contradiction": "Contradiction ratio",
    "top_exploration": "Exploration",
    "batch_difficulty": "Batch easiness ratio",
    "batch_agreement": "Batch classifier agreement ratio",
}

iters = {
    "cifar10": np.arange(1000, 10001, 1000),
    "cifar100": np.arange(1000, 10001, 1000),
    "mnist": np.arange(100, 5000, 200),
    "fashion": np.arange(100, 5000, 200),
    "nomao": np.arange(20, 401, 20),
    "phishing": np.arange(50, 1001, 50),
    "news": np.arange(100, 2001, 100),
    "ldpa": np.arange(100, 3001, 100),
    "wall_robot": np.arange(15, 226, 15),
}


def namify(tag):
    return tag.replace('_', ' ').capitalize()

# We want to have a uniform selection of style / colors in all plots
mpl_options = {
    'random': dict(c=cm.tab10(0), linestyle='solid'),
    'entropy': dict(c=cm.tab10(1), linestyle='dotted'),
    'wkmeans': dict(c=cm.tab10(2), linestyle='dashed'),
    'kmeans': dict(c=cm.tab10(3), linestyle='dashdot'),
    'margin': dict(c=cm.tab10(4), linestyle=(0, (3, 1, 1, 1))),
    'uncertainty': dict(c=cm.tab10(5), linestyle=(0, (3, 1, 1, 1, 1, 1))),
    "cifar10": dict(c=cm.tab20(1)),
    "cifar100": dict(c=cm.tab20(3)),
    "mnist": dict(c=cm.tab20(5)),
    "fashion": dict(c=cm.tab20(7)),
    "nomao": dict(c=cm.tab20(9)),
    "phishing": dict(c=cm.tab20(11)),
    "news": dict(c=cm.tab20(13)),
    "ldpa": dict(c=cm.tab20(15)),
    "wall_robot": dict(c=cm.tab20(17)),
}


def init_figure():
    plt.figure(figsize=(10, 8))
    plt.grid(zorder=0)


def plot_by_method(df, cumsum=False, log=False):

    transform = lambda x: x
    if cumsum:
        transform = lambda x: np.cumsum(x)

    for method, mdf in df.groupby('method'):

        if method not in ['random', 'margin', 'entropy', 'uncertainty', 'kmeans', 'wkmeans']:
            continue

        gmdf = mdf.groupby('n_samples').agg([
            ('mean',lambda x: np.mean(transform(x))),
            ('q10', lambda x: np.quantile(transform(x), 0.1, axis=0)),
            ('q90', lambda x: np.quantile(transform(x), 0.9, axis=0))
        ])['value'].sort_index()
    
        x = gmdf.index.values
        mean = gmdf['mean'].values
        q10 = gmdf['q10'].values
        q90 = gmdf['q90'].values
    
        # Plot the mean line and get its color
        line = plt.plot(x, mean, label=names.get(method, namify(method)), **mpl_options[method])
        color = line[0].get_c()
    
        # Plot confidence intervals
        plt.fill_between(x, q90, q10, alpha=.3, color=color, zorder=2)
        
        if log:
            plt.xscale('log')
            plt.minorticks_off()
            scale_loc = (np.arange(x.size) ** 1.5).astype(int)
            scale_loc = scale_loc[scale_loc < x.size]
            plt.gcf().autofmt_xdate()
            plt.gca().set_xticks(x[scale_loc])
            plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())    

def save_figure(dataset, metric, legend_kwargs={}):

    if legend_kwargs is not None:
        plt.legend(**legend_kwargs)
    plt.xlabel('Training sample count')
    plt.ylabel(names.get(metric, namify(metric)))
    
    plt.savefig('{}_{}.pdf'.format(dataset, metric), bbox_inches='tight', pad_inches=0)
    plt.close()