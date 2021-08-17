from copy import copy

from .version import check_modules

check_modules('examples', 'plotting')  # noqa

from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np


def plot_confidence_interval(*args, label=None, q_inf=0.1, q_sup=0.9, alpha=.3, smoothing=None, dots=False):
    y_data = np.asarray(args[-1])
    if len(args) == 1:
        # We only have the data. We create x-axis values.
        x_data = np.arange(y_data.shape[0])
    else:
        x_data = np.asarray(args[0])

    avg = np.mean(y_data, axis=0)
    q10 = np.quantile(y_data, q_inf, axis=0)
    q90 = np.quantile(y_data, q_sup, axis=0)

    if smoothing is not None:
        x_plot = np.linspace(x_data.min(), x_data.max(), x_data.shape[0] * smoothing) 
        avg_plot = make_interp_spline(x_data, avg, k=2)(x_plot)
        q10_plot = make_interp_spline(x_data, q10, k=2)(x_plot)
        q90_plot = make_interp_spline(x_data, q90, k=2)(x_plot)
    else:
        x_plot = x_data
        avg_plot = avg
        q10_plot = q10
        q90_plot = q90
        
    line = plt.plot(x_plot, avg_plot, label=label)
    color = line[0].get_c()

    if dots:
        plt.scatter(x_data, avg, c=color)

    plt.fill_between(x_plot, q90_plot, q10_plot, color=color, alpha=alpha)


def smooth_line(line, smoothing=10, k=2):
    x = line.get_xdata()
    y = line.get_ydata()
    x_plot = np.linspace(x.min(), x.max(), x.shape[0] * smoothing) 
    y_plot = make_interp_spline(x, y, k=k)(x_plot)
    line.set_xdata(x_plot)
    line.set_ydata(y_plot)


def smooth_lines(axis=None, smoothing=10, k=2):
    if axis is None:
        axis = plt.gca()
    
    for line in axis.lines:
        smooth_line(line, smoothing=smoothing, k=k)
