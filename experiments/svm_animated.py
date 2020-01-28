"""
Hello
=====

This is a test
"""


from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import numpy as np
from cardinAL.random import RandomSampler
from cardinAL.uncertainty import UncertaintySampler
from copy import copy
import matplotlib.animation as animation
from matplotlib.cm import tab20


# Variables
n = 30
batch_size = 2
n_iter = 5

# Initialize the data
np.random.seed(8)

X, y = make_blobs(n_samples=n, centers=2,
                  random_state=0, cluster_std=0.80)

model = SVC(kernel='linear', C=1E10, probability=True)

# Vars for plotting purpose
min_X = np.min(X[:, 0])
max_X = np.max(X[:, 0])

# We need to run the simulation in advance
al_models = [
    ('random', RandomSampler(batch_size=batch_size, random_state=0)),
    ('uncertainty', UncertaintySampler(model, batch_size))
]

# We force having one sample in each class for the init
init_idx = [np.where(y == 0)[0][0], np.where(y == 1)[0][0]]


results = dict()
for (name, al_model) in al_models:
    selected = np.zeros(n, dtype=bool)
    selected[init_idx] = True
    result = []
    for i in range(n_iter):
        model.fit(X[selected], y[selected])
        al_model.fit(X[selected], y[selected])
        w = model.coef_[0]
        result.append((-w[0] / w[1], - model.intercept_[0] / w[1], model.score(X, y), selected.copy()))
        new_selected = al_model.predict(X[~selected])
        selected[~selected] = new_selected
    results[name] = result


fig = plt.figure()
label = plt.text(.5, .5, '', fontsize=15)
artists = dict()

ax = plt.subplot(1, 2, 1)
line, = plt.plot([], [])
sca = plt.scatter(np.arange(4), np.arange(4), c=np.arange(4), cmap=tab20)
plt.axis('off')
plt.xlim(min_X - 0.5, max_X + 0.5)
plt.ylim(np.min(X[:, 1]) - 0.5,np.max(X[:, 1]) + 0.5)
label = plt.text(.5, 0., '', horizontalalignment='center', fontsize=10, transform=ax.transAxes)
artists['random'] = (line, sca, label)

ax = plt.subplot(1, 2, 2)
line, = plt.plot([], [])
sca = plt.scatter(np.arange(4), np.arange(4), c=np.arange(4), cmap=tab20)
plt.axis('off')
plt.xlim(min_X - 0.5, max_X + 0.5)
plt.ylim(np.min(X[:, 1]) - 0.5,np.max(X[:, 1]) + 0.5)
label = plt.text(.5, 0., '', horizontalalignment='center', fontsize=10, transform=ax.transAxes)
artists['uncertainty'] = (line, sca, label)

def plot(name, i):
    a, b, score, selected = results[name][i]
    line, sca, label = artists[name]
    label.set_text('{} {}%'.format(name.capitalize(), int(score * 100)))
    y_ = y.copy()
    y_[~selected] = 3 - y_[~selected]
    sca.set_offsets(X)
    sca.set_array(y_)
    sca.set_cmap('tab20')
    f = lambda x: a * x + b
    xx = np.asarray([min_X, max_X])
    line.set_xdata(xx)
    line.set_ydata(f(xx))
    return [sca, line, label]

def plot_all(i):
    ra = plot('random', i)
    un = plot('uncertainty', i)
    return ra + un

ani = animation.FuncAnimation(fig, plot_all, np.arange(n_iter)) #, fps=10, interval=500, blit=True)
#ani.to_jshtml('test.html')
#ani.save('svm.gif')
plot_all(0)
plt.show()
