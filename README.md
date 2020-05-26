<img src="doc/_static/cardinal.png" width="300">

## Introduction

cardinal is a Python package to perform and monitor Active Learning experiments
leveraging various query sampling methods and metrics. 

The project is currently maintained by [Dataiku's](https://www.dataiku.com/) research team.

## Getting started

Cardinal extensive [documentation](https://dataiku.github.io/cardinal/) features some examples helping you getting started with Active Learning:
* [Lowest confidence vs. Random sampling](https://dataiku.github.io/cardinal/auto_examples/plot_random_vs_confidence.html) presents a basic active learning pipeline and explains why it is better than random
* [Lowest confidence vs. KMeans sampling](https://dataiku.github.io/cardinal/auto_examples/plot_confidence_vs_diversity.html) presents more advanced techniques
* [Active learning on digit recognition and metrics](https://dataiku.github.io/cardinal/auto_examples/plot_digits_metrics.html) presents an experiment on MNIST dataset and proposes some metrics to estimate the accuracy uplift during an experiment

## cardinal taking off  

Let `X_unlabeled` the pool of unlabeled data to be labeled and `(X_labeled, y_labeled)` the original labeled data to train our model.
One iteration of Active Learning can be written as:

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from cardinal.uncertainty import ConfidenceSampler

model = RandomForestClassifier()
batch_size = 20
sampler = ConfidenceSampler(model, batch_size)

model.fit(X_labelled, y_labelled)  
sampler.fit(X_labelled, y_labelled)
selected = sampler.select_samples(X_unlabelled)

#Updating the labeled and unlabeled pool
X_labelled = np.concatenate([X_labelled, selected])
#The selected samples are sent to be labeled as y_selected
y_labelled = np.concatenate([y_labelled, y_selected])
```

But how to evaluate the performance of the Active Learning process ?

Active Learning comes in two flavors: with *fixed testing set* and with *incremental testing set*. The former is almost always the only one proposed in the fixed environement of the Active Learning literature while the latter is most common in the wild.  

* In the *fixed testing set*, there is already a large enough and representative testing set for the task at hand. This corresponds to the situation where a model has already been trained and tested, perhaps even deployed. As new data comes in, the machine learning practitioner can both score it with the existing model or manually label it. The same testing set will be used to evaluate potential additional performance gain.   

Let `(X_test, y_test)` denote the fixed testing set. The above then becomes:

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from cardinal.uncertainty import ConfidenceSampler

model = RandomForestClassifier()
batch_size = 20
sampler = ConfidenceSampler(model, batch_size)
accuracies = []

model.fit(X_labelled, y_labelled)  
sampler.fit(X_labelled, y_labelled)
selected = sampler.select_samples(X_unlabelled)

#Evaluating performance
accuracies.append(model.score(X_test, y_test))

#Updating the labeled and unlabeled pool
X_labelled = np.concatenate([X_labelled, selected])
#The selected samples are sent to be labeled as y_selected
y_labelled = np.concatenate([y_labelled, y_selected])
```

* When starting a new machine learning project and data has to be collected and labeled, we are in the *incremental testing set* settings. There is no ground truth labelled set to start with and part of the new labeled data will make the testing set at each labeling iteration.
This is the corresponding Active Learning iteration:

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from cardinal.uncertainty import ConfidenceSampler
from sklearn.model_selection import train_test_split

model = RandomForestClassifier()
batch_size = 20
sampler = ConfidenceSampler(model, batch_size)
accuracies = []

X_train, X_test, y_train, y_test =  train_test_split(X_labelled, y_labelled, test_size=0.2, random_state=123)
model.fit(X_train, y_train)  
sampler.fit(X_train, y_train)
selected = sampler.select_samples(X_unlabelled)

#Evaluating performance
accuracies.append(model.score(X_test, y_test))

#Updating the labeled and unlabeled pool
X_labelled = np.concatenate([X_labelled, selected])
#The selected samples are sent to be labeled as y_selected
y_labelled = np.concatenate([y_labelled, y_selected])
```

Here it is important to note that contrary to the beautiful learning curves from the literature or our documentation, in this setting
it can be non-monotonic when using small sample sizes ¯\_(ツ)_/¯.


## Installation

### Dependencies

cardinal depends on:
- Python >= 3.5
- NumPy >= 1.11
- SciPy >= 0.19
- scikit-learn >= 0.19 (optional)
- matplotlib >= 2.0 (optional)
- apricot-select >= 0.5.0 (optional)

Additional features are available in cardinal through different options:
* `sklearn` requires scikit-learn and provides a KMeans based sampler and a Batch method
* `submodular` requires apricot-select and scikit-learn. It allows to use a query sampler
  based on a submodular facility location problem solver.
* `examples` requires scikit-learn, apricot-select, and matplotlib. It provides plotting
  abilities and all the packages necessary to run the examples.
* `all` includes all of the above.


### Installing with pip

The easiest way to install cardinal is to use `pip`. For a vanilla install, simply type:

    pip install -U cardinal

Optional dependencies are also handled by `pip` in the following way:

    pip install -U 'cardinal[option]'

*option* can be one of:
- *sklearn* to enable scikit-learn related samplers such as clustering based ones
- *submodular* to install apricot-select to run the submodular sampler
- *examples* to install all required dependencies to run the examples
- *doc* to install the required dependencies to generate the sphinx-based documentation
- *all* to install all of the above

## Contributing

Contributions are welcome. Check out our [contributing guidelines](CONTRIBUTING.md).
