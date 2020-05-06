![](doc/_static/cardinal.png)

# cardinal

cardinal is a Python package for active learning. It proposes a framework to
perform active learning experiments along with query sampling methods and
metrics.

It is currently maintained by the research team of [Dataiku](https://www.dataiku.com/).

[cardinal's website](https://dataiku.github.io/cardinal/).

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
