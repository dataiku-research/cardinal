====================================
Welcome to cardinal's documentation!
====================================

cardinal is a python package that allows simple and scikit-learn compatible active learning
experiments. It provides the building blocks necessary to perform simple or more complex
query sampling strategies.

The code of the project is on Github: `cardinal <https://github.com/dataiku-research/cardinal>`_


.. _install:

Installation
============

Using Pypi
----------

cardinal can be installed through Pypi using:

.. code-block:: bash

    $ pip install "cardinal[extra]"

Extras bring additional features to cardinal:

* ``sklearn`` allows to use the KMeansSampler and RankedBatchModeSampler
* ``examples`` intalls all packages necessary to run the examples
* ``submodular`` installs apricot-select, and allows using SubmodularSampler
* ``doc`` installs sphinx-gallery to be able to generate the documentation
* ``all`` is an alias on all of the above.

Installing locally
------------------

You can also fetch the code and install the package from your local repository.
Again, the preferred way is to use pip.

.. code-block:: bash

    $ git clone https://github.com/dataiku-research/cardinal
    $ cd cardinal
    $ pip install -e ".[extra]"


.. toctree::
   :maxdepth: 2
   :caption: Active learning

   introduction
   uncertainty

.. toctree::
   :maxdepth: 2
   :caption: Example galleries

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: API and developer reference

   reference
   changes
   Fork cardinal on Github <https://github.com/dataiku-research/cardinal>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
