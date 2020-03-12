====================================
Welcome to cardinal's documentation!
====================================

cardinal is a python package that allows simple and scikit-learn compatible active learning
experiments. It provides the building blocks necessary to perform simple or more complex
query sampling strategies.

The code of the project is on Github: `cardinal <https://github.com/dataiku/cardinal>`_


.. _install:

Installation
============

Fetch the code
--------------

cardinal is still in alpha development phase and thus not available through
the Pypi repository. In order to install cardinal, one must download its source
code and perform the installation manually.

You will also need to install the dependencies list in the requirements file.

To install everything do:

.. code-block:: bash

    $ git clone https://github.com/dataiku/cardinal
    $ cardinal
    $ pip install -r requirements.txt
    $ pip install -e .

In addition, you will need the following dependencies to build the
``sphinx-gallery`` documentation:

* sphinx
* sphinx-gallery


.. toctree::
   :maxdepth: 2
   :caption: Using cardinal

   introduction

.. toctree::
   :maxdepth: 2
   :caption: Example galleries

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: API and developer reference

   reference
   changes
   Fork cardinal on Github <https://github.com/dataiku/cardinal>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
