====================================
Welcome to cardinAL's documentation!
====================================

cardinAL is a python package that allows simple and scikit-learn compatible active learning
experiments. It provides the building blocks necessary to perform simple or more complex
query sampling strategies.

The code of the project is on Github: `cardinAL <https://github.com/dataiku/cardinAL>`_


.. _install:

Installation
============

Fetch the code
--------------

cardinAL is still in alpha development phase and thus not available through
the Pypi repository. In order to install cardinAL, one must download its source
code and perform the installation manually.

You will also need to install the dependencies list in the requirements file.

To install everything do:

.. code-block:: bash

    $ git clone https://github.com/dataiku/cardinAL
    $ cardinAL
    $ pip install -r requirements.txt
    $ pip install -e .

In addition, you will need the following dependencies to build the
``sphinx-gallery`` documentation:

* sphinx
* sphinx-gallery


.. toctree::
   :maxdepth: 2
   :caption: Using cardinAL

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
   Fork cardinAL on Github <https://github.com/dataiku/cardinAL>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
