.. _installation:


============
Installation
============

Preparation
===========

You need to edit your :code:`PYTHONPATH` to include the rllab directory:

.. code-block:: bash

    export PYTHONPATH=path_to_extr:$PYTHONPATH

Express Install
===============

The fastest way to set up dependencies for extr is via running the setup script.

- On Linux, run the following:

.. code-block:: bash

    ./scripts/setup_linux.sh

- On Mac OS X, run the following:

.. code-block:: bash

    ./scripts/setup_osx.sh

The script sets up a conda environment, which is similar to :code:`virtualenv`. To start using it, run the following:

.. code-block:: bash

    source activate extr


