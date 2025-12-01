Installation
============

.. note::
    This documentation is for ``tace`` version 0.0.6, which is an development release and not a stable version.
    We do not rely on any complex or heavy dependencies, so we recommend using the default latest library. 
    
    It is recommended to use ``conda create`` to create a clean environment.

You can install the package in ways as described below.

Install from Source (recommended)
---------------------------------

.. code-block:: bash

    git clone https://github.com/xvzemin/tace.git
    cd tace
    pip install .
    
    # If you want to use les, you can run the following command.
    # If your network connection is poor, you can manually download and install it from https://github.com/ChengUCB/les
    pip install les@git+https://github.com/ChengUCB/les


Install via pip (not recommended, may be have bug)
---------------

.. code-block:: bash

    pip install tace 

    # If you want to use les, you can run the following command.
    # If your network connection is poor, you can manually download and install it from https://github.com/ChengUCB/les
    pip install les@git+https://github.com/ChengUCB/les


