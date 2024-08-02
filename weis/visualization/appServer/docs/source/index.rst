.. .. WEIS Visualization APP documentation master file, created by
..    sphinx-quickstart on Tue Jul 30 16:34:42 2024.
..    You can adapt this file completely to your liking, but it should at least
..    contain the root `toctree` directive.

.. Welcome to WEIS Visualization APP's documentation!
.. ==================================================

..    :maxdepth: 2
..    :caption: Contents:



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

WEIS Visualization APP
--------

Full-stack development for WEIS input/output visualization. This application provides a web-based graphical user interface to visualize input/output from WEIS. The app provides three types of output visualization - OpenFAST, Optimization with DLC Analysis, and WISDEM (blade, cost).

::

   appServer
      └──app/
         ├── assets/
         ├── mainApp.py              
         └── pages/
            ├── home.py
            ├── visualize_openfast.py
            ├── visualize_opt.py
            ├── visualize_wisdem_blade.py
            └── visualize_wisdem_cost.py
         
      └──share/
         ├── auto_launch_DashApp.sh
         ├── sbatch_DashApp.sh                
         └── vizFileGen.py


.. toctree::

   installation
   results