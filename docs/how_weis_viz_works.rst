WEIS Visualization APP
=======================

Full-stack development for WEIS input/output visualization. This application provides a web-based graphical user interface to visualize input/output from WEIS. The app provides three types of output visualization - OpenFAST, Optimization with DLC Analysis, and WISDEM (blade, cost).

::

   visualization
        └──appServer
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
        └──utils.py


Installation
------------

We offer two types of installation for users who wants to leverage HPC and the ones who have only local machine to use. For users who want to have a main set up on HPC, the whole installation, including WEIS and its dependencies, optimization, applications, will be done on HPC. From our preliminary study, the app was able to successfully run with the example optimization case which takes around 430GB.

Set up on HPC
~~~~~~~~~~~~~
1. Get an interactive node

.. code-block:: console

   salloc --time=60:00 --account=weis --partition=debug

2. Go to preferred directory

.. code-block:: console

   cd WEIS-Demo

3. Install WEIS and dependencies

We created a bash script which installs all of the related libraries with a single command. We recommend to download a file first and then run the script.

.. code-block:: console

   wget https://raw.githubusercontent.com/WISDEM/WEIS/main/share/kestrel_install.sh -O kestrel_install.sh
   bash kestrel_install.sh -p [conda_env_path] -raft -wisdem
   # For example: bash kestrel_install.sh -p env/weis-env -raft -wisdem

The whole installation process might take around 20 mins. Please check if the installation of weis, conda virtual environment, openfast, rosco, wisdem and raft are successful.

4. Generate visualization input yaml file

.. code-block:: console

   module load conda
   conda activate env/weis-env
   (.weis-env) $ cd weis/weis/visualization/appServer/share/
   (.weis-env) $ python vizFileGen.py --modeling_options [path_to_modeling_options] --analysis_options [path_to_analysis_options] --wt_input [path_to_final_wind_io] --output vizInput.yaml

Note that you can use the modeling and analysis options generated within the output folder of the WEIS run.

5. Run the server

.. code-block:: console
   
   cd ../app
   (.weis-env) $ python mainApp.py --input [path_to_viz_input] --host [host_number] --port [port_number]

Now, you are able to see the hosting url with defined port number where your app server is running.

6. Connect the app with local machine

After finishing the set up from the hpc, open a new terminal from your local machine and run:

.. code-block:: console

   ssh -L [port_number]:[host_name from \#1]:[port_number] kl1.hpc.nrel.gov
   # For example, if you have not assigned specific port number to app: ssh -L 8050:[host_name from \#1]:8050 kl1.hpc.nrel.gov

Open a web browser, preferably Safari or Chrome, and go to the hosting url that shows from step \#5.


Set up on Local Machine
~~~~~~~~~~~~~~~~~~~~~~~

1. Go to preferred directory

.. code-block:: console

   cd WEIS-Demo

2. Install WEIS and dependencies

We created a bash script which installs all of the related libraries with a single command. We recommend to download a file first and then run the script.

.. code-block:: console

   wget https://raw.githubusercontent.com/WISDEM/WEIS/main/share/kestrel_install.sh -O kestrel_install.sh
   bash kestrel_install.sh -p [conda_env_path] -raft -wisdem
   # For example: bash kestrel_install.sh -p env/weis-env -raft -wisdem

The whole installation process might take around 20 mins. Please check if the installation of weis, conda virtual environment, openfast, rosco, wisdem and raft are successful.

3. Generate visualization input yaml file

.. code-block:: console

   module load conda
   conda activate env/weis-env
   (.weis-env) $ cd weis/weis/visualization/appServer/share/
   (.weis-env) $ python vizFileGen.py --modeling_options [path_to_modeling_options] --analysis_options [path_to_analysis_options] --wt_input [path_to_final_wind_io] --output vizInput.yaml

Note that you can use the modeling and analysis options generated within the output folder of the WEIS run.

4. Run the server

.. code-block:: console
   
   cd ../app
   (.weis-env) $ python mainApp.py --input [path_to_viz_input] --host [host_number] --port [port_number]

Now, you are able to see the hosting url with defined port number where your app server is running. Open a web browser, preferably Safari or Chrome, and enter the hosting url to start.



Results
------------

All of the graphical objects has been generated via Plotly library, which makes users to easily interact, zoom in, download with it. We also offer graph channel saving functions, which help users to resume their research from their previous status. Note that graph channels from the OpenFAST page will be saved once save button has been clicked.

OpenFAST
~~~~~~~~
Read OpenFAST related variables from the input yaml file, including OpenFAST output file paths and graph x,y axis settings, and visualize the graphs based on them. Note that we allow maximum 5 files to visualize and please keep 5 rows. If you have only three files to visualize, keep file4 and file5 values as 'None' and don't delete them. We recommend the file paths to be absolute path.

.. image:: ../images/viz/openfast_yaml.png

.. image:: ../images/viz/OpenFAST.pdf


Optimization
~~~~~~~~~~~~


OpenFAST optimization
*********************

First, we need to check if the optimization type is correct. For OpenFAST Optimization, please check if status is true and type is 2 from the userOptions/optimization. Then, we read design constraints and variables from userPreferences/optimization.

.. image:: ../images/viz/of_opt_yaml.png

.. image:: ../images/viz/Optimize2_1.pdf

.. image:: ../images/viz/Optimize2_2.pdf

Optimization convergence trend data will be first shown on the left layout from the analyzed log_opt.sql file. Then, user can click specific iteration and corresponding DLC visualization will be shown on the right layout. The specific OpenFAST time-series plots can be visualized as well via clicking specific outlier data.


RAFT optimization
*****************

First, we need to check if the optimization type is correct. For RAFT Optimization, please check if status is true and type is 1 from the userOptions/optimization. Then, we read platform design variables from userPreferences/optimization/convergence/channels.

.. image:: ../images/viz/raft_opt_yaml.png

.. image:: ../images/viz/Optimize1.pdf

Once clicking specific iteration, the corresponding 3D platform design plot appears from the right layout.



WISDEM - Blade
~~~~~~~~~~~~~~
Read blade related properties and WISDEM output file path from the input yaml file, and visualize the graphs based on them.

.. image:: ../images/viz/wisdem_yaml.png

.. image:: ../images/viz/WISDEM-Blade.pdf



WISDEM - Cost
~~~~~~~~~~~~~
Cost related variables are already defined from the code. Read WISDEM output file path from the input yaml file, and visualize the cost-breakdown nested graph. Note that cost calculation is based on NREL CSM model (https://wisdem.readthedocs.io/en/master/wisdem/nrelcsm/theory.html#blades).

.. image:: ../images/viz/WISDEM-Cost.pdf
