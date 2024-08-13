Installation
=====

We offer two types of installation for users who wants to leverage HPC and the ones who have only local machine to use. For users who want to have a main set up on HPC, the whole installation, including WEIS and its dependencies, optimization, applications, will be done on HPC. From our preliminary study, the app was able to successfully run with the example optimization case which takes around 430GB.

Set up on HPC
------------
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
------------

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
