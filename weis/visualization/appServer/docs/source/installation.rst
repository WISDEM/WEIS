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

.. code-block:: console

   wget https://gist.githubusercontent.com/mayankchetan/6a29d08700267e260c82ecfe686c712b/raw/aeee4b866903432c4a5a525a4819db15c5a9844d/kestrel_install.sh -O kestrel_install.sh
   bash kestrel_install.sh -p env/weis-viz-demo -raft -wisdem

Check the installation of weis, conda virtual environment, openfast, tosco, wisdem, raft (it takes around 20mins).

4. Generate visualization input yaml file

.. code-block:: console

   conda activate weis-env
   (.weis-env) $ cd weis/weis/visualization/appServer/
   (.weis-env) $ python share/vizFileGen.py --modeling_options [path_to_modeling_options] --analysis_options [path_to_analysis_options] --wt_input [path_to_final_wind_io] --output vizInput.yaml

Note that you can use the modeling and analysis options generated within the output folder of the WEIS run.

5. Run the server

.. code-block:: console

   (.weis-env) $ python app/mainApp.py --yaml [path_to_viz_input]


6. Connect the app with local machine

After finishing the set up from the hpc, open a new terminal from your local machine and run:

.. code-block:: console

   ssh -L 8050:[host_name from \#1]:8050 kl1.hpc.nrel.gov

Open a web browser, preferably Safari or Chrome, and go to hosting url (e.g., https://localhost:8050).


Set up on Local Machine
------------

1. Go to preferred directory

.. code-block:: console

   cd WEIS-Demo

2. Install WEIS and dependencies

.. code-block:: console

   wget https://gist.githubusercontent.com/mayankchetan/6a29d08700267e260c82ecfe686c712b/raw/aeee4b866903432c4a5a525a4819db15c5a9844d/kestrel_install.sh -O kestrel_install.sh
   bash kestrel_install.sh -p env/weis-viz-demo -raft -wisdem

Check the installation of weis, conda virtual environment, openfast, tosco, wisdem, raft (it takes around 20mins).

3. Generate visualization input yaml file

.. code-block:: console

   conda activate weis-env
   (.weis-env) $ cd weis/weis/visualization/appServer/
   (.weis-env) $ python share/vizFileGen.py --modeling_options [path_to_modeling_options] --analysis_options [path_to_analysis_options] --wt_input [path_to_final_wind_io] --output vizInput.yaml

Note that you can use the modeling and analysis options generated within the output folder of the WEIS run.

4. Run the server

.. code-block:: console

   (.weis-env) $ python app/mainApp.py --yaml [path_to_viz_input]

Open a web browser, preferably Safari or Chrome, and go to hosting url (e.g., https://localhost:8050).


.. 1. Make sure conda environment '**weis-env**' is ready. We are using the same virtual environment from WEIS, and all of the visualization-related functions are located under **WEIS/weis/visualization/** directory.
.. 2. For customizing user preferences and saving their local changes, we require input yaml file to run the tool. Please refer to 'app/test.yaml' file and follow the same structure. [MC - to add code/explanation how to generate requirement.yaml]

.. To use Lumache, first install it using pip:

.. .. code-block:: console

..    (.venv) $ pip install lumache


.. How to Start
.. ------------

.. .. code-block:: console

..    conda activate weis-env                                                   # Activate the new environment
..    git clone https://github.com/sora-ryu/WEIS-Visualization.git              # Clone the repository
..    cd app/
..    python mainApp.py --port [port_number] --host [host_number] --debug [debug_flag] --yaml [input_yaml_file_path]  # Run the App (e.g., python mainApp.py --port 8060 --host 0.0.0.0 --debug False --yaml test.yaml)