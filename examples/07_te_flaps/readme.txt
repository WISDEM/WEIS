Example 07_te_flaps

Contents:
    - dac_driver.py - the main script used to run the simulation
    - modeling_options.yaml - contains modeling parameters for simulation (simulation/control settings)
    - BAR_USC_flaps.yaml - the geometry input file (define physical characteristics here)
    - analysis_options.yaml - sets the directory and name for analysis output
    - plotResults.py - an example script for viewing results from the simulation run

Basic Instructions:
    1. Edit the modeling_options.yaml, Level3/xfoil/path parameter to point to the path where
        you're specific xfoil executable exists
    2. Review the ROSCO parameters in modeling_options.yaml file that begin with "DAC" or "dac"
        in order to get a general understanding of how the parameters affect the way the DAC 
        controller behaves. 
    3. Save any changes to modeling_options.yaml
    4. From the command line (terminal), navigate to the WEIS/examples/07_te_flaps directory
    5. From the command line, run command 'python dac_driver.py'
    6. Once simulation has run, you should have several new sub-directories and file within the 
        07_te_flaps directory
    7. To view a plot of the results, run the command 'python plotResults.py' from the command line
