# 17_User_Custom_Setp
WEIS leverages the WISDEM framework for many of it features. WISDEM (and therefore WEIS) is built on top of the OpenMDAO library released by NASA. OpenMDAO allows to define any input as a design variable and any output as either a figure of merit that should be minimized or as a constraint. Please refer to the OpenMDAO tutorials to know more.

WISDEM allows users to set design variables, figure of merit, and constraints from the ``analysis_options.yaml`` file that users populate. The full list of WISDEM predefined options is specified in the `modeling_options.yaml <https://github.com/WISDEM/WISDEM/blob/develop/examples/02_reference_turbines/modeling_options.yaml>`. 

In addition, WISDEM and WEIS now offer the option to build your own optimization problem by setting any available input as a design variable and any available output as either a constraint or a figure of merit. The WISDEM example #11 shows how to build your customized ``analysis_options.yaml``.

The example is available in the WISDEM repo at https://github.com/WISDEM/WISDEM/tree/develop/examples/11_user_custom
The example is explained at https://wisdem.readthedocs.io/en/latest/examples/11_user_custom/tutorial.html

Users should not need to change anything in the inputs to run the example within WEIS. Simply activate your Conda environment and launch the python file.