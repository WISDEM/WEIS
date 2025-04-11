WEIS Modules
==============


ROSCO
---------

WEIS contains the Reference Open Source Controller for wind turbines (`link to paper <https://wes.copernicus.org/articles/7/53/2022/>_`).
The ROSCO dynamic library links with OpenFAST, recieving sensor signals and providing control inputs to the turbine.
The ROSCO toolbox is used to automatically tune turbine controllers using turbine information and a few controller parameters.
Both the ROSCO dynamic library and python toolbox are installed via conda with WEIS.
Within WEIS, there is an OpenMDAO wrapper connecting the turbine information from WISDEM (or an OpenFAST model) as inputs, and saves controller parameters that will be set in the ROSCO DISCON.IN input file.
The DISCON.IN file contains detailed parameters, including gains, setpoints, and limits specific to a turbine model.
Users can investigate the DISCON.IN file included in the WEIS-generated OpenFAST set when troubleshooting.
By setting the ``LoggingLevel > 0``, users can inspect the internal signals of ROSCO.


ROSCO controller
^^^^^^^^^^^^^^^^^^
The ROSCO control library uses signals like the

- generator speed
- nacelle tower top acceleration
- blade root moments
- nacelle wind vane and anemometer

to prescribe control actions for the turbine: primarily blade pitch and generator torque.

The generator torque controller is used to maintain the optimal tip speed ratio (TSR) below rated.
The optimal TSR is determined using the Cp surface from CC-Blade in WISDEM.
A wind speed estimate used to determine the optimal rotor speed. 
It is recommended to use ``VS_ControlMode: 2`` to achieve this type of torque control.
ROSCO can also set constant torque or constant power control above rated, using the ``VS_ConstPower`` input.

The pitch controller in ROSCO is a gain-schedule proportional-integral controller.
If a WISDEM turbine is used to tune the controller, the rotor speed set point is determined by the geometry inputs ``VS_maxspd`` and ``maxTS``, whichever results in a lower rotor speed.
The pitch control gains are automatically determined using the CC-blade-generated Cp surfaces.

ROSCO implements peak shaving using a look-up table from the wind speed estimate to the minimum pitch limit, which is determined using the Ct table generated in WISDEM.
ROSCO can also apply floating feedback control, where a nacelle acelleration signal is used to generate a pitch offset for damping platform motion.

Yaw control, control of structural elements (like TMDs or external forces), and mooring cable control are available in ROSCO and OpenFAST, but are not yet supported in WEIS.  
Soon, start-up and shutdown control will be availalble in both ROSCO and WEIS; they will be enabled by the DLC driver.
Shutdown control will include safety checks for high wind speed, generator overspeed, and large yaw misalignments.

pCrunch
---------

pyHAMS
-----------


DTQP
-----------