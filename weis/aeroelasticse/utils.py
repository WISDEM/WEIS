from openmdao.api import Group, ExplicitComponent
from wisdem.rotorse.rotor_structure import ComputeStrains, DesignConstraints, BladeRootSizing
import numpy as np

'''
import sys
from wisdem.commonse.distributions import RayleighCDF_func
import fatpack
'''

class RotorLoadsDeflStrainsWEIS(Group):
    # OpenMDAO group to compute the blade elastic properties, deflections, and loading
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        modeling_options = self.options['modeling_options']
        opt_options      = self.options['opt_options']

        self.add_subsystem("m2pa", MtoPrincipalAxes(modeling_options=modeling_options), promotes=['alpha', 'M1', 'M2'])
        self.add_subsystem("strains", ComputeStrains(modeling_options=modeling_options), promotes=['alpha', 'M1', 'M2'])
        self.add_subsystem("constr", DesignConstraints(modeling_options=modeling_options, opt_options=opt_options))
        self.add_subsystem("brs", BladeRootSizing(rotorse_options=modeling_options["WISDEM"]["RotorSE"]))

        # Strains from frame3dd to constraint
        self.connect('strains.strainU_spar', 'constr.strainU_spar')
        self.connect('strains.strainL_spar', 'constr.strainL_spar')
         
class MtoPrincipalAxes(ExplicitComponent):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        rotorse_options = self.options["modeling_options"]["WISDEM"]["RotorSE"]
        self.n_span = n_span = rotorse_options["n_span"]
        
        self.add_input(
            "alpha",
            val=np.zeros(n_span),
            units="deg",
            desc="Angle between blade c.s. and principal axes",
        )

        self.add_input(
            "Mx",
            val=np.zeros(n_span),
            units="N*m",
            desc="distribution along blade span of edgewise bending moment",
        )
        self.add_input(
            "My",
            val=np.zeros(n_span),
            units="N*m",
            desc="distribution along blade span of flapwise bending moment",
        )

        self.add_output(
            "M1",
            val=np.zeros(n_span),
            units="N*m",
            desc="distribution along blade span of bending moment w.r.t principal axis 1",
        )
        self.add_output(
            "M2",
            val=np.zeros(n_span),
            units="N*m",
            desc="distribution along blade span of bending moment w.r.t principal axis 2",
        )

    def compute(self, inputs, outputs):
        
        alpha = inputs['alpha']

        Mx = inputs['Mx']
        My = inputs['My']

        ca = np.cos(np.deg2rad(alpha))
        sa = np.sin(np.deg2rad(alpha))

        def rotate(x, y):
            x2 = x * ca + y * sa
            y2 = -x * sa + y * ca
            return x2, y2
        
        M1, M2 = rotate(-My, -Mx)

        outputs['M1'] = M1
        outputs['M2'] = M2


def OLAFParams(omega_rpm, U0, R, a=0.3, aScale=1.2,
          deltaPsiDeg=6, nPerRot=None,
          targetFreeWakeLengthD=1,
          targetWakeLengthD=4.,
          nNWrot=8, nFWrot=0, nFWrotFree=0,
          verbose=True, dt_glue_code=None):
    """
    Computes recommended time step and wake length for OLAF based on:

    INPUTS:
     - omega_rpm: rotational speed [RPM]
     - U0: mean wind speed [m/s]
     - R: rotor radius [m]

    OPTIONS FOR TIME STEP:
      - either:
         - deltaPsiDeg : target azimuthal discretization [deg]
              or
         - nPerRot     : number of time step per rotations.
                deltaPsiDeg  -  nPerRot
                     5            72
                     6            60
                     7            51.5
                     8            45
     - dt_glue_code: glue code time step. If provided, the time step of OLAF will be approximated
                     such that it is a multiple of the glue-code time step.

    OPTIONS FOR WAKE LENGTH:
     - a: average axial induction factor at the rotor [-]
     - aScale: scaling factor to estimate induction, such that the wake convection velocity is:
               Uc=U0(1-aScale*a)
     - targetWakeLengthD: target wake length in diameter [D]
     - nNWrot     : minimum number of near wake rotations
     - nFWrot     : minimum number of far wake rotations
     - nFWrotFree : minimum number of far wake rotations (free panels)

    """
    def myprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # Rotational period
    omega = omega_rpm*2*np.pi/60
    T = 2*np.pi/omega
    # Convection velocity
    Uc = U0 * (1-aScale*a)

    # Desired time step
    if nPerRot is not None:
        dt_wanted    = np.around(T/nPerRot,5)
        deltaPsiDeg  = np.around(omega*dt_wanted*180/np.pi ,2)
    else:
        dt_wanted    = np.around(deltaPsiDeg/(6*omega_rpm),5)
        nPerRot = int(2*np.pi /(deltaPsiDeg*np.pi/180))

    # Adapting desired time step based on glue code time step
    if dt_glue_code is not None:
        dt_rounded = round(dt_wanted/dt_glue_code)*dt_glue_code
        deltaPsiDeg2 = np.around(omega*dt_rounded *180/np.pi ,2)
        myprint('>>> To satisfy glue-code dt:')
        myprint('    Rounding dt   from {} to {}'.format(dt_wanted, dt_rounded    ))
        myprint('    Changing dpsi from {} to {}'.format(deltaPsiDeg, deltaPsiDeg2))
        dt_fvw   = dt_rounded
        deltaPsiDeg = deltaPsiDeg2
        nPerRot = int(2*np.pi /(deltaPsiDeg*np.pi/180))
    else:
        dt_fvw = dt_wanted

    # Useful functions
    n2L = lambda n: (n * dt_fvw * Uc)/(2*R)  # convert number of panels to distance
    n2R = lambda n:  n * dt_fvw / T          # convert number of panels to number of rotations

    # All Wake (AW) panels - Wake length from mean wind speed
    targetWakeLength = targetWakeLengthD * 2 * R
    nAWPanels_FromU0 = int(targetWakeLength / (Uc*dt_fvw))
    # Free near wake panels (based on distance)
    targetFreeWakeLength = targetFreeWakeLengthD * 2 * R
    nNWPanelsFree_FromU0 = int(targetFreeWakeLength / (Uc*dt_fvw))
    # Far wake (FW) panels, always from number of rotations
    nFWPanels     = int(nFWrot*nPerRot)
    nFWPanelsFree = int(nFWrotFree*nPerRot)
    # Wake length from rotational speed and number of rotations
    nAWPanels_FromRot = int(nNWrot*nPerRot) # Total number of panels NW+FW

    # Below we chose between criteria on number of rotation or donwstream distance
    # This can be adapted/improved
    myprint('Number of panels (NW free) from wind speed and distance:{:15d}'.format(nNWPanelsFree_FromU0))
    myprint('Number of panels (NW+FW)   from wind speed and distance:{:15d}'.format(nAWPanels_FromU0))
    myprint('Number of panels (NW+FW)   from number of rotations    :{:15d}'.format(nAWPanels_FromRot))
    myprint('Number of panels (NW+FW)   from average between two    :{:15d}'.format(int((nAWPanels_FromRot+nAWPanels_FromU0)/2)))
    if nAWPanels_FromRot>nAWPanels_FromU0:
        # Criteria based on rotation wins:
        myprint('[INFO] Using number of rotations to setup number of panels')
        nAWPanels = nAWPanels_FromRot # Total number of panels NW+FW
    else:
        myprint('[INFO] Using wind speed and distance to setup number of panels')
        # Wake distance wins, we keep the nFW from rot but increase nNW
        nAWPanels = nAWPanels_FromU0  # Total number of panels NW+FW
    nNWPanels = nAWPanels - nFWPanels # nNW = All-Far Wake

    # See "free" near wake
    nNWPanelsFree = nNWPanelsFree_FromU0
    if nNWPanelsFree>nNWPanels:
        nNWPanelsFree=nNWPanels
        myprint('[INFO] Capping number of free NW panels to max.')
    if nNWPanelsFree<nNWPanels and nFWPanelsFree>0:
        nFWPanelsFree=0
        myprint('[INFO] Setting number of Free FW panels to zero because a frozen near wake is used')

    # Transient time (twice the time to develop the full wake extent)
    # This is the minimum recommended time before convergence of the wake is expected
    # (might be quite long)
    tMin = 2 * dt_fvw*nAWPanels
    if verbose:
        myprint('')
        myprint('{:15.2f} Transient time   ({:5.1f} rot)'.format(tMin, tMin/T))
        myprint('{:15d} nAWPanels        ({:5.1f} rot, {:5.1f}D)'.format(nAWPanels, n2R(nAWPanels), n2L(nAWPanels)))
        myprint('')
        myprint('OLAF INPUT FILE:')
        myprint('----------------------- GENERAL OPTIONS ---------------------')
        myprint('{:15.6f} DTFVW        (delta psi = {:5.1f}deg)'.format(dt_fvw, deltaPsiDeg))
        myprint('--------------- WAKE EXTENT AND DISCRETIZATION --------------')
        myprint('{:15d} nNWPanels     ({:5.1f} rot, {:5.1f}D)'.format(nNWPanels    , n2R(nNWPanels    ), n2L(nNWPanels    )))
        myprint('{:15d} nNWPanelsFree ({:5.1f} rot, {:5.1f}D)'.format(nNWPanelsFree, n2R(nNWPanelsFree), n2L(nNWPanelsFree)))
        myprint('{:15d} nFWPanels     ({:5.1f} rot, {:5.1f}D)'.format(nFWPanels    , n2R(nFWPanels    ), n2L(nFWPanels    )))
        myprint('{:15d} nFWPanelsFree ({:5.1f} rot, {:5.1f}D)'.format(nFWPanelsFree, n2R(nFWPanelsFree), n2L(nFWPanelsFree)))

    return dt_fvw, tMin, nNWPanels, nNWPanelsFree, nFWPanels, nFWPanelsFree

        
'''
def BladeFatigue(self, FAST_Output, case_list, dlc_list, inputs, outputs, discrete_inputs, discrete_outputs):

    # Perform rainflow counting
    if self.options['modeling_options']['General']['verbosity']:
        print('Running Rainflow Counting')
        sys.stdout.flush()

    rainflow = {}
    var_rainflow = ["RootMxb1", "Spn1MLxb1", "Spn2MLxb1", "Spn3MLxb1", "Spn4MLxb1", "Spn5MLxb1", "Spn6MLxb1", "Spn7MLxb1", "Spn8MLxb1", "Spn9MLxb1", "RootMyb1", "Spn1MLyb1", "Spn2MLyb1", "Spn3MLyb1", "Spn4MLyb1", "Spn5MLyb1", "Spn6MLyb1", "Spn7MLyb1", "Spn8MLyb1", "Spn9MLyb1"]
    for i, (datai, casei, dlci) in enumerate(zip(FAST_Output, case_list, dlc_list)):
        if dlci in [1.1, 1.2]:

            # Get wind speed and seed of output file
            ntm  = casei[('InflowWind', 'FileName_BTS')].split('NTM')[-1].split('_')
            U    = float(".".join(ntm[1].strip("U").split('.')[:-1]))
            Seed = float(".".join(ntm[2].strip("Seed").split('.')[:-1]))

            if U not in list(rainflow.keys()):
                rainflow[U]       = {}
            if Seed not in list(rainflow[U].keys()):
                rainflow[U][Seed] = {}

            # Rainflow counting by var
            if len(var_rainflow) == 0:
                var_rainflow = list(datai.keys())


            # index for start/end of time series
            idx_s = np.argmax(datai["Time"] >= self.T0)
            idx_e = np.argmax(datai["Time"] >= self.TMax) + 1

            for var in var_rainflow:
                ranges, means = fatpack.find_rainflow_ranges(datai[var][idx_s:idx_e], return_means=True)

                rainflow[U][Seed][var] = {}
                rainflow[U][Seed][var]['rf_amp']  = ranges.tolist()
                rainflow[U][Seed][var]['rf_mean'] = means.tolist()
                rainflow[U][Seed][var]['mean']    = float(np.mean(datai[var]))

    # save_yaml(self.FAST_resultsDirectory, 'rainflow.yaml', rainflow)
    # rainflow = load_yaml(self.FatigueFile, package=1)

    # Setup fatigue calculations
    U       = list(rainflow.keys())
    Seeds   = list(rainflow[U[0]].keys())
    chans   = list(rainflow[U[0]][Seeds[0]].keys())
    r_gage  = np.r_[0., self.R_out_ED_bl]
    r_gage /= r_gage[-1]
    simtime = self.simtime
    n_seeds = float(len(Seeds))
    n_gage  = len(r_gage)

    r       = (inputs['r']-inputs['r'][0])/(inputs['r'][-1]-inputs['r'][0])
    m_default = 8. # assume default m=10  (8 or 12 also reasonable)
    m       = [mi if mi > 0. else m_default for mi in inputs['m']]  # Assumption: if no S-N slope is given for a material, use default value TODO: input['m'] is not connected, only using the default currently

    eps_uts = inputs['Xt'][:,0]/inputs['E'][:,0]
    eps_ucs = inputs['Xc'][:,0]/inputs['E'][:,0]
    gamma_m = 1.#inputs['gamma_m']
    gamma_f = 1.#inputs['gamma_f']
    yrs     = 20.  # TODO
    t_life  = 60.*60.*24*365.24*yrs
    U_bar   = inputs['V_mean_iec']

    # pdf of wind speeds
    binwidth = np.diff(U)
    U_bins   = np.r_[[U[0] - binwidth[0]/2.], [np.mean([U[i-1], U[i]]) for i in range(1,len(U))], [U[-1] + binwidth[-1]/2.]]
    pdf = np.diff(RayleighCDF_func(U_bins, xbar=U_bar))
    if sum(pdf) < 0.9:
        print('Warning: Cummulative probability of wind speeds in rotor_loads_defl_strains.BladeFatigue is low, sum of weights: %f' % sum(pdf))
        print('Mean winds speed: %f' % U_bar)
        print('Simulated wind speeds: ', U)
        sys.stdout.flush()

    # Materials of analysis layers
    te_ss_var_ok       = False
    te_ps_var_ok       = False
    spar_cap_ss_var_ok = False
    spar_cap_ps_var_ok = False
    for i_layer in range(self.n_layers):
        if self.te_ss_var in self.layer_name:
            te_ss_var_ok        = True
        if self.te_ps_var in self.layer_name:
            te_ps_var_ok        = True
        if self.spar_cap_ss_var in self.layer_name:
            spar_cap_ss_var_ok  = True
        if self.spar_cap_ps_var in self.layer_name:
            spar_cap_ps_var_ok  = True

    # if te_ss_var_ok == False:
    #     print('The layer at the trailing edge suction side is set for Fatigue Analysis, but "%s" does not exist in the input yaml. Please check.'%self.te_ss_var)
    # if te_ps_var_ok == False:
    #     print('The layer at the trailing edge pressure side is set for Fatigue Analysis, but "%s" does not exist in the input yaml. Please check.'%self.te_ps_var)
    if spar_cap_ss_var_ok == False:
        print('The layer at the spar cap suction side is set for Fatigue Analysis, but "%s" does not exist in the input yaml. Please check.'%self.spar_cap_ss_var)
    if spar_cap_ps_var_ok == False:
        print('The layer at the spar cap pressure side is set for Fatigue Analysis, but "%s" does not exist in the input yaml. Please check.'%self.spar_cap_ps_var)
    sys.stdout.flush()

    # Get blade properties at gage locations
    y_tc       = remap2grid(r, inputs['y_tc'], r_gage)
    x_tc       = remap2grid(r, inputs['x_tc'], r_gage)
    chord      = remap2grid(r, inputs['chord'], r_gage)
    rthick     = remap2grid(r, inputs['rthick'], r_gage)
    pitch_axis = remap2grid(r, inputs['pitch_axis'], r_gage)
    EIyy       = remap2grid(r, inputs['beam:EIyy'], r_gage)
    EIxx       = remap2grid(r, inputs['beam:EIxx'], r_gage)

    te_ss_mats = np.floor(remap2grid(r, inputs['te_ss_mats'], r_gage, axis=0)) # materials is section
    te_ps_mats = np.floor(remap2grid(r, inputs['te_ps_mats'], r_gage, axis=0))
    sc_ss_mats = np.floor(remap2grid(r, inputs['sc_ss_mats'], r_gage, axis=0))
    sc_ps_mats = np.floor(remap2grid(r, inputs['sc_ps_mats'], r_gage, axis=0))

    c_TE       = chord*(1.-pitch_axis) + y_tc
    c_SC       = chord*rthick/2. + x_tc #this is overly simplistic, using maximum thickness point, should use the actual profiles
    sys.stdout.flush()

    C_miners_SC_SS_gage = np.zeros((n_gage, self.n_mat, 2))
    C_miners_SC_PS_gage = np.zeros((n_gage, self.n_mat, 2))
    C_miners_TE_SS_gage = np.zeros((n_gage, self.n_mat, 2))
    C_miners_TE_PS_gage = np.zeros((n_gage, self.n_mat, 2))

    # Map channels to output matrix
    chan_map   = {}
    for i_var, var in enumerate(chans):
        # Determine spanwise position
        if 'Root' in var:
            i_span = 0
        elif 'Spn' in var and 'M' in var:
            i_span = int(var.strip('Spn').split('M')[0])
        else:
            # not a spanwise output channel, skip
            print('Fatigue Model: Skipping channel: %s, not a spanwise moment' % var)
            sys.stdout.flush()
            chans.remove(var)
            continue
        # Determine if edgewise of flapwise moment
        if 'M' in var and 'x' in var:
            # Flapwise
            axis = 1
        elif 'M' in var and 'y' in var:
            # Edgewise
            axis = 0
        else:
            # not an edgewise / flapwise moment, skip
            print('Fatigue Model: Skipping channel: "%s", not an edgewise/flapwise moment' % var)
            sys.stdout.flush()
            continue

        chan_map[var] = {}
        chan_map[var]['i_gage'] = i_span
        chan_map[var]['axis']   = axis

    # Map composite sections
    composite_map = [['TE', 'SS', te_ss_var_ok],
                     ['TE', 'PS', te_ps_var_ok],
                     ['SC', 'SS', spar_cap_ss_var_ok],
                     ['SC', 'PS', spar_cap_ps_var_ok]]

    if self.options['modeling_options']['General']['verbosity']:
        print("Running Miner's Rule calculations")
        sys.stdout.flush()

    ########
    # Loop through composite sections, materials, output channels, and simulations (wind speeds * seeds)
    for comp_i in composite_map:

        #skip this composite section?
        if not comp_i[2]:
            continue

        #
        C_miners = np.zeros((n_gage, self.n_mat, 2))
        if comp_i[0]       == 'TE':
            c = c_TE
            if comp_i[1]   == 'SS':
                mats = te_ss_mats
            elif comp_i[1] == 'PS':
                mats = te_ps_mats
        elif comp_i[0]     == 'SC':
            c = c_SC
            if comp_i[1]   == 'SS':
                mats = sc_ss_mats
            elif comp_i[1] == 'PS':
                mats = sc_ps_mats

        for i_mat in range(self.n_mat):

            for i_var, var in enumerate(chans):
                i_gage = chan_map[var]['i_gage']
                axis   = chan_map[var]['axis']

                # skip if material at this spanwise location is not included in the composite section
                if mats[i_gage, i_mat] == 0.:
                    continue

                # Determine if edgewise of flapwise moment
                pitch_axis_i = pitch_axis[i_gage]
                chord_i      = chord[i_gage]
                c_i          = c[i_gage]
                if axis == 0:
                    EI_i     = EIxx[i_gage]
                else:
                    EI_i     = EIyy[i_gage]

                for i_u, u in enumerate(U):
                    for i_s, seed in enumerate(Seeds):
                        M_mean = np.array(rainflow[u][seed][var]['rf_mean']) * 1.e3
                        M_amp  = np.array(rainflow[u][seed][var]['rf_amp']) * 1.e3

                        for M_mean_i, M_amp_i in zip(M_mean, M_amp):
                            n_cycles = 1.
                            eps_mean = M_mean_i*c_i/EI_i
                            eps_amp  = M_amp_i*c_i/EI_i

                            if eps_amp != 0.:
                                Nf = ((eps_uts[i_mat] + np.abs(eps_ucs[i_mat]) - np.abs(2.*eps_mean*gamma_m*gamma_f - eps_uts[i_mat] + np.abs(eps_ucs[i_mat]))) / (2.*eps_amp*gamma_m*gamma_f))**m[i_mat]
                                n  = n_cycles * t_life * pdf[i_u] / (simtime * n_seeds)
                                C_miners[i_gage, i_mat, axis]  += n/Nf

        # Assign outputs
        if comp_i[0] == 'SC' and comp_i[1] == 'SS':
            outputs['C_miners_SC_SS'] = remap2grid(r_gage, C_miners, r, axis=0)
        elif comp_i[0] == 'SC' and comp_i[1] == 'PS':
            outputs['C_miners_SC_PS'] = remap2grid(r_gage, C_miners, r, axis=0)
        # elif comp_i[0] == 'TE' and comp_i[1] == 'SS':
        #     outputs['C_miners_TE_SS'] = remap2grid(r_gage, C_miners, r, axis=0)
        # elif comp_i[0] == 'TE' and comp_i[1] == 'PS':
        #     outputs['C_miners_TE_PS'] = remap2grid(r_gage, C_miners, r, axis=0)

    return outputs, discrete_outputs


class ModesElastoDyn(ExplicitComponent):
    """
    Component that adds a multiplicative factor to axial, torsional, and flap-edge coupling stiffness to mimic ElastoDyn

    Parameters
    ----------
    EA : numpy array[n_span], [N]
        1D array of the actual axial stiffness
    EIxy : numpy array[n_span], [Nm2]
        1D array of the actual flap-edge coupling stiffness
    GJ : numpy array[n_span], [Nm2]
        1D array of the actual torsional stiffness
    G  : numpy array[n_mat], [N/m2]
        1D array of the actual shear stiffness of the materials

    Returns
    -------
    EA_stiff : numpy array[n_span], [N]
        1D array of the stiff axial stiffness
    EIxy_stiff : numpy array[n_span], [Nm2]
        1D array of the stiff flap-edge coupling stiffness
    GJ_stiff : numpy array[n_span], [Nm2]
        1D array of the stiff torsional stiffness
    G_stiff  : numpy array[n_mat], [N/m2]
        1D array of the stiff shear stiffness of the materials

    """
    def initialize(self):
        self.options.declare('modeling_options')

    def setup(self):
        n_span          = self.options['modeling_options']['WISDEM']['RotorSE']['n_span']
        n_mat           = self.options['modeling_options']['materials']['n_mat']

        self.add_input('EA',    val=np.zeros(n_span), units='N',        desc='axial stiffness')
        self.add_input('EIxy',  val=np.zeros(n_span), units='N*m**2',   desc='coupled flap-edge stiffness')
        self.add_input('GJ',    val=np.zeros(n_span), units='N*m**2',   desc='torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')

        self.add_input('G',     val=np.zeros([n_mat, 3]), units='Pa',   desc='2D array of the shear moduli of the materials. Each row represents a material, the three columns represent G12, G13 and G23.')


        self.add_output('EA_stiff',  val=np.zeros(n_span), units='N',        desc='artifically stiff axial stiffness')
        self.add_output('EIxy_zero', val=np.zeros(n_span), units='N*m**2',   desc='artifically stiff coupled flap-edge stiffness')
        self.add_output('GJ_stiff',  val=np.zeros(n_span), units='N*m**2',   desc='artifically stiff torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')
        self.add_output('G_stiff',   val=np.zeros([n_mat, 3]), units='Pa',   desc='artificially stif 2D array of the shear moduli of the materials. Each row represents a material, the three columns represent G12, G13 and G23.')

    def compute(self, inputs, outputs):

        k = 10.

        outputs['EA_stiff']   = inputs['EA']   * k
        outputs['EIxy_zero']  = inputs['EIxy'] * 0.
        outputs['GJ_stiff']   = inputs['GJ']   * k
        outputs['G_stiff']    = inputs['G']    * k

'''

