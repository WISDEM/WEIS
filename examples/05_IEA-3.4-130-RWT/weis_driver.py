
from weis.glue_code.runWEIS     import run_weis
from wisdem.commonse.mpi_tools  import MPI
import os, time, sys

## File management
run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
fname_wt_input         = run_dir + "IEA-3p4-130-RWT.yaml"
fname_modeling_options = run_dir + "modeling_options.yaml"
fname_analysis_options = run_dir + "analysis_options.yaml"

plot_flag = False

tt = time.time()
wt_opt, modeling_options, opt_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options)

rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

if rank == 0 and plot_flag:
    print('Run time: %f'%(time.time()-tt))
    sys.stdout.flush()

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm

    X = wt_opt['sse_tune.aeroperf_tables.pitch_vector']
    Y = wt_opt['sse_tune.aeroperf_tables.tsr_vector']
    X, Y = np.meshgrid(X, Y)
    pitch_schedule = wt_opt['rotorse.rp.powercurve.pitch']
    tsr_schedule = wt_opt['rotorse.rp.powercurve.Omega'] / 30. * np.pi * wt_opt['rotorse.Rtip'] / wt_opt['rotorse.rp.powercurve.V']

    # Plot the Cp surface
    fig, ax = plt.subplots()
    Z = wt_opt['sse_tune.aeroperf_tables.Cp']
    cs = ax.contourf(X, Y, Z[:,:,0], cmap=cm.inferno, levels = [0, 0.1, 0.2, 0.3, 0.4, 0.44, 0.47, 0.50, 0.53, 0.56])
    ax.plot(pitch_schedule, tsr_schedule, 'w--', label = 'Regulation trajectory')
    ax.set_xlabel('Pitch angle (deg)', fontweight = 'bold')
    ax.set_ylabel('Tip speed ratio (-)', fontweight = 'bold')
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel('Aerodynamic power coefficient (-)', fontweight = 'bold')
    plt.legend()

    # Plot the Ct surface
    fig, ax = plt.subplots()
    Z = wt_opt['sse_tune.aeroperf_tables.Ct']
    cs = ax.contourf(X, Y, Z[:,:,0], cmap=cm.inferno)
    ax.plot(pitch_schedule, tsr_schedule, 'w--', label = 'Regulation trajectory')
    ax.set_xlabel('Pitch angle (deg)', fontweight = 'bold')
    ax.set_ylabel('Tip speed ratio (-)', fontweight = 'bold')
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel('Aerodynamic thrust coefficient (-)', fontweight = 'bold')
    plt.legend()

    # Plot the Cq surface
    fig, ax = plt.subplots()
    Z = wt_opt['sse_tune.aeroperf_tables.Cq']
    cs = ax.contourf(X, Y, Z[:,:,0], cmap=cm.inferno)
    ax.plot(pitch_schedule, tsr_schedule, 'w--', label = 'Regulation trajectory')
    ax.set_xlabel('Pitch angle (deg)', fontweight = 'bold')
    ax.set_ylabel('Tip speed ratio (-)', fontweight = 'bold')
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel('Aerodynamic torque coefficient (-)', fontweight = 'bold')
    plt.legend()

    plt.show()

