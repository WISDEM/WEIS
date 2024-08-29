import numpy as np
import os
import pickle 
import matplotlib.pyplot as plt
import matplotlib as mpl
import openmdao.api as om

if __name__ == '__main__':

    # path to this folder
    this_dir = os.path.dirname(os.path.realpath(__file__))

    sql_dir = this_dir + os.sep + 'outputs' + os.sep +'1p6_DOE_omega_19'
    sql_file = sql_dir  +os.sep + 'log_opt.sql'


    cr = om.CaseReader(sql_file)
    driver_cases = cr.get_cases('driver')

    DVs = []

    for idx, case in enumerate(driver_cases):
        dvs = case.get_design_vars(scaled=False)
        for key in dvs.keys():
            DVs.append(dvs[key])
    
    # outputs dir
    fol = this_dir + os.sep + 'outputs'+ os.sep + '1p6_DOE_omega_19' + os.sep + 'iteration_'
    pickle_files_all = []
    for iter in range(10):
        pickle_files = [fol + str(x) +os.sep+ 'timeseries' + os.sep + 'IEA_w_TMD_'+str(iter)+'.p' for x in range(10)]
        pickle_files_all+=pickle_files


    omega = np.linspace(0.1,0.3,10)
    omega_rep = np.repeat(omega,10)
    omega_all = np.reshape(omega_rep,[10,10],order = 'F')
    omega_all = np.reshape(omega_all,[100,])



    channels = ['GenTq','BldPitch1','PtfmPitch','GenSpeed','TwrBsFxt','TwrBsMyt','NcIMURAys','GenPwr','TTDspFA']

    norm = mpl.colors.Normalize(vmin=0, vmax=7)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)

    # plot properties
    markersize = 10
    linewidth = 1.5
    fontsize_legend = 16
    fontsize_axlabel = 12
    fontsize_tick = 12
    tspan_ = [0,700]

    fig1,ax1 = plt.subplots(1)
    ax1.set_ylabel('RtVAvgxh',fontsize = fontsize_axlabel)
    ax1.set_xlim(tspan_)
    ax1.tick_params(labelsize=fontsize_tick)
    ax1.set_xlabel('Time [s]',fontsize = fontsize_axlabel)

    fig2,ax2 = plt.subplots(1)

    ax2.set_ylabel('GenTq',fontsize = fontsize_axlabel)
    ax2.set_xlim(tspan_)
    ax2.tick_params(labelsize=fontsize_tick)
    ax2.set_xlabel('Time [s]',fontsize = fontsize_axlabel)

    fig3,ax3 = plt.subplots(1)

    ax3.set_ylabel('BldPitch1',fontsize = fontsize_axlabel)
    ax3.set_xlim(tspan_)
    ax3.tick_params(labelsize=fontsize_tick)
    ax3.set_xlabel('Time [s]',fontsize = fontsize_axlabel)

    fig4,ax4 = plt.subplots(1)

    ax4.set_ylabel('Wave1Elev',fontsize = fontsize_axlabel)
    ax4.set_xlim(tspan_)
    ax4.tick_params(labelsize=fontsize_tick)
    ax4.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
    test_dataset = []

    for i_file,pkl_file in enumerate(pickle_files_all):
        print(i_file)

        test_data = {}

        with open(pkl_file,'rb') as handle:
            outputs = pickle.load(handle)

        time = outputs['Time']
        rtvavgxh = outputs['RtVAvgxh']
        gentq = outputs['GenTq']
        bldpitch1 = outputs['BldPitch1']
        waveelev = outputs['Wave1Elev']

        test_data['Time'] = time 
        test_data['Wind'] = rtvavgxh 
        test_data['Wave'] = waveelev
        test_data['omega'] = omega_all[i_file]

        for chan in channels:
            test_data[chan] = outputs[chan]


        im1 = ax1.plot(time,rtvavgxh,color = cmap.to_rgba(i_file))
        im2 = ax2.plot(time,gentq,color = cmap.to_rgba(i_file))
        im3 = ax3.plot(time,bldpitch1,color = cmap.to_rgba(i_file))
        ax4.plot(time,waveelev,color = cmap.to_rgba(i_file))

        test_dataset.append(test_data)


    

    pkl_name = 'fowt_omega_19.pkl'
    with open(pkl_name,'wb') as handle:
        pickle.dump(test_dataset,handle)

    plt.show()
    #breakpoint()



