import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import fatpack
from pCrunch import Crunch,FatigueParams, AeroelasticOutput,read
from scipy.stats import weibull_min
fatigue_channels = {'TwrBsMyt': FatigueParams(slope=1),}

# plot properties
markersize = 5
linewidth = 1.5
fontsize_legend = 12
fontsize_axlabel = 18
fontsize_tick = 15
format = '.pdf'
bins = 100
slope  =1

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

if __name__ == '__main__':

    # get path to this directory
    this_dir = os.path.dirname(os.path.realpath(__file__))

    folders = ['rated_35_4var','rated_42_4var','rated_51_4var','rated_57_4var','rated_65_4var']
    cs = [35,42,51,57,65]

    fig,ax = plt.subplots(1)
    ax.set_xlabel('Amplitude [kNm]',fontsize = fontsize_axlabel)
    ax.set_ylabel('Count',fontsize = fontsize_axlabel)
    ax.tick_params(labelsize=fontsize_tick)

    fig1,ax1 = plt.subplots(1)
    ax1.set_xlabel('Amplitude [kNm]',fontsize = fontsize_axlabel)
    ax1.set_ylabel('Count',fontsize = fontsize_axlabel)
    ax1.tick_params(labelsize=fontsize_tick)

    DEL_list = []
    DEL2_list = []

    

    k = 2
    v_avg = 11.4
    from scipy.special import gamma


    for ifol,fol in enumerate(folders):

        ts_folder = this_dir + os.sep + 'outputs' +os.sep +fol

        ts_files = [ts_folder + os.sep +'DLC1.6_0_weis_job_{:02}'.format(x)+'.outb' for x in range(0,30)]
        
        
        FM = []

        d2 = []
        mean_wind = []
        chan_time_list = []
        ae_output_list = []
        F = []
        t_trans = 200

        
        for i,ts_f in enumerate(ts_files):

            channels = read(ts_f)

            outdata = {}
            t_ind = channels['Time'] > t_trans
            outdata['Time'] = channels['Time'][t_ind] - t_trans
            outdata['TwrBsMyt'] = channels['TwrBsMyt'][t_ind]
            mean_wind.append(np.mean(channels['RtVAvgxh'][t_ind]))
            
            elapsed = 800 - t_trans
            


            chan_time_list.append(channels)

            S, Mrf = fatpack.find_rainflow_ranges(outdata['TwrBsMyt'], return_means=True)
            Nrf,Frf = fatpack.find_range_count(S, bins)
            DELs = Frf ** slope * Nrf / elapsed
            DEL1 = DELs.sum() ** (1.0 / slope)

            F, Fmean = fatpack.find_rainflow_ranges(outdata['TwrBsMyt'], return_means=True)
            Nrf, Frf = fatpack.find_range_count(F, bins)
            DELs_= Frf ** slope * Nrf / elapsed
            DEL2 = DELs_.sum() ** (1.0 / slope)



            d2.append(DEL2)
            FM.append(F.reshape(-1,1))

            # get output
            
            ae_output = AeroelasticOutput(outdata, dlc = 'openfast_'+str(i),  fatigue_channels = fatigue_channels)
            

            ae_output_list.append(ae_output)
        FM = np.vstack(FM)

        ind1 = FM>00000/1
        ind2 = FM>00000
        
        cruncher = Crunch(outputs = [],lean = True)

        for output in ae_output_list:
            cruncher.add_output(output)

        cruncher.outputs = ae_output_list
        idx = np.arange(0,i+1)
        cruncher.set_probability_turbine_class(mean_wind,'I',idx = idx)
        

        prob = cruncher.prob
        d = np.sum(np.array(cruncher.dels['TwrBsMyt'])*prob)*1e-5
        #d = np.sum(np.array(cruncher.damage['TwrBsMyt'])*prob)*1e-5

        DEL_list.append(d)
        DEL2_list.append(np.sum(d2*prob)*1e-5)

        if ifol == 2:

            #ax.stairs(counts,bins,label = 'CS = '+str(cs[ifol]))
            ax.hist(FM[ind1]/1,bins = 100,label = 'CS = '+str(cs[ifol]),histtype = 'bar',alpha = 1,color = '#2ca02c')
            ax1.hist(FM/1,bins = 100,label = 'CS = '+str(cs[ifol]),histtype = 'bar',alpha = 1,linewidth = 2)
        else:

            # if ifol>2:
            if cs[ifol] == 57:
                color = '#d62728'
            elif cs[ifol] == 65:
                color = '#9467bd' 

            ax.hist(FM[ind1]/1,bins = 100,label = 'CS = '+str(cs[ifol]),histtype = 'step',linewidth = 2)
            ax1.hist(FM/1,bins = 100,label = 'CS = '+str(cs[ifol]),histtype = 'step',linewidth = 2)
    
    ax.set_xlim([100000,350000])
    ax.set_ylim([0,150])
    ax.legend(ncol=1)
    ax1.legend(ncol=1)

    fig.savefig('DELcount_HM.pdf')

    ax1.set_xlim([0000,20000])
    fig1.savefig('DELcount1.pdf')

    ax1.set_xlim([00000,100000])
    #ax1.set_ylim([0,1000])
    fig1.savefig('DELcount2.pdf')

    ax1.set_xlim([100000,350000])
    ax1.set_ylim([0,150])
    fig1.savefig('DELcount3.pdf')

    ax1.set_xlim([350000,500000])
    ax1.set_ylim([0,10])
    fig1.savefig('DELcount4.pdf')



    plt.show()
    #breakpoint()





