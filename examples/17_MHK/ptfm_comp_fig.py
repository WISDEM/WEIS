import pickle
import glob
import os
import sys
import time

import numpy as np
import pandas as pd 
import multiprocessing as mp 

from weis.aeroelasticse import FileTools

import raft

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



def main():
    with open('/Users/dzalkind/Projects/USFLOWT/WEIS/outputs/0_RAFT/0_setup/raft_designs/raft_design_0.pkl','rb') as f:
        design = pickle.load(f)

    # set up the model
    model1 = raft.Model(design)
    model1.analyzeUnloaded(
        ballast= False, 
        heave_tol = 1.0
        )

    fix, axs = model1.plot()

    if False:
        with open('/Users/dzalkind/Projects/FloatingRM1_Controls/WEIS/outputs/48_ptfm_hull_mass_opt/raft_designs/raft_design_48.pkl','rb') as f:
            design = pickle.load(f)

        # set up the model
        model1 = raft.Model(design)
        model1.analyzeUnloaded(
            ballast= False, 
            heave_tol = 1.0
            )

        model1.plot(color='r',ax=axs)


    if True:
        
        print('Set breakpoint here and find nice camera angle')

        azm=axs.azim
        ele=axs.elev

        xlm=axs.get_xlim3d() #These are two tupples
        ylm=axs.get_ylim3d() #we use them in the next
        zlm=axs.get_zlim3d() #graph to reproduce the magnification from mousing

        print(f'axs.azim = {axs.azim}')
        print(f'axs.elev = {axs.elev}')
        print(f'axs.set_xlim3d({xlm})')
        print(f'axs.set_ylim3d({ylm})')
        print(f'axs.set_zlim3d({zlm})')

    else:
        print('Setting ')
        axs.azim = -143.27922077922082
        axs.elev = 6.62337662337643
        axs.set_xlim3d((-33.479380470031096, 33.479380470031096))
        axs.set_ylim3d((-11.01295410198392, 11.01295410198392))
        axs.set_zlim3d((-30.005888228174538, -19.994111771825473))

        print('here')

    axs.set_ylabel('(m)',labelpad=10)
    axs.set_xlabel('(m)',labelpad=10)

    axs.grid(False)
    axs.set_axis_off()

    custom_lines = [Line2D([0], [0], color='black', lw=1),Line2D([0], [0], color='red', lw=1)]

    # plt.legend(custom_lines, ['A', 'B'])

    axs.legend(custom_lines,('Original RM1','Optimized RM1 with large rotor'))


    # plt.savefig('/Users/dzalkind/Projects/IEA-22MW/FloaterIEA22/ptfm_comp.pdf', dpi=300, bbox_inches='tight',format='pdf')
    plt.savefig('/Users/dzalkind/Projects/CT-Opt/QR-Slides/ptfm_orig.png', dpi=300, bbox_inches='tight')


if __name__=='__main__':
    main()


