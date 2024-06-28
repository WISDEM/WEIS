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

import imageio


GEN_IMAGE = True


def main():
    image_filenames = gen_plots()

    gen_gif(image_filenames)


def gen_gif(image_filenames):

    gif_dir = os.path.realpath(os.path.join(os.path.dirname(image_filenames[0]),'..'))

    # Create a GIF from the images
    with imageio.get_writer(os.path.join(gif_dir,'ptfm_opt.gif'), mode='I', duration=200) as writer:
        for filename in image_filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)


def gen_plots():

    raft_design_dir = '/Users/dzalkind/Projects/FloatingRM1_Controls/WEIS/outputs/56_big_gen2/raft_designs'
    plot_dir = os.path.join(raft_design_dir,'..','raft_plots')
    os.makedirs(plot_dir,exist_ok=True)

    n_plots = len(os.listdir(raft_design_dir))

    image_filenames = []

    for i_plot in range(n_plots):

        if GEN_IMAGE:

            with open(os.path.join(raft_design_dir,f'raft_design_{i_plot}.pkl'),'rb') as f:
                design = pickle.load(f)

            # set up the model
            model1 = raft.Model(design)
            model1.analyzeUnloaded(
                ballast= False, 
                heave_tol = 1.0
                )

            fix, axs = model1.plot()

            axs.azim = -136.17868512810136
            axs.elev = 22.723230077315748
            axs.set_xlim3d((-25.764115966005516, 38.10054840022804))
            axs.set_ylim3d((-2.318164479980485, 18.689948798385814))
            axs.set_zlim3d((-23.775121108929632, -14.225978709672228))

            axs.set_ylabel('(m)',labelpad=10)
            axs.set_xlabel('(m)',labelpad=10)

        image_filename = os.path.join(plot_dir,f'ptfm_{i_plot}.png')
        image_filenames.append(image_filename)

        if GEN_IMAGE:
            plt.savefig(image_filename, dpi=300, bbox_inches='tight')

    
    return image_filenames
    

    # custom_lines = [Line2D([0], [0], color='black', lw=1),Line2D([0], [0], color='red', lw=1)]

    # plt.legend(custom_lines, ['A', 'B'])

    # axs.legend(custom_lines,('Original RM1','Optimized RM1'))


    # plt.savefig('/Users/dzalkind/Projects/IEA-22MW/FloaterIEA22/ptfm_comp.pdf', dpi=300, bbox_inches='tight',format='pdf')
    # plt.savefig('/Users/dzalkind/Projects/CT-Opt/QR-Slides/ptfm_orig.png', dpi=300, bbox_inches='tight')


if __name__=='__main__':
    main()


