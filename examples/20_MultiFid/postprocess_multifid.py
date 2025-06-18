import numpy as np
import matplotlib.pyplot as plt
import os,dill
import matplotlib.tri as tri

if __name__ == '__main__':

    # get path to this directory
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # 2. OpenFAST directory that has all the required files to run an OpenFAST simulations
    OF_dir = this_dir + os.sep + 'outputs/below_rated_p05' + os.sep + 'openfast_runs'


    n_pts = 10

    objs = np.zeros((n_pts,2))
    opt_pts = np.zeros((n_pts,2))

    w1 = np.linspace(1,0,n_pts)
    w2 = 1-w1

    obj1 = 'TwrBsMyt_DEL'
    obj2 = 'GenSpeed_Std'

    objs = ['TwrBsMyt_DEL','GenSpeed_Std']

    save_dir = this_dir + os.sep + 'multi_fid_results_br'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    lf_warmstart_file = OF_dir + os.sep +'lf_ws_file_oz_25.dill'
    hf_warmstart_file = OF_dir + os.sep +'hf_ws_file_oz_25.dill'


    with open(hf_warmstart_file,'rb') as handle:
        hf_results = dill.load(handle)

    with open(lf_warmstart_file,'rb') as handle:
        lf_results = dill.load(handle)

    DV = np.array(hf_results['desvars'])

    for i_ob,obj in enumerate(objs):

        outputs_hf = np.zeros((len(DV),))
        outputs_lf = np.zeros((len(DV),))


        for i in range(len(DV)):
            outputs_hf[i] = hf_results['outputs'][i][obj]
            outputs_lf[i] = lf_results['outputs'][i][obj]
            

        fig1,ax1 = plt.subplots(1)

        ax1.tricontourf(DV[:,0],DV[:,1],outputs_hf)
        ax1.plot(DV[:,0],DV[:,1],'.k')

        ax1.set_xlabel('Omega VS')
        ax1.set_ylabel('Zeta VS')
        fig1.savefig(save_dir+os.sep+obj+'_hf.pdf')

        fig,ax = plt.subplots(1)

        ax.tricontourf(DV[:,0],DV[:,1],outputs_lf)
        ax.plot(DV[:,0],DV[:,1],'.k')

        ax1.set_xlabel('Omega VS')
        ax1.set_ylabel('Zeta VS')

        fig.savefig(save_dir+os.sep+obj+'_lf.pdf')

    for i_pt in range(n_pts):
        
        hf_warmstart_file_iter = OF_dir + os.sep +'multi_fid_results_br'+os.sep+'multiobj_iter_hf_'+ str(i_pt)+'.dill'

        with open(hf_warmstart_file_iter,'rb') as handle:
            hf_results_iter = dill.load(handle)

        desvars = np.array(hf_results_iter['desvars'])
        n_iter = len(desvars)

        outputs_hf_iter = np.zeros((n_iter,2))
        w1_ = w1[i_pt]
        w2_ = w2[i_pt] 

        for i in range(n_iter):
            outputs_hf_iter[i,0] = hf_results_iter['outputs'][i][obj1]
            outputs_hf_iter[i,1] = hf_results_iter['outputs'][i][obj2]

        wt_multi_obj_iter = outputs_hf_iter[:,0]*w1_ + outputs_hf_iter[:,1]*w2_
        

        fig1,ax1 = plt.subplots(1)

        ax1.tricontourf(desvars[:,0],desvars[:,1],wt_multi_obj_iter)
        ax1.plot(desvars[:,0],desvars[:,1],'.r')
        ax1.plot(desvars[-1,0],desvars[-1,1],'*')

        ax1.set_xlabel('Omega PC')
        ax1.set_ylabel('Zeta PC')
        fig1.savefig(save_dir+os.sep+'wt_objs_'+str(i_pt)+'.pdf')


        



    breakpoint()
