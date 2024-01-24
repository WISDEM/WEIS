import numpy as np
import pandas
import os
import matplotlib.pyplot as plt
from scipy import stats

if __name__ == '__main__':

    # get path to current directory
    mydir = os.path.dirname(os.path.realpath(__file__))

    # path to csv file
    csv_file = mydir + os.sep + 'weibull_dist.csv'

    # read file
    weibull_dist = pandas.read_csv(csv_file, header = None)

    # convert to numpy
    weibull_dist = weibull_dist.to_numpy()
    weibull_dist[:,0] = np.arange(0.1,3.7,0.1)
    weibull_dist[:,1] = weibull_dist[:,1]/100
    # Plot
    fig,ax = plt.subplots(1)

    ax.hist(weibull_dist[:,1],bins = weibull_dist[:,0])
    ax.set_xlabel('Current Speed [m/s]')
    ax.set_ylabel('PDF')
    #ax.set_xlim([0,3.6])
    plt.show()

    weibull_pd = pandas.DataFrame(weibull_dist)

    file_path = 'weibull_dist_cook_inlet.xlsx'




    #weibull_pd.to_excel(file_path, index = False)
    breakpoint()
    

    
    
    

