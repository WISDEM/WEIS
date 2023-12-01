# example script for running RAFT from a YAML input file

import sys
import matplotlib.pyplot as plt
import yaml
import raft

def run_example(plot_flag = False):
    # open the design YAML file and parse it into a dictionary for passing to raft
    with open('VolturnUS-S_example.yaml') as file:
        design = yaml.load(file, Loader=yaml.FullLoader)

    # Create the RAFT model (will set up all model objects based on the design dict)
    model = raft.Model(design)  

    # Evaluate the system properties and equilibrium position before loads are applied
    model.analyzeUnloaded()

    # Compute natural frequencie
    model.solveEigen()

    # Simule the different load cases
    model.analyzeCases(display=1)

    if plot_flag:
        # Plot the power spectral densities from the load cases
        model.plotResponses()

        # Visualize the system in its most recently evaluated mean offset position
        model.plot()

        plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        plot_flag = sys.argv[1].lower() in ["1", "t", "true", "y", "yes", 1, True]
    elif len(sys.argv) == 1:
        plot_flag = True
    else:
        print("Usage: python example_from_yaml.py <True/False>")
        print("  The last argument is an optional declaration to show or suppress the plots (default is True)")
        
    run_example(plot_flag = plot_flag)
        
