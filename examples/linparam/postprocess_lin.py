import numpy as np
import openmdao.api as om
import pickle


with open("tower_doe/ABCD_matrices.pkl", 'rb') as handle:
    ABCD_list = pickle.load(handle)

cr = om.CaseReader("tower_doe/log_opt.sql")

driver_cases = cr.get_cases('driver')

A_plot = []
tower_dia = []
for idx, case in enumerate(driver_cases):
    print('===================')
    dvs = case.get_design_vars(scaled=False)
    for key in dvs.keys():
        print(key)
        print(dvs[key])
    print()
    print("A matrix")
    print(ABCD_list[idx]['A'])
    print()
    
    A_plot.append(ABCD_list[idx]['A'][1, 1])
    tower_dia.append(dvs[key])
    
import matplotlib.pyplot as plt

A_plot = np.array(A_plot)
tower_dia = np.array(tower_dia)

print(tower_dia)
print(tower_dia[:, 0])
print(A_plot)

plt.scatter(tower_dia[:, 0], A_plot)

plt.xlabel('Tower base diameter, m')
plt.ylabel('A[1, 1]')
plt.tight_layout()

plt.show()