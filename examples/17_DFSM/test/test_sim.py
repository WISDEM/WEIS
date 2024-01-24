import numpy as np
from scipy.integrate import solve_ivp

def lotkavolterra(t, z, a, b, c, d,param):
    
    param['iteration']+= 1
    param['iterlist'].append(param['iteration'])
    x, y = z
    dx = [a*x - b*x*y, -c*y + d*x*y]
    return dx


if __name__ == '__main__':
    
    iter_ = 0
    param = {'iteration':0,'iterlist':[]}
    sol = solve_ivp(lotkavolterra, [0, 15], [10, 5], args=(1.5, 1, 3, 1,param),
                    dense_output=True)
    
    
    t = np.linspace(0, 15, 300)
    z = sol.sol(t)
    
    import matplotlib.pyplot as plt
    plt.plot(t, z.T)
    plt.xlabel('t')
    plt.legend(['x', 'y'], shadow=True)
    plt.title('Lotka-Volterra System')
    plt.show()
