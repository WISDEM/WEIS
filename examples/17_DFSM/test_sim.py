import numpy as np
from scipy.integrate import solve_ivp

def lotkavolterra(t, z, a, b, c, d,param):
    
    param['iteration']+= 1
    param['iterlist'].append(param['iteration'])
    x, y = z
    dx = [a*x - b*x*y, -c*y + d*x*y]
    return dx

def vanderpol(t,x,u_fun,a):
    
    # control function
    u = u_fun(t)
    
    # extract states
    x0 = x[0];x1 = x[1]
    
    # ecaluate surrogate model
    #dx = np.array([x1+np.sin(x1)*np.cos(x0),((1-(x0)**2))*x0*x1+u])
    
    dx = np.array([-x0+x1**2,x0-x1-x0**2+u])
    

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