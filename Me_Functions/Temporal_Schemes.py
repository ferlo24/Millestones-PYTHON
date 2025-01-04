from scipy.optimize import newton
from numpy import matmul, array, zeros, float64, dot, empty   
from numpy.linalg import norm 

def Euler (U, dt, t, F): # Explícito
    
    return U + dt * F(U, t)


def Crank_Nicholson (U, dt, t, F): 
    def G(X): 
        return X - U - dt/2 * (F(X, t) + F(U, t))
    
    return newton(G, U)

def RK4 (U, dt, t, F): 
    k1 = F(U, t)
    k2 = F(U + dt/2 * k1, t + dt/2)
    k3 = F(U + dt/2 * k2, t + dt/2)
    k4 = F(U + dt * k3, t + dt)
    return U + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def Inverse_Euler(U, dt, t, F): # Implícito	
    def G(X):
        return X - U - dt * F(X, t)

    return newton(G, U)