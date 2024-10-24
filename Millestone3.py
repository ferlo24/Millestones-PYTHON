from numpy import zeros, array, linspace, float64
import matplotlib.pyplot as plt 
from scipy.optimize import newton

#CAUCHY
def Cauchy_problem( F, t, U0, Temporal_scheme, q=None, Tolerance=None ): 
     """
  

    Inputs:  
           F(U,t) : Function dU/dt = F(U,t) 
           t : time partition t (vector of length N+1)
           U0 : initial condition at t=0
           Temporal_schem 
           q : order o scheme (optional) 
           Tolerance: Error tolerance (optional)

    Return: 
           U: matrix[N+1, Nv], Nv state values at N+1 time steps     
     """

     N =  len(t)-1
     Nv = len(U0)
     #U = zeros( (N+1, Nv), dtype=type(U0) )
     U = zeros( (N+1, Nv), dtype=float64 ) 

     U[0,:] = U0

     for n in range(N):

        if q != None: 
          U[n+1,:] = Temporal_scheme( U[n, :], t[n+1] - t[n], t[n],  F, q, Tolerance ) 

        else: 
           U[n+1,:] = Temporal_scheme( U[n, :], t[n+1] - t[n], t[n],  F ) 

     return U
 
 #EULER
def Euler(U, dt, t, F):
    return U + dt * F(U, t)

 #EULER INVERSO
def Euler_inverso(U,dt,t,F):
    def G(X):
        return X - U - dt * F(X, t)

    return newton(G,U)

#KEPLER
def Kepler(U, t): 

    x = U[0]; y = U[1]; dxdt = U[2]; dydt = U[3]
    d = ( x**2  +y**2 )**1.5

    return  array( [ dxdt, dydt, -x/d, -y/d ] ) 



tf = 7 #si el newton no converge reducir el tiempo y el N (delta t)
N = 2000
t = linspace(0, tf, N)
U=Cauchy_problem(Kepler, t, array([1.,0.,0.,1.]), Euler)
plt.plot(U[:,0], U[:,1])
#plt.plot(t,U[:,0])
plt.show()


U=Cauchy_problem(Kepler, t, array([1.,0.,0.,1.]), Euler_inverso)
plt.plot(U[:,0], U[:,1])
#plt.plot(t,U[:,0])
plt.show()