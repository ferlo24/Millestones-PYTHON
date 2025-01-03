from numpy import zeros, array, linspace, float64
import matplotlib.pyplot as plt 
from scipy.optimize import newton
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
 
def Euler(U, dt, t, F): 

    return U + dt * F(U, t)


 
 #REFINAR MALLA:
 # Desde la partición t_1 con N+1 'puntos' obtiene la partición de t_2 que tiene con 2N+1 'puntos'.
 # Los modos pares en t_2 seguirán siendo los mismo de t_1. Los impares serán los puntos medios de los pares.
 
def refinar_malla(t_1):
    N = len(t_1)-1
    t_2 = zeros(2*N+1)
    for i in range(0,N): 
        t_2[2*i] = t_1[i] #pares
        t_2[2*i+1] =  (t_1[i+1]+t_1[i])/2 #impares
    t_2[2*N] = t_1[N]
    return t_2
 
 #Hace una partición equiespaciada en N trozos de un segmento de la recta real entre a y b
 
def particion(a, b, N): #se puede hacer con linspace
     t=zeros(N+1)
     for i in range(0,N+1):
         t[i] = a+(b-a)/N + i
     return t

# Método de Richardson:
 
def Cauchy_error(F, t, U0, Temporal_scheme):
    
    N = len(t)-1
    
    a = t[0]
    b = t[N]
    
    Error = zeros((N+1, len(U0)))
    
    t_1 = t
    t_2 = particion(a, b, 2*N)
 
    U_1 = Cauchy_problem(F, t_1, U0, Temporal_scheme) #Cauchy devuelve una matriz
    U_2 = Cauchy_problem(F, t_2, U0, Temporal_scheme)
    
    for i in range(0, N+1):
        Error[i, :] = U_2[2*i,:] - U_1[i,:] #almacena todos los instantes del problema, el segundo almaacena la característica de la variable
 
    return U_1, Error

def Oscilador(U, t):
    x = U[0]
    xdot = U[1]
    F = array([xdot, -x])
    return F

    
t_1 = particion(a = 0, b = 1, N = 1000)

Error = Cauchy_error(F = Oscilador, t = t_1, U0 = array([1, 0]), Temporal_scheme=Euler )

#a,b = 0, 1
#N = 5
#t_1 = linspace(a, b, N) #recibe número de nodos
#print(t)

#t = particion(a,b,N) #recibe número de elementos
#print(t)

#t_1 = particion(a, b, N) #vector modo vasto
#print(t_1)

#t_2 = refinar_malla(t_1) #vector refinado
#print(t_2)

#t_2 = particion(a, b, 2*N) 
#print(t_2)
#plt.plot(t_1, U_1[:, 0])
plt.plot(t_1, Error[:, 0])
plt.show()