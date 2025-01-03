from numpy import array, zeros, linspace
import matplotlib.pyplot as plt
from scipy.optimize import newton

############################################################################################################
#                                       FUNCIONES ESQUEMAS TEMPORALES                                      #
############################################################################################################
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

############################################################################################################
#                                              PROBLEMA CAUCHY                                             #
############################################################################################################
def Cauchy_problem(Temporal_Scheme, U0, F, t):
    # Inputs:  
        #    F(U,t) : Function dU/dt = F(U,t) 
        #    t : time partition t (vector of length N+1)
        #    U0 : initial condition at t=0
        #    Temporal_scheme: Temporal scheme to be used (Euler, Crank_Nicholson, RK4)
        #Return: 
        #   U: matrix[N+1, Nv], Nv state values at N+1 time steps
    N = len(t) - 1
    Nv = len(U0)
    #U = zeros( (N+1, Nv), dtype=type(U0) )
    U = zeros( (N+1, Nv) ) 
    
    U[0,:] = U0
    for n in range(N): 
        U[n+1,:] = Temporal_Scheme( U[n, :], t[n+1] - t[n], t[n],  F ) 
        
    return U
    
############################################################################################################
#                                                   KEPLER                                                 #
############################################################################################################
def Kepler(U,t): 

    x = U[0]
    y = U[1]
    vx = U[2]
    vy = U[3]
    denominador = ( x**2  + y**2 )**1.5

    return  array( [ vx, vy, -x/denominador, -y/denominador ] ) 

############################################################################################################
#                                                   DATOS                                                  #
############################################################################################################
# Condiciones iniciales
x0 = 1
y0 = 0
vx0 = 0
vy0 = 1

# Definicón de instantes inicial, final y paso del tiempo
t0 = 0
tf = 7
#N = int((tf - t0) / dt) # o se puede poner directamente el número de intervalos: 
N = 2000 #(esta opción puede ser la más adecuada ya que si el newton no converge reducir el tiempo y el N (delta t))

# Separación equiespaciada de instantes de tiempo
t = linspace(t0, tf, N)

# Vector de condiciones iniciales
U0 = array([x0, y0, vx0, vy0])

############################################################################################################
#                                                RESULTADOS                                                #
############################################################################################################

# Resolución del problema de Cauchy con el método de EULER
U_Euler = Cauchy_problem(Euler, U0, Kepler, t) #Cauchy_problem(Temporal_Scheme, U0, F, t)

# Resolución del problema de Cauchy con el método de Crank-Nicholson
U_Crank_Nicholson = Cauchy_problem(Crank_Nicholson, U0, Kepler, t) #Cauchy_problem(Temporal_Scheme, U0, F, t)

# Resolución del problema de Cauchy con el método RK4
U_RK4 = Cauchy_problem(RK4, U0, Kepler, t) #Cauchy_problem(Temporal_Scheme, U0, F, t)

# Resolución del problema de Cauchy con el método de Euler Inverso
U_Inverse_Euler = Cauchy_problem(Inverse_Euler, U0, Kepler, t) #Cauchy_problem(Temporal_Scheme, U0, F, t)

############################################################################################################
#                                                GRÁFICAS                                                  #
############################################################################################################
plt.figure()
plt.plot(U_Euler[:, 0], U_Euler[:, 1], label="Euler Explícito")
plt.plot(U_Crank_Nicholson[:, 0], U_Crank_Nicholson[:, 1], label="Crank-Nickolson")
plt.plot(U_RK4[:, 0], U_RK4[:, 1], label="Runge-Kutta 4 etapas")
plt.plot(U_Inverse_Euler[:,0], U_Inverse_Euler[:,1], label="Euler Inverso")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solución de la EDO usando diferentes métodos numéricos")
plt.grid()
plt.show()

# Resultados Crank-Nicholson
plt.figure()
plt.plot(U_Crank_Nicholson[:, 0], U_Crank_Nicholson[:, 1], label="Crank-Nickolson")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solución de la EDO usando Crank-Nickolson")
plt.grid()
plt.show()

# Comparativa Crank-Nicholson y RK4
plt.figure()
plt.plot(U_Crank_Nicholson[:, 0], U_Crank_Nicholson[:, 1], label="Crank-Nickolson")
plt.plot(U_RK4[:, 0], U_RK4[:, 1], label="Runge-Kutta 4 etapas")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparativa de Crank-Nickolson y RK4")
plt.grid()
plt.show()

# Comparativa Euler y Euler Inverso
plt.figure()
plt.plot(U_Euler[:, 0], U_Euler[:, 1], label="Euler Explícito")
plt.plot(U_Inverse_Euler[:,0], U_Inverse_Euler[:,1], label="Euler Inverso")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparativa de Euler y Euler Inverso")
plt.grid()
plt.show()


