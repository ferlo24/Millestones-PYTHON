from numpy import array, zeros, linspace, concatenate, float64
import matplotlib.pyplot as plt
from scipy.optimize import newton # Import Newton Method



#################################################### DATOS ########################################################
# Condiciones iniciales
x0 = 1.
y0 = 0.
vx0 = 0.
vy0 = 1.

# Variables de tiempo y número de intervalos 
tf = 7
N = 2000


################################################# FUNCIONES #######################################################
# KEPLER
def Kepler(U, t): 

    x = U[0]; y = U[1]; dxdt = U[2]; dydt = U[3]
    d = ( x**2  +y**2 )**1.5

    return  array( [ dxdt, dydt, -x/d, -y/d ] ) 


# Problema CAUCHY: consiste en obtener la solución de un problema de CI dada una CI y un esquema temporal
def Cauchy_problem(F, t, U0, Esquema):  
    N =  len(t)-1
    Nv = len(U0)
    #U = zeros( (N+1, Nv), dtype=type(U0) )
    U = zeros( (N+1, Nv), dtype=float64 ) 
    U[0,:] = U0
    
    for n in range(N): 
        U[n+1,:] = Esquema( U[n, :], t[n+1] - t[n], t[n],  F ) 
    return U


# Esquema EULER EXPLÍCITO
def Euler (U, dt, t, F):
    
    return U + dt * F(U, t)


# Esquema RUNGE-KUTTA órden 4
def RK4 (U, dt, t, F):
    k1 = F (U, t)
    k2 = F ( U + k1 * dt/2, t + dt/2)
    k3 = F ( U + k2 * dt/2, t + dt/2)
    k4 = F ( U + k3 * dt , t + dt/2)
    return U + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


# Esquema implícito CRANK-NICKOLSON
def Crank_Nickolson(U, dt, t, F): 
    def G(X): # Siendo X = U(n-1)
        return X - U - dt/2 * (F(X, t) + F(U, t))
    return newton(G, U)


# Esquema implícito EULER IMPLÍCITO
def Euler_Inverso(U, dt, t, F): 
    def G(X): # Siendo X = U(n-1)
        return X - U - dt * F(X, t)
    return newton(G, U) # U no se modifica al "pasar" por la función, la U que sale es la que devuelve Newton



################################################### CÓDIGO ########################################################
# Separación equiespaciada de instantes de tiempo en los que calcular la solución
t = linspace(0, tf, N)

# Vector de condiciones iniciales
U0 = array ([x0, y0, vx0, vy0])

# Resolución del problema de Cauchy con el método de EULER
U_Euler = Cauchy_problem (Kepler, t, U0, Euler)
# plt.plot(U_Euler[:,0], U_Euler[:,1])
# plt.show()

# Resolución del problema de Cauchy con el método de RK4
U_RK4 = Cauchy_problem (Kepler, t, U0, RK4)
# plt.plot(U_RK4[:,0], U_RK4[:,1])
# plt.show()

# Resolución del problema de Cauchy con el método de CN
U_CN = Cauchy_problem (Kepler, t, U0, Crank_Nickolson)
# plt.plot(U_CN[:,0], U_CN[:,1])
# plt.show()

# Resolución del problema de Cauchy con el método de EULER INVERSO
U_EulerInverso = Cauchy_problem (Kepler, t, U0, Euler_Inverso)
# plt.plot(U_EulerInverso[:,0], U_EulerInverso[:,1])
# plt.show()


################################################# GRÁFICAS #########################################################
# Gráficas de las soluciones
plt.figure(figsize=(10, 6))
plt.plot(U_Euler[:, 0], U_Euler[:, 1], label="Euler Explícito", alpha=0.6)
plt.plot(U_RK4[:, 0], U_RK4[:, 1], label="Runge-Kutta 4 etapas", alpha=0.6)
plt.plot(U_CN[:, 0], U_CN[:, 1], label="Crank-Nickolson", alpha=0.6)
plt.plot(U_EulerInverso[:,0], U_EulerInverso[:,1], label="Euler Inverso", alpha=0.6)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solución de la EDO usando diferentes métodos numéricos")
plt.grid()
plt.show()