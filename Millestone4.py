from numpy import zeros, linspace, transpose, abs, float64, sqrt
# from sympy import symbols, Eq, lambdify
from scipy.optimize import fsolve, root
import matplotlib.pyplot as plt




################################################# FUNCIONES ########################################################
# # REGIONES DE ESTABILIDAD
# def Region_Estabilidad(N, x0, xf, y0, yf, Pi):
#     r = Funcion_despejar_r(Pi)
#     w_final = linspace(x0, xf, N) + 1j * linspace(y0, yf, N)[:, None]
#     r_func = lambdify(w, r, 'numpy')
#     r_final = r_func(w_final)
#     return abs(r_final) <= 1


# # Definición de las funciones Pi
# Pi_Euler(r,w) = r - 1 - w
# Pi_CN(r,w) = r - 1 + w / 2 * (1 + r)
# Pi_RK4(r,w) = - r + 1 + w + w**2/2 + w**3/6 + w**4/24
# Pi_IE(r,w) = 1 - r - w
# Pi_LF(r,w) = r ** 2 - 1 - 2 * w * r
# Pi(r,w) = [Pi_Euler, Pi_RK4, Pi_CN, Pi_EI, Pi_LF]


# FUNCIONES DE ESTABILIDAD
def Funcion_Estabilidad_Euler(w):
    return 1 + w

def Funcion_Estabilidad_CN(w):
    return (1 + w / 2) / (1 - w / 2)

def Funcion_Estabilidad_RK4(w):
    return 1 + w + w**2/2 + w**3/6 + w**4/24

def Funcion_Estabilidad_IE(w):
    return 1 / (1 - w)

def Funcion_Estabilidad_LF(w):
    return w + sqrt(w**2 + 1), w - sqrt(w**2 + 1)


# REGIONES DE ESTABILIDAD
def Region_Estabilidad_Euler(N, x0, xf, y0, yf):
    wij = linspace(x0, xf, N) + 1j * linspace(y0, yf, N)[:, None]  # Crear la cuadrícula compleja
    Region_Estabilidad_Euler = zeros((N, N), dtype=bool)  # Matriz para almacenar la estabilidad
    for i in range(N):
        for j in range(N):
            w = wij[i, j]
            r = Funcion_Estabilidad_Euler(w) 
            Region_Estabilidad_Euler[i, j] = abs(r) <= 1  # Verificar si |r| <= 1 (estabilidad)
    return Region_Estabilidad_Euler

def Region_Estabilidad_CN(N, x0, xf, y0, yf):
    wij = linspace(x0, xf, N) + 1j * linspace(y0, yf, N)[:, None]
    Region_Estabilidad_CN = zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            w = wij[i, j]
            r = Funcion_Estabilidad_CN(w) 
            Region_Estabilidad_CN[i, j] = abs(r) <= 1
    return Region_Estabilidad_CN

def Region_Estabilidad_RK4(N, x0, xf, y0, yf):
    wij = linspace(x0, xf, N) + 1j * linspace(y0, yf, N)[:, None]
    Region_Estabilidad_RK4 = zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            w = wij[i, j]
            r = Funcion_Estabilidad_RK4(w) 
            Region_Estabilidad_RK4[i, j] = abs(r) <= 1
    return Region_Estabilidad_RK4

def Region_Estabilidad_IE(N, x0, xf, y0, yf):
    wij = linspace(x0, xf, N) + 1j * linspace(y0, yf, N)[:, None]
    Region_Estabilidad_EI = zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            w = wij[i, j]
            r = Funcion_Estabilidad_IE(w) 
            Region_Estabilidad_EI[i, j] = abs(r) <= 1
    return Region_Estabilidad_EI

def Region_Estabilidad_LF(N, x0, xf, y0, yf):
    wij = linspace(x0, xf, N) + 1j * linspace(y0, yf, N)[:, None]
    Region_Estabilidad_LF = zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            w = wij[i, j]
            r1, r2 = Funcion_Estabilidad_LF(w)
            Region_Estabilidad_LF[i, j] = max(abs(r1), abs(r2)) <= 1  # Verificar si el máximo de |r| <= 1 (estabilidad)
    return Region_Estabilidad_LF



############################################################################################################
#                                               RESULTADOS                                                 #
############################################################################################################  
N = 400
x0 = -3
xf = 3
y0 = -3
yf = 3


Region_Estabilidad_Euler = Region_Estabilidad_Euler(N, x0, xf, y0, yf)
Region_Estabilidad_RK4 = Region_Estabilidad_RK4(N, x0, xf, y0, yf)
Region_Estabilidad_CN = Region_Estabilidad_CN(N, x0, xf, y0, yf)
Region_Estabilidad_IE = Region_Estabilidad_IE(N, x0, xf, y0, yf) 
Region_Estabilidad_LF = Region_Estabilidad_LF(N, x0, xf, y0, yf) 


############################################################################################################
#                                                GRÁFICAS                                                  #
############################################################################################################

# Región de estabilidad de Euler
plt.figure()
plt.imshow(Region_Estabilidad_Euler, extent=[x0, xf, y0, yf], origin='lower', cmap='Greens')
plt.title('Región de estabilidad de Euler')
plt.xlabel('Re(r)')
plt.ylabel('Im(r)')
plt.show()

# Región de estabilidad de Crank-Nicholson
plt.figure()
plt.imshow(Region_Estabilidad_CN, extent=[x0, xf, y0, yf], origin='lower', cmap='Oranges')
plt.title('Región de estabilidad de Crank-Nicolson')
plt.xlabel('Re(r)')
plt.ylabel('Im(r)')
plt.show()

# Región de estabilidad de RK4
plt.figure()
plt.imshow(Region_Estabilidad_RK4, extent=[x0, xf, y0, yf], origin='lower', cmap='Purples')
plt.title('Región de estabilidad de RK4')
plt.xlabel('Re(r)')
plt.ylabel('Im(r)')
plt.show()

# Región de estabilidad de Euler Inverso
plt.figure()
plt.imshow(Region_Estabilidad_IE, extent=[x0, xf, y0, yf], origin='lower', cmap='Blues')
plt.title('Región de estabilidad de Euler Inverso')
plt.xlabel('Re(r)')
plt.ylabel('Im(r)')
plt.show()

# Región de estabilidad de Leap-Frog
plt.figure()
plt.imshow(Region_Estabilidad_LF, extent=[x0, xf, y0, yf], origin='lower', cmap='Greys')
plt.title('Región de estabilidad de Leap-Frog')
plt.xlabel('Re(r)')
plt.ylabel('Im(r)')
plt.show()