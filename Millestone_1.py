from numpy import array, zeros, linspace
import matplotlib.pyplot as plt

############################################################################################################
#                                                   DATOS                                                  #
############################################################################################################
# Condiciones iniciales
x0 = 1.0
y0 = 0.0
vx0 = 0.0
vy0 = 1.0

# Definicón de instantes inicial, final y paso del tiempo
t0 = 0.0
tf = 30.0
dt = 0.01

# Cálculo del número de intervalos o pasos temporales
N = int((tf - t0) / dt)
#print(N)

############################################################################################################
#                                 MÉTODO DE EULER EXPLÍCITO                                                #
############################################################################################################
# Inicialización del vector de instantes temporales:
t = linspace(t0, tf, N) # Vector de instantes temporales
# n = Índice de instantes temporales o pasos. Con este índice se recorre el vector de instantes temporales
#print(n)

# Vector de condiciones iniciales
U0 = array ([x0, y0, vx0, vy0])

# Inicialización de la matriZ de solución
U_Euler = zeros((N,4)) 

  

# Implementación del método de Euler
U_Euler[0] = U0
for n in range(N-1):
    # Coordenadas de posición y velocidad en cada instante de tiempo
    x = U_Euler[n, 0]
    y = U_Euler[n, 1]
    vx = U_Euler[n, 2]
    vy = U_Euler[n, 3]
    
    # Cálculo de la F a usar en métodos (compuesta por velocidades y aceleraciones)
        # Esta debe ser calculada con los valores de posición en el instate de tiempo n
    # vx = F[0]
    # vy = F[1]
    # ax = F[2]
    # ay = F[3]
    denominador = ((x**2 + y**2)**(3/2))
    ax = -x/denominador
    ay = -y/denominador

    U_Euler[n+1, 0] = x + dt * vx
    U_Euler[n+1, 1] = y + dt * vy
    U_Euler[n+1, 2] = vx + dt * ax
    U_Euler[n+1, 3] = vy + dt * ay
    
# Gráficas de la solución de Euler Explícito
plt.figure()
plt.plot(U_Euler[:, 0], U_Euler[:, 1], label="Euler Explícito") # Gráfica de la solución: POSICIONES
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solución de la EDO usando el método de Euler Explícito")
plt.grid()
plt.show()

############################################################################################################
#                                 MÉTODO DE CRANK-NICHOLSON                                                #
############################################################################################################
# Inicialización de la matriz de solución
U_C_N = zeros((N, 4))

# Implementación del método de Crank-Nicholson
U_C_N[0] = U0

for n in range(N-1):
    x = U_C_N[n, 0]
    y = U_C_N[n, 1]
    vx = U_C_N[n, 2]
    vy = U_C_N[n, 3]

    
    denominador = ((x**2 + y**2)**(3/2))
    ax = -x/denominador
    ay = -y/denominador

    # Cálculo de las nuevas posiciones y velocidades
    U_C_N[n+1, 0] = U_C_N[n, 0] + dt * vx + 0.5 * dt**2 * ax
    U_C_N[n+1, 1] = U_C_N[n, 1] + dt * vy + 0.5 * dt**2 * ay
    
    # Aceleración en el nuevo tiempo (utilizando las posiciones actualizadas)
    x_new = U_C_N[n+1, 0]
    y_new = U_C_N[n+1, 1]
    denominador_new = (x_new**2 + y_new**2)**(3/2)
    ax_new = -x_new / denominador_new
    ay_new = -y_new / denominador_new
    
    
    # Actualizar las velocidades usando la aceleración promedio
    U_C_N[n+1, 2] = U_C_N[n, 2] + 0.5 * dt * (ax + ax_new)
    U_C_N[n+1, 3] = U_C_N[n, 3] + 0.5 * dt * (ay + ay_new)
    
# Gráficas de la solución de Crank-Nicholson
plt.figure()
plt.plot(U_C_N[:, 0], U_C_N[:, 1], label="Crank-Nicholson") # Gráfica de la solución: POSICIONES
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solución de la EDO usando el método de Crank-Nicholson")
plt.grid()
plt.show()

############################################################################################################
#                                      MÉTODO RUNGE-KUTTA 4                                                #
############################################################################################################
# Inicialización de las matriz de la solución
U_RK4 = zeros((N, 4))

# Implementación del método de Runge-Kutta 4
U_RK4[0] = U0

for n in range(N-1):
    x = U_RK4[n, 0]
    y = U_RK4[n, 1]
    vx = U_RK4[n, 2]
    vy = U_RK4[n, 3]

    # Cálculo de la aceleración evitando dividir entre 0
    denominador = ((x**2 + y**2)**(3/2))
    ax = -x/denominador
    ay = -y/denominador 
   
    # Calculo de k1
    k1_x = vx
    k1_y = vy
    k1_vx = ax
    k1_vy = ay

    # Calculo de k2
    k2_x = vx + 0.5 * dt * k1_vx
    k2_y = vy + 0.5 * dt * k1_vy
    k2_vx = ax + 0.5 * dt * k1_vx
    k2_vy = ay + 0.5 * dt * k1_vy

    # Calculo de k3
    k3_x = vx + 0.5 * dt * k2_vx
    k3_y = vy + 0.5 * dt * k2_vy
    k3_vx = ax + 0.5 * dt * k2_vx
    k3_vy = ay + 0.5 * dt * k2_vy

    # Calculo de k4
    k4_x = vx + dt * k3_vx
    k4_y = vy + dt * k3_vy
    k4_vx = ax + dt * k3_vx
    k4_vy = ay + dt * k3_vy

    # Actualizar las posiciones y velocidades
    U_RK4[n+1, 0] = U_RK4[n, 0] + (dt/6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
    U_RK4[n+1, 1] = U_RK4[n, 1] + (dt/6) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
    U_RK4[n+1, 2] = U_RK4[n, 2] + (dt/6) * (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx)
    U_RK4[n+1, 3] = U_RK4[n, 3] + (dt/6) * (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy)

# Gráficas de la solución de RK4
plt.figure()
plt.plot(U_RK4[:, 0], U_RK4[:, 1], label="Runge-Kutta 4 etapas")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solución de la EDO usando el método RK4")
plt.grid()
plt.show()

############################################################################################################
#                                             GRÁFICAS                                                    #
############################################################################################################
# Gráficas de todas las soluciones juntas
plt.figure()
plt.plot(U_Euler[:, 0], U_Euler[:, 1], label="Euler Explícito")
plt.plot(U_C_N[:, 0], U_C_N[:, 1], label="Crank-Nicholson")
plt.plot(U_RK4[:, 0], U_RK4[:, 1], label="Runge-Kutta 4 etapas")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solución de la EDO usando diferentes métodos numéricos")
plt.grid()
plt.show()

