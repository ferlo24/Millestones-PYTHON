from numpy import zeros, array, log10, polyfit, linspace, pi
from numpy.linalg import norm
import matplotlib.pyplot as plt 
from Me_Functions.Cauchy_problem import  Cauchy_problem                                                
from Me_Functions.Temporal_Schemes import Euler, Crank_Nicholson, RK4, Inverse_Euler

############################################################################################################
#                                                OSCILADOR                                                 #
############################################################################################################
def Oscilador(U, t):
    x = U[0]
    xdot = U[1]
    F = array( [xdot, -x] )
    
    return F
 
############################################################################################################
#                                               REFINAR MALLA                                              #
############################################################################################################
# Refina la malla t_1 en una malla t_2 con el doble de puntos.
# Desde la partición t_1 con N+1 'puntos' obtiene la partición de t_2 que tiene 2N+1 'puntos'.
# Los modos pares en t_2 seguirán siendo los mismo de t_1. Los impares serán los puntos medios de los pares.
def refinar_malla(t_1):
    N = len(t_1)-1
    t_2 = zeros(2*N+1)
    for i in range(N+1): 
        t_2[2*i] = t_1[i] #pares
        t_2[2*i+1] =  (t_1[i+1]-t_1[i])/2 #impares
    t_2[2*N] = t_1[N]
    return t_2

############################################################################################################
#                                               PARTICIÓN                                                 #
############################################################################################################ 
#Hace una partición equiespaciada en N trozos de un segmento de la recta real entre a y b
def particion(a, b, N): #se puede hacer con linspace
     t = zeros(N+1)
     for i in range(0,N+1):
         t[i] = a+i*(b-a)/N  
     return t

############################################################################################################
#                                      Extrapolación de Richardson                                         #
############################################################################################################
# Calcula la extrapolación de Richardson de la solución del problema de Cauchy en t_1 y t_2.
def Temporal_Schemes_Error(F, t, U0, Problem, Temporal_Scheme): 
    
    N = len(t)-1
    
    a = t[0]
    b = t[N]
    
    Error = zeros((N+1, len(U0)))
    
    t_1 = t
    t_2 = particion(a, b, 2*N)
 
    U_1 = Problem(Temporal_Scheme, U0, F, t_1) #Cauchy devuelve una matriz
    U_2 = Problem(Temporal_Scheme, U0, F, t_2) #Cauchy_problem(Temporal_Scheme, U0, F, t)
    
    # Para calcular el error se hace la resta, pero un vector no se puede restar de otro si uno mide N+1 y el otro N, por eso se hace la resta en los nodos pares
    for i in range(0, N+1):
        Error[i, :] = U_2[2*i,:] - U_1[i,:] #almacena todos los instantes del problema, el segundo almaacena la característica de la variable
 
    return U_1, Error

############################################################################################################
#                                            CONVERGENCIA                                                  #
############################################################################################################
def Temporal_convergence_rate( t, F, U0, Problem, Temporal_Scheme): 
                           
    N = len(t) - 1
    m = 6
    log_E = zeros(m)
    log_N = zeros(m)

    t_1 = t 
    for i in range(0,m):    
          
        U, Error =  Temporal_Schemes_Error( F, t_1, U0, Problem, Temporal_Scheme) 
        log_E[i] = log10 ( norm( Error[N, :]  ) ) 
        log_N[i] = log10( float(N) )  
        N = 2*N
        #print(" Error =", norm(Error[N,:]), " N = ", N )

        t_1 = linspace(t[0], t[-1], N+1)
      
    y = log_E[ log_E > -12 ]
    x = log_N[ 0:len(y) ]
    order, b = polyfit(x, y, 1)

    #print("order =", order, "b =", b)

    #log_E = log_E - log10( 1 - 1./2**abs(order) ) 

    return  log_N, log_E, order

############################################################################################################
#                                                DATOS                                                     #
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

# Vector de condiciones iniciales
U0 = array([x0, y0])


############################################################################################################
#                                               RESULTADOS                                                 #
############################################################################################################  
# Separación equiespaciada de instantes de tiempo en los que calcular la solución
t_1 = particion(a = 0, b = 20*pi, N = 1000)

# Error
U_E, Error_Euler = Temporal_Schemes_Error(Oscilador, t_1, U0, Cauchy_problem, Euler)
U_CN, Error_CN = Temporal_Schemes_Error(Oscilador, t_1, U0, Cauchy_problem, Crank_Nicholson)
U_RK4, Error_RK4 = Temporal_Schemes_Error(Oscilador, t_1, U0, Cauchy_problem, RK4)
U_IE, Error_IE = Temporal_Schemes_Error(Oscilador, t_1, U0, Cauchy_problem, Inverse_Euler)

# Convergencia
log_N_E, log_E_E, Order_E = Temporal_convergence_rate(t_1, Oscilador, U0, Cauchy_problem, Euler)
log_N_CN, log_E_CN, Order_CN = Temporal_convergence_rate(t_1, Oscilador, U0, Cauchy_problem, Crank_Nicholson)
log_N_RK4, log_E_RK4, Order_RK4 = Temporal_convergence_rate(t_1, Oscilador, U0, Cauchy_problem, RK4)
log_N_IE, log_E_IE, Order_IE = Temporal_convergence_rate(t_1, Oscilador, U0, Cauchy_problem, Inverse_Euler)
print ("Order Euler =", Order_E)
print ("Order CN =", Order_CN)
print ("Order RK4 =", Order_RK4)
print ("Order EI =", Order_IE)

############################################################################################################
#                                                GRÁFICAS                                                  #
############################################################################################################
# Representación de la solución en función del tiempo para los distintos métodos
plt.plot(t_1, U_E[:, 0], label = 'Euler')
plt.plot(t_1, U_CN[:, 0], label = 'Crank-Nicholson')
plt.plot(t_1, U_RK4[:, 0], label = 'RK4')
plt.plot(t_1, U_IE[:, 0], label = 'Inverse Euler')
# Representación del error en función del tiempo para los distintos métodos
plt.plot(t_1, Error_Euler[:, 0], label = ' Error Euler')
plt.plot(t_1, Error_CN[:, 0], label = 'Error Crank-Nicholson')
plt.plot(t_1, Error_RK4[:, 0], label = 'Error RK4')
plt.plot(t_1, Error_IE[:, 0], label = 'Error Inverse Euler')
plt.legend()
plt.xlabel('t')
plt.title('Solución para los distintos métodos')
plt.show()

# Gráfica de Euler y su error
plt.plot(t_1, U_E[:, 0], label="Euler")
plt.plot(t_1, Error_Euler[:, 0],  label="Error Euler")
plt.title("Solución del esquema de Euler y su error")
plt.legend()
plt.xlabel("t")
plt.show()

# Gráfica de Crank-Nickolson y su error
plt.plot(t_1, U_CN[:, 0], label="Crank-Nickolson")
plt.plot(t_1, Error_CN[:, 0],  label="Error CN")
plt.title("Solución del esquema de Crank-Nickolson y su error")
plt.legend()
plt.xlabel("t")
plt.show()

# Gráfica de RK4 y su error
plt.plot(t_1, U_RK4[:, 0], label="RK4")
plt.plot(t_1, Error_RK4[:, 0],  label="Error RK4")
plt.title("Solución del esquema de RK4 y su error")
plt.legend()
plt.xlabel("t")
plt.show()

# Gráfica de Euler Inverso y su error
plt.plot(t_1, U_IE[:, 0], label="Euler Inverso")
plt.plot(t_1, Error_IE[:, 0],  label="Error EI")
plt.title("Solución del esquema de Euler Inverso y su error")
plt.legend()
plt.xlabel("t")
plt.show()

# Gráfica de convergencia de todos los esquemas
plt.axis('equal') # Cada unidad en el eje 𝑥 x tiene la misma escala visual que cada unidad en el eje y
plt.xlabel('logN')
plt.ylabel('logE')
plt.plot(log_N_E, log_E_E, '-b')
plt.plot(log_N_CN, log_E_CN, '-r')
plt.plot(log_N_RK4, log_E_RK4, '-g')
plt.plot(log_N_IE, log_E_IE, '-m')
plt.title("Convergencia de todos los esquemas numéricos")
plt.show()

# Gráfica de convergencia de Euler
plt.axis('equal') 
plt.xlabel('logN_E')
plt.ylabel('logE_E')
plt.plot(log_N_E, log_E_E, '-b')
plt.title("Convergencia del esquema de Euler")
plt.show()

# Gráfica de convergencia de RK4
plt.axis('equal') 
plt.xlabel('logN_RK4')
plt.ylabel('logE_RK4')  
plt.plot(log_N_RK4, log_E_RK4, '-g')
plt.title("Convergencia del esquema de RK4")
plt.show()

# Gráfica de convergencia de Cranck-Nickolson
plt.axis('equal') 
plt.xlabel('logN_CN')
plt.ylabel('logE_CN')
plt.plot(log_N_CN, log_E_CN, '-r')
plt.title("Convergencia del esquema de Crank-Nickolson")
plt.show()

# Gráfica de convergencia de Euler Inverso
plt.axis('equal') 
plt.xlabel('logN_EI')
plt.ylabel('logE_EI')
plt.plot(log_N_E, log_E_E, '-m')
plt.title("Convergencia del esquema de Euler Inverso")
plt.show()


# HECHO EN CLASE:

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
