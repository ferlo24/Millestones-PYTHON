from numpy import  zeros,  log10, polyfit, linspace
from numpy.linalg import norm 

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


def refinar_malla(t_1):
    N = len(t_1)-1
    t_2 = zeros(2*N+1)
    for i in range(N + 1): 
        t_2[2*i] = t_1[i] #pares
        t_2[2*i+1] =  (t_1[i+1]-t_1[i])/2 #impares
    t_2[2*N] = t_1[N]
    return t_2


def particion(a, b, N): #se puede hacer con linspace
     t = zeros(N+1)
     for i in range(0,N+1):
         t[i] = a+i*(b-a)/N  
     return t
 


def Temporal_Schemes_Error(F, t, U0, Cauchy_problem, Temporal_Scheme): 
    
    N = len(t)-1
    
    a = t[0]
    b = t[N]
    
    Error = zeros((N+1, len(U0)))
    
    t_1 = t
    t_2 = particion(a, b, 2*N)
 
    U_1 = Cauchy_problem(Temporal_Scheme, U0, F, t_1) #Cauchy devuelve una matriz
    U_2 = Cauchy_problem(Temporal_Scheme, U0, F, t_2) #Cauchy_problem(Temporal_Scheme, U0, F, t)
    
    # Para calcular el error se hace la resta, pero un vector no se puede restar de otro si uno mide N+1 y el otro N, por eso se hace la resta en los nodos pares
    for i in range(0, N+1):
        Error[i, :] = U_2[2*i,:] - U_1[i,:] #almacena todos los instantes del problema, el segundo almaacena la característica de la variable
 
    return U_1, Error

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