from numpy import zeros, array, linspace, float64, exp
import matplotlib.pyplot as plt 

def Newton(F, x_0, Fp, tolerance= 1e-8): 
    
    xn = x_0
    Error = tolerance + 1
    while Error > tolerance:
        xn1 = xn-F(xn)/Fp(xn)
        Error = abs(xn1-xn)
        print("xn =", xn, "(xn+1)-xn = ", xn1-xn)
        xn = xn1
       
    return F

def exponencial(x): #funciones exportadas de numpy (funciones son vectoriales) si las importamos de math la funcion será escalar
    
    return exp(x)-2*x

def exponencialp(x):
    return exp(x)-2

def particion(a, b, N): #se puede hacer con linspace
     t=zeros(N+1)
     for i in range(0,N+1):
         t[i] = a+(b-a)/N + i
     return t


#Solution = Newton(F=exponencial, x_0 = 0.5, Fp=exponencialp, tolerance=1e-8)
x = particion(a = -2, b= 2, N=100)
y = exponencial(x) #bucle implícito
plt.plot(x, y)
plt.show()