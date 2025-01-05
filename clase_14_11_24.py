from numpy import zeros, array, linspace, float64
import matplotlib.pyplot as plt 
from scipy.optimize import newton
a=3
def f(x):
    return x**2+a #esto no es una función, devuelve una cosa y luego devuelve otra. Si esto ocurre es que está mal hecho

print("f(2)=", f(2))

a = 4
print("f(2)=", f(2)) 

# Data structures:
# 1) sets
# 2) tuples
# 3) lists: suma dos listas es concatenacion de las mismas
# 4) dictionaries
# 5) vectors, matrices

# Paradigms: 
  # FP (math)
  # OOP (physics)
  # Event Programming 
  
# Object:
    # set of data
    # set of methods (functions)

# N body problem:
    # S = ....
    # S.lower()

V = array([1, 2, 3])
pV = V
pV[0] = 4
print(V)

print(id(pV))
print(id(V))

if id(pV)==id(V):
    print("equal id, same memory space")
else:
    print("no equal id")
    
#pV = V ---> ALIAS
#U = V ----> CLONNING

