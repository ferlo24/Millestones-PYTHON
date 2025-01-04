from numpy import array, sqrt 

def Kepler(U,t): 

    x = U[0]
    y = U[1]
    vx = U[2]
    vy = U[3]
    denominador = ( x**2  + y**2 )**1.5

    return  array( [ vx, vy, -x/denominador, -y/denominador ] ) 

def Oscilador(U, t):
    x = U[0]
    xdot = U[1]
    F = array( [xdot, -x] )
    
    return F