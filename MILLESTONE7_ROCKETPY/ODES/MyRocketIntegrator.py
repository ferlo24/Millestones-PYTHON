from ODES.Cauchy_problem import Cauchy_problem, Cauchy_problem_error
from ODES.Temporal_schemes import Euler, Inverse_Euler, Crank_Nicolson, Embedded_RK, RK4
from numpy import zeros, array





def MyIntegrator( Vuelo , t , Uo , Temporal_scheme):


    # Wrapper: Calcula la derivada del estado
    def F_rocket(U, t):

        U = array(U)  # Asegurarse de que U es un array
        F = array(Vuelo.u_dot_generalized(t, U))  # Derivadas

        return F

    N = len(t) - 1 

    U = array(zeros([N + 1, 13]))
    U[0, :] = Uo  # Condición inicial

    U0 = U[0, :]

    print('simulating...')

    U = Cauchy_problem( F_rocket, t, U0, Temporal_scheme)

    print('simulation done. :)')

    return U


def MyIntegratorError( Vuelo , t , Uo , Temporal_scheme):


    # Wrapper: Calcula la derivada del estado
    def F_rocket(U, t):

        U = array(U)  # Asegurarse de que U es un array
        F = array(Vuelo.u_dot_generalized(t, U))  # Derivadas

        return F

    N = len(t) - 1 

    U = array(zeros([N + 1, 13]))
    U[0, :] = Uo  # Condición inicial

    U0 = U[0, :]

    print('Evaluating the error...')


    (U, Error) = Cauchy_problem_error( F_rocket, t, U0, Temporal_scheme)

    return  Error