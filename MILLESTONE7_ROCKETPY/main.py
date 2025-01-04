
from ODES.Temporal_schemes import Euler, Inverse_Euler, Crank_Nicolson, Embedded_RK, RK4 
from ODES.MyRocketIntegrator import MyIntegrator, MyIntegratorError
from rocketpy import Environment, Flight
from cohetes import get_calisto
from numpy import array, arange
import matplotlib.pyplot as plt
import datetime


#################################################################
# Configuración del entorno 
env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)

tomorrow = datetime.date.today() + datetime.timedelta(days=1)

env.set_date(
    (tomorrow.year, tomorrow.month, tomorrow.day, 12)
)  # Hour given in UTC time

env.set_atmospheric_model(type="Windy", file="GFS")



###################################################################

# Configuración del cohete

cohete = get_calisto()


Vuelo = Flight(cohete, env, rail_length = 2, inclination = 80, heading = 0, verbose=True,)


#################################################################

# Mi simulación  propia




to = 0 # tiempo inicial desede el que quieres simular
tf = 25 # tiempo final de la simulación
dt = 0.1 # salto temporal


Uo = array([0, 0, 1400, 0, 0, 0, 0.923, -0.040, 0.017, 0.3820, 0, 0, 0])  # Condición inicial



t = arange(to, tf+dt, dt) # vector de tiempos
N = len(t) - 1

temp = Crank_Nicolson # Esquema temporal con el que se va a resolver


U = MyIntegrator( Vuelo , t , Uo , temp )
Error = MyIntegratorError( Vuelo , t , Uo , temp )



# Visualización de resultados de ambas simulaciones
plt.figure()

# Simulación propia
plt.plot(t, U[:, 2], label=f"Mi simulación ({temp.__name__})")  # Agregar etiqueta con el nombre del esquema temporal

# Simulación de rocketpy
plt.plot(Vuelo.time, Vuelo.z[:,1], label="Simulación rocketpy")



plt.title("Comparación de Simulaciones de Altura")
plt.xlabel("Tiempo (s)")
plt.ylabel("Altura (m)")
plt.grid(True)
plt.legend()  # Agregar leyenda
plt.xlim(0, 25)
plt.show()



plt.figure()
plt.plot(t, Error[:, 2])  # La altura está en la columna 2 (z)
plt.title(f"Error de mi simulación por ({temp.__name__})")
plt.xlabel("Tiempo (s)")
plt.ylabel("Error altura (m)")
plt.grid(True)
plt.show()



#############################################################################

graph = "off"

if graph == "on":
    # 4- Get a summary of the results

    Vuelo.info()

    # 5- See the available results

    Vuelo.all_info()

    # 6- See the trajectory on Google Earth (RocketPy can export a KML file)

    Vuelo.export_kml(file_name="Vuelo.kml")

    # Export results in an external file



    # Export results in a txt file

    output_file_txt = "datos_simulacion.txt"

    with open(output_file_txt, "w") as file:
        file.write("Resumen de la Simulación:\n")
        # file.write(f"{Vuelo.all_info} \n")
        file.write(f"Apogeo: {Vuelo.apogee} m\n")
        # file.write(f"Tiempo de apogeo: {Vuelo.apogee_time} s\n")
        # file.write(f"Aceleración máxima: {Vuelo.max_acceleration} m/s²\n")

    print(f"Results saved in {output_file_txt}")

    # Export results in a csv file

    output_file_csv = "datos_simulacion.csv"

    with open(output_file_csv, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Parametro", "Valor"])
        writer.writerow(["Apogeo (m)", Vuelo.apogee])


    print(f"Results saved in {output_file_csv}")






