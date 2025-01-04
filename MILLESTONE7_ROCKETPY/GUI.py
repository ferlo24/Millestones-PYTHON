import tkinter as tk
from tkinter import ttk, messagebox
from rocketpy import Environment, SolidMotor, LiquidMotor, Rocket, Flight,Fluid, CylindricalTank, MassFlowRateBasedTank
from cohetes import get_calisto
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from math import exp
from ODES.Temporal_schemes import Euler, Inverse_Euler, Crank_Nicolson, Embedded_RK, RK4 
from ODES.MyRocketIntegrator import MyIntegrator, MyIntegratorError
from rocketpy import Environment, Flight
from cohetes import get_calisto
from numpy import array, arange
import matplotlib.pyplot as plt

class RocketPyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RocketPy GUI")
        
        #Cálculo de dimensiones de la pantalla:
        self.root.update_idletasks()  # Actualiza el tamaño de la ventana según el contenido
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f"+{x}+{y}")
        # Crear un notebook para pestañas
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Configuración global de expansión
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Crear pestañas
        self.env_frame = ttk.Frame(self.notebook)
        self.motor_frame = ttk.Frame(self.notebook)
        self.rocket_frame = ttk.Frame(self.notebook)
        # self.flight_frame = ttk.Frame(self.notebook)
        
        # Contenedor para los gráficos
        self.plot_frame = ttk.Frame(self.notebook)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10)


        self.notebook.add(self.env_frame, text="Environment")
        self.notebook.add(self.motor_frame, text="Motor")
        self.notebook.add(self.rocket_frame, text="Rocket")
        
        self.notebook.add(self.plot_frame, text="Gráficos")
        # self.notebook.add(self.flight_frame, text="Flight")

        # Configurar cada pestaña
        self.setup_environment_tab()
        self.setup_motor_tab()
        self.setup_rocket_tab()
        # self.setup_flight_tab()

        # Botón para ejecutar simulación
        tk.Button(root, text="Ejecutar Simulación", command=self.run_simulation).pack(pady=10)

    def create_label_entry(self, frame, text, var_dict, key, row):
        label = tk.Label(frame, text=text)
        label.grid(row=row, column=0, sticky="e", padx=5, pady=5)
        
        entry = tk.Entry(frame)
        entry.grid(row=row, column=1, padx=5, pady=5)
        
        var_dict[key] = entry

    def setup_environment_tab(self):
        self.env_vars = {}
        self.create_label_entry(self.env_frame, "Latitud (°):", self.env_vars, "latitude", 1)
        self.create_label_entry(self.env_frame, "Longitud (°):", self.env_vars, "longitude", 2)
        self.create_label_entry(self.env_frame, "Elevación (m):", self.env_vars, "elevation", 3)
        self.create_label_entry(self.env_frame, "Año:", self.env_vars, "year", 4)
        self.create_label_entry(self.env_frame, "Mes:", self.env_vars, "month", 5)
        self.create_label_entry(self.env_frame, "Día:", self.env_vars, "day", 6)
        self.create_label_entry(self.env_frame, "Hora (UTC) (h):", self.env_vars, "hour", 7)
        
        separator = ttk.Separator(self.env_frame, orient="horizontal")
        separator.grid(row=8, column=0, columnspan=2, pady=10, sticky="ew")

        # Combobox para el modelo atmosférico
        label = tk.Label(self.env_frame, text="Modelo Atmosférico:")
        label.grid(row=9, column=0, sticky="e", padx=5, pady=5)

        self.atmosphere_model_combo = ttk.Combobox(
        self.env_frame,
        state="readonly",
        values=[
            "standard_atmosphere", 
            "wyoming_sounding", 
            "windy_atmosphere", 
            "Forecast", 
            "Reanalysis", 
            "Ensemble", 
            "custom_atmosphere"
        ]
    )
        self.atmosphere_model_combo.grid(row=9, column=1, padx=5, pady=5)
        self.atmosphere_model_combo.set("Forecast")  # Valor predeterminado
        # Parámetros adicionales dinámicos
        self.extra_params_frame = ttk.Frame(self.env_frame)
        self.extra_params_frame.grid(row=10, column=0, columnspan=2, padx=5, pady=10)
        
        self.atmosphere_model_combo.bind("<<ComboboxSelected>>", self.update_extra_params)
    
    def update_extra_params(self, event=None):
        # Limpia el frame de parámetros adicionales
        for widget in self.extra_params_frame.winfo_children():
            widget.destroy()
        
        selected_model = self.atmosphere_model_combo.get()
        
        if selected_model == "wyoming_sounding":
                tk.Label(self.extra_params_frame, text="URL:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
                self.env_vars["file"] = tk.Entry(self.extra_params_frame)
                self.env_vars["file"].grid(row=0, column=1, padx=5, pady=5)
        if selected_model == "Reanalysis":
                tk.Label(self.extra_params_frame, text="Archivo (.CSV):").grid(row=0, column=0, sticky="e", padx=5, pady=5)
                self.env_vars["file"] = tk.Entry(self.extra_params_frame)
                self.env_vars["file"].grid(row=0, column=1, padx=5, pady=5)
            # Agregar el desplegable para seleccionar los modelos adicionales
        if selected_model == "windy_atmosphere":
                tk.Label(self.extra_params_frame, text="Archivo:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
                self.wind_model_combo = ttk.Combobox(
                    self.extra_params_frame,
                    state="readonly",
                    values=["ECMWF", "GFS", "ICON"]
                )
                self.wind_model_combo.grid(row=1, column=1, padx=5, pady=5)
                #self.wind_model_combo.set("ECMWF")  # Valor predeterminado

        if selected_model == "Forecast":
                tk.Label(self.extra_params_frame, text="Archivo:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
                self.forecast_model_combo = ttk.Combobox(
                    self.extra_params_frame,
                    state="readonly",
                    values=["GFS", "RAP", "NAM"]
                )
                self.forecast_model_combo.grid(row=1, column=1, padx=5, pady=5)
                #self.forecast_model_combo.set("GFS")  # Valor predeterminado

        if selected_model == "Ensemble":
                tk.Label(self.extra_params_frame, text="Archivo:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
                self.ensemble_model_combo = ttk.Combobox(
                    self.extra_params_frame,
                    state="readonly",
                    values=["GEFS"]
                )
                self.ensemble_model_combo.grid(row=1, column=1, padx=5, pady=5)
                #self.ensemble_model_combo.set("GEFS")  # Valor predeterminado

        elif selected_model == "custom_atmosphere":
            # Selección de método de entrada (valor, vector o archivo CSV)
            tk.Label(self.extra_params_frame, text="Método de entrada:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
            self.input_method_combo = ttk.Combobox(
                self.extra_params_frame,
                state="readonly",
                values=["Valor", "Vector", "Archivo CSV"]
            )
            self.input_method_combo.grid(row=0, column=1, padx=5, pady=5)
            #self.input_method_combo.set("Vector")  # Valor predeterminado
            self.input_method_combo.bind("<<ComboboxSelected>>", self.update_input_fields)

    def update_input_fields(self, event=None):
        # Limpia los campos de entrada previos
        for widget in self.extra_params_frame.winfo_children():
            widget.grid_forget()

        selected_method = self.input_method_combo.get()
    
        # Dependiendo del método seleccionado, se muestran los campos adecuados
        if selected_method == "Valor":
            # Mostrar entradas para valores individuales
            tk.Label(self.extra_params_frame, text="Presión (Pa):").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            self.env_vars["pressure"] = tk.Entry(self.extra_params_frame)
            self.env_vars["pressure"].grid(row=1, column=1, padx=5, pady=5)

            tk.Label(self.extra_params_frame, text="Temperatura (K):").grid(row=2, column=0, sticky="e", padx=5, pady=5)
            self.env_vars["temperature"] = tk.Entry(self.extra_params_frame)
            self.env_vars["temperature"].grid(row=2, column=1, padx=5, pady=5)

            tk.Label(self.extra_params_frame, text="Viento-u (m/s):").grid(row=3, column=0, sticky="e", padx=5, pady=5)
            self.env_vars["wind_u"] = tk.Entry(self.extra_params_frame)
            self.env_vars["wind_u"].grid(row=3, column=1, padx=5, pady=5)

            tk.Label(self.extra_params_frame, text="Viento-v (m/s):").grid(row=4, column=0, sticky="e", padx=5, pady=5)
            self.env_vars["wind_v"] = tk.Entry(self.extra_params_frame)
            self.env_vars["wind_v"].grid(row=4, column=1, padx=5, pady=5)

        elif selected_method == "Vector":
            # Mostrar entradas para vectores (listados por comas)
            tk.Label(self.extra_params_frame, text="Presión (Pa) [valores separados por comas]:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            self.env_vars["pressure"] = tk.Entry(self.extra_params_frame)
            self.env_vars["pressure"].grid(row=1, column=1, padx=5, pady=5)

            tk.Label(self.extra_params_frame, text="Temperatura (K) [valores separados por comas]:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
            self.env_vars["temperature"] = tk.Entry(self.extra_params_frame)
            self.env_vars["temperature"].grid(row=2, column=1, padx=5, pady=5)

            tk.Label(self.extra_params_frame, text="Viento-u (m/s) [valores separados por comas]:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
            self.env_vars["wind_u"] = tk.Entry(self.extra_params_frame)
            self.env_vars["wind_u"].grid(row=3, column=1, padx=5, pady=5)

            tk.Label(self.extra_params_frame, text="Viento-v (m/s) [valores separados por comas]:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
            self.env_vars["wind_v"] = tk.Entry(self.extra_params_frame)
            self.env_vars["wind_v"].grid(row=4, column=1, padx=5, pady=5)

        elif selected_method == "Archivo CSV":
            # Mostrar campo para escribir el nombre del archivo CSV
            tk.Label(self.extra_params_frame, text="Nombre del archivo CSV:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            self.env_vars["csv_file"] = tk.Entry(self.extra_params_frame)
            self.env_vars["csv_file"].grid(row=1, column=1, padx=5, pady=5)
        
    def setup_motor_tab(self):
        self.motor_vars = {}
        # Seleccion motor:
        label = tk.Label(self.motor_frame, text="Motor:")
        label.grid(row=5, column=0, sticky="e", padx=5, pady=5)
        self.motor_type_combo = ttk.Combobox(
            self.motor_frame,
            state="readonly",
            values=["Calisto", "Custom"]
        )
        self.motor_type_combo.grid(row=9, column=1, padx=5, pady=5)
        self.motor_type_combo.set("Custom")  # Valor predeterminado
        self.extra_params_motor_frame = ttk.Frame(self.motor_frame)
        self.extra_params_motor_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=10)
        
        self.motor_type_combo.bind("<<ComboboxSelected>>", self.update_extra_motor_params)
        
    def update_extra_motor_params(self, event=None):
        # Limpia el frame de parámetros adicionales
            for widget in self.extra_params_frame.winfo_children():
                widget.destroy()
            selected_motor_model = self.motor_type_combo.get()
            
            if selected_motor_model == "Custom":
                # Seleccion tipo de motor:
                tk.Label(self.motor_frame, text="Tipo de motor:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
                self.motor_type_combo = ttk.Combobox(
                    self.motor_frame,
                    state="readonly",
                    values=["Sólido", "Líquido"]
                )
                self.motor_type_combo.grid(row=5, column=1, padx=5, pady=5)
                self.motor_type_combo.set("Sólido")  # Valor predeterminado

                # Crear un marco para parámetros específicos del tipo de motor
                self.motor_specific_frame = tk.Frame(self.motor_frame)
                self.motor_specific_frame.grid(row=6, column=0, columnspan=2, pady=10)

                # Actualizar los campos según el tipo de motor seleccionado
                self.motor_type_combo.bind("<<ComboboxSelected>>", self.update_motor_specific_fields)

    def update_motor_specific_fields(self, event=None):
       # """Actualiza los campos adicionales según el tipo de motor seleccionado."""
        # Limpiar el marco de parámetros específicos
        for widget in self.motor_specific_frame.winfo_children():
            widget.destroy()

        selected_motor_type = self.motor_type_combo.get()

        if selected_motor_type == "Sólido":
            # Campos específicos para motor sólido
            self.create_label_entry(self.motor_specific_frame, "Curva empuje:", self.motor_vars, "empuje", 1)
            self.create_label_entry(self.motor_specific_frame, "Masa seca (kg):", self.motor_vars, "dry_mass", 2)
            self.create_label_entry(self.motor_specific_frame, "Inercia (kgm²):", self.motor_vars, "inertia", 3)
            self.create_label_entry(self.motor_specific_frame, "Radio tobera (m):", self.motor_vars, "nozzle_radius", 4)
            self.create_label_entry(self.motor_specific_frame, "Número de granos:", self.motor_vars, "gain_n", 5)
            self.create_label_entry(self.motor_specific_frame, "Densidad de grano (kg/m³):", self.motor_vars, "grain_density", 6)
            self.create_label_entry(self.motor_specific_frame, "Radio externo grano (m):", self.motor_vars, "grain_outer_radius", 7)
            self.create_label_entry(self.motor_specific_frame, "Radio inicial interno grano (m):", self.motor_vars, "internal_grain_radius", 8)
            self.create_label_entry(self.motor_specific_frame, "Altura inicial grano (m):", self.motor_vars, "grain_initial_height", 9)
            self.create_label_entry(self.motor_specific_frame, "Separación de grano (m):", self.motor_vars, "grain_separation", 10)
            self.create_label_entry(self.motor_specific_frame, "Posición centro de masas granos (m):", self.motor_vars, "grains_mc_position", 11)
            self.create_label_entry(self.motor_specific_frame, "Posición centro de masas (masa seca) (m):", self.motor_vars, "mc_position", 12)
            self.create_label_entry(self.motor_specific_frame, "Posición tobera (m):", self.motor_vars, "nozzle_p", 13)
            self.create_label_entry(self.motor_specific_frame, "Tiempo quemado (s):", self.motor_vars, "t_b", 14)
            self.create_label_entry(self.motor_specific_frame, "Radio garganta (m):", self.motor_vars, "nozzle_t", 15)
        elif selected_motor_type == "Líquido":
            # Campos específicos para motor líquido
            self.create_label_entry(self.motor_specific_frame, "Densidad ox. (kg/m³):", self.motor_vars, "density_ox_l", 1)
            self.create_label_entry(self.motor_specific_frame, "Densidad red. (kg/m³):", self.motor_vars, "density_fuel_l", 2)
            self.create_label_entry(self.motor_specific_frame, "Radio tanque (m):", self.motor_vars, "tank_radius", 3)
            self.create_label_entry(self.motor_specific_frame, "Altura tanque (m):", self.motor_vars, "tank_height", 4)
            self.create_label_entry(self.motor_specific_frame, "Masa inicial oxidante líquido (kg):", self.motor_vars, "mass_i_ox_l", 5)
            self.create_label_entry(self.motor_specific_frame, "Masa inicial oxidante líquido gas (kg):", self.motor_vars, "mass_i_ox_g", 6)
            self.create_label_entry(self.motor_specific_frame, "Gasto másico entrada oxidante líq. (kg/s):", self.motor_vars, "mass_rate_in", 7)
            self.create_label_entry(self.motor_specific_frame, "Gasto másico entrada oxidante gas. (kg/s):", self.motor_vars, "mass_rate_in_g", 8)
            self.create_label_entry(self.motor_specific_frame, "Tiempo de flujo (s):", self.motor_vars, "flux_time", 9)
            self.create_label_entry(self.motor_specific_frame, "Masa inicial fuel líquido (kg):", self.motor_vars, "mass_i_f_l", 10)
            self.create_label_entry(self.motor_specific_frame, "Masa inicial fuel gas (kg):", self.motor_vars, "mass_i_f_g", 11)
            self.create_label_entry(self.motor_specific_frame, "Gasto másico entrada fuel líq. (kg/s):", self.motor_vars, "liquid_initial_mass_flow_rate_in", 12)
            self.create_label_entry(self.motor_specific_frame, "Gasto másico entrada fuel gas. (kg/s):", self.motor_vars, "gas_initial_mass_flow_rate_in", 13)
            self.create_label_entry(self.motor_specific_frame, "Masa seca (kg):", self.motor_vars, "dry_mass_l", 14)
            self.create_label_entry(self.motor_specific_frame, "Inercia (kgm²):", self.motor_vars, "inertia_l", 15)
            self.create_label_entry(self.motor_specific_frame, "Radio tobera (m):", self.motor_vars, "nozzle_radius_l", 16)
            self.create_label_entry(self.motor_specific_frame, "Posición centro de masas (masa seca) (m):", self.motor_vars, "mc_position_l", 17)
            self.create_label_entry(self.motor_specific_frame, "Posición tobera (m):", self.motor_vars, "nozzle_p_l", 18)
            self.create_label_entry(self.motor_specific_frame, "Densidad ox. gas (kg/m³):", self.motor_vars, "density_o_gas", 19)
            self.create_label_entry(self.motor_specific_frame, "Densidad red. gas (kg/m³):", self.motor_vars, "density_f_gas", 20)
            self.create_label_entry(self.motor_specific_frame, "Posición tanque oxidante (m):", self.motor_vars, "position_o_tank", 21)
            self.create_label_entry(self.motor_specific_frame, "Posición tanque reductor (m):", self.motor_vars, "position_f_tank", 22)
        
    def setup_rocket_tab(self):
        self.rocket_vars = {}
        self.create_label_entry(self.rocket_frame, "Radio (m):", self.rocket_vars, "radius", 1)
        self.create_label_entry(self.rocket_frame, "Masa (kg):", self.rocket_vars, "dry_mass", 2)
        self.create_label_entry(self.rocket_frame, "Inercia (kgm²):", self.rocket_vars, "inertia", 3)
        self.create_label_entry(self.rocket_frame, "Centro de masas sin motor (m):", self.rocket_vars, "motor_cm", 4)
        self.create_label_entry(self.rocket_frame, "Coeficiente de arrastre:", self.rocket_vars, "drag_coefficient", 5)

    # def setup_flight_tab(self):
    #     self.flight_vars = {}
    #     self.create_label_entry(self.flight_frame, "Altitud inicial (m):", self.flight_vars, "initial_altitude", 0)

    def run_simulation(self):
        try:
            # Obtener modelo atmosférico seleccionado
            atmosphere_model = self.atmosphere_model_combo.get()
            
            # Crear entorno
            env = Environment(
                latitude=float(self.env_vars["latitude"].get()),
                longitude=float(self.env_vars["longitude"].get()),
                elevation=float(self.env_vars["elevation"].get())
            )
            
            # Establecer fecha y hora
            env.set_date(
                        (int(self.env_vars["year"].get()), 
                        int(self.env_vars["month"].get()), 
                        int(self.env_vars["day"].get()), 
                        int(self.env_vars["hour"].get()))
                         ) 

            # Manejo del modelo atmosférico
            if atmosphere_model == "wyoming_sounding":
                # Para estos modelos, cargamos el archivo o configuraciones relacionadas
                file_param = self.env_vars["file"].get()  # Nombre del archivo si existe
                env.set_atmospheric_model(type=atmosphere_model, file=file_param)
            if atmosphere_model ==  "Reanalysis":
                # Para estos modelos, cargamos el archivo o configuraciones relacionadas
                file_param = self.env_vars["file"].get()  # Nombre del archivo si existe
                env.set_atmospheric_model(type=atmosphere_model, file=file_param)
            if atmosphere_model ==  "windy_atmosphere":
                # Para estos modelos, cargamos el archivo o configuraciones relacionadas
                model1 = self.wind_model_combo.get()  # Nombre del archivo si existe
                env.set_atmospheric_model(type=atmosphere_model, file=model1)
            if atmosphere_model ==  "Forecast":
                # Para estos modelos, cargamos el archivo o configuraciones relacionadas
                model2 = self.forecast_model_combo.get()  # Nombre del archivo si existe
                env.set_atmospheric_model(type=atmosphere_model, file=model2)
            if atmosphere_model ==  "Ensemble":
                # Para estos modelos, cargamos el archivo o configuraciones relacionadas
                model3 = self.ensemble_model_combo.get()  # Nombre del archivo si existe
                env.set_atmospheric_model(type=atmosphere_model, file=model3)
            elif atmosphere_model == "custom_atmosphere":
                # Recoger los parámetros dependiendo del método seleccionado
                input_method = self.input_method_combo.get()

                if input_method == "Valor":
                    # Recoger los valores individuales de los campos
                    pressure = float(self.env_vars["pressure"].get())
                    temperature = float(self.env_vars["temperature"].get())
                    wind_u = float(self.env_vars["wind_u"].get())
                    wind_v = float(self.env_vars["wind_v"].get())
                    
                    # Configurar el modelo atmosférico para "Valor"
                    env.set_atmospheric_model(type=atmosphere_model, file=None, dictionary=None, pressure=pressure, temperature=temperature, wind_u=wind_u, wind_v=wind_v)

                elif input_method == "Vector":
                    # Recoger los vectores de valores (separados por comas)
                    pressure = list(map(float, self.env_vars["pressure"].get().split(",")))
                    temperature = list(map(float, self.env_vars["temperature"].get().split(",")))
                    wind_u = list(map(float, self.env_vars["wind_u"].get().split(",")))
                    wind_v = list(map(float, self.env_vars["wind_v"].get().split(",")))
                    
                    # Configurar el modelo atmosférico para "Vector"
                    env.set_atmospheric_model(type=atmosphere_model, file=None, dictionary=None, pressure=pressure, temperature=temperature, wind_u=wind_u, wind_v=wind_v)

                elif input_method == "Archivo CSV":
                    # Recoger el nombre del archivo CSV
                    csv_file = self.env_vars["csv_file"].get()
                    
                    # Configurar el modelo atmosférico para "Archivo CSV"
                    env.set_atmospheric_model(type=atmosphere_model, file=csv_file, dictionary=None, pressure=None, temperature=None, wind_u=None, wind_v=None)
            #env.info()
            
            # Crear motor:
            if self.motor_type_combo.get() == "Calisto":
                motor = get_calisto()
                motor.info()
            elif self.motor_type_combo.get() == "Custom":
                if self.motor_type_combo.get()=="Solid":
                    motor = SolidMotor(
                        thrust_source= self.motor_vars["empuje"].get(),
                        dry_mass=float(self.motor_vars["dry_mass"].get()),
                        dry_inertia=list(map(float, self. motor_vars["inertia"].get().split(","))),
                        nozzle_radius=float(self.motor_vars["nozzle_radius"].get()),
                        grain_number=float(self.motor_vars["gain_n"].get()),
                        grain_density=float(self.motor_vars["grain_density"].get()),
                        grain_outer_radius=float(self.motor_vars["grain_outer_radius"].get()),
                        grain_initial_inner_radius=float(self.motor_vars["internal_grain_radius"].get()),
                        grain_initial_height=float(self.motor_vars["grain_initial_height"].get()),
                        grain_separation=float(self.motor_vars["grain_separation"].get()),
                        grains_center_of_mass_position=float(self.motor_vars["grains_mc_position"].get()),
                        center_of_dry_mass_position=float(self.motor_vars["mc_position"].get()),
                        nozzle_position=float(self.motor_vars["nozzle_p"].get()),
                        burn_time=float(self.motor_vars["t_b"].get()),
                        throat_radius=float(self.motor_vars["nozzle_t"].get()),
                        coordinate_system_orientation="nozzle_to_combustion_chamber",
                    )
                    motor.info()
                if self.motor_type_combo.get()=="Liquid":
                        # Define fluids
                        oxidizer_liq = Fluid(name="N2O_l", density=float(self.motor_vars["density_ox_l"].get()))
                        oxidizer_gas = Fluid(name="N2O_g", density=float(self.motor_vars["density_o_gas"].get()))   
                        fuel_liq = Fluid(name="ethanol_l", density=float(self.motor_vars["density_fuel_l"].get()))
                        fuel_gas = Fluid(name="ethanol_g", density=float(self.motor_vars["density_f_gas"].get()))

                        # Define tanks geometry
                        tanks_shape = CylindricalTank(
                            radius = float(self.motor_vars["tank_radius"].get()),    
                            height = float(self.motor_vars["tank_height"].get()),    
                            spherical_caps = True)

                        # Define tanks
                        oxidizer_tank = MassFlowRateBasedTank(
                            name="oxidizer tank",
                            geometry=tanks_shape,
                            flux_time=float(self.motor_vars["flux_time"].get()),
                            initial_liquid_mass=float(self.motor_vars["mass_i_ox_l"].get()),
                            initial_gas_mass=float(self.motor_vars["mass_i_ox_g"].get()),   
                            liquid_mass_flow_rate_in=float(self.motor_vars["mass_rate_in"].get()),
                            liquid_mass_flow_rate_out=lambda t: 32 / 3 * exp(-0.25 * t),
                            gas_mass_flow_rate_in=float(self.motor_vars["mass_rate_in_g"].get()),
                            gas_mass_flow_rate_out=0,
                            liquid=oxidizer_liq,
                            gas=oxidizer_gas,
                        )

                        fuel_tank = MassFlowRateBasedTank(
                            name="fuel tank",
                            geometry=tanks_shape,
                            flux_time=float(self.motor_vars["flux_time"].get()),
                            initial_liquid_mass=float(self.motor_vars["mass_i_f_l"].get()),
                            initial_gas_mass=float(self.motor_vars["mass_i_f_g"].get()),
                            liquid_mass_flow_rate_in=float(self.motor_vars["liquid_initial_mass_flow_rate_in"].get()),
                            liquid_mass_flow_rate_out=lambda t: 21 / 3 * exp(-0.25 * t),
                            gas_mass_flow_rate_in=float(self.motor_vars["gas_initial_mass_flow_rate_in"].get()),
                            gas_mass_flow_rate_out=lambda t: 0.01 / 3 * exp(-0.25 * t),
                            liquid=fuel_liq,
                            gas=fuel_gas,
                        )
                        motor = LiquidMotor(
                            thrust_source=lambda t: 4000 - 100 * t**2,
                            dry_mass=float(self.motor_vars["dry_mass_l"].get()),
                            dry_inertia=list(map(float, self.motor_vars["inertia_l"].get().split(","))),
                            nozzle_radius=float(self.motor_vars["nozzle_radius_l"].get()),
                            center_of_dry_mass_position=float(self.motor_vars["mc_position_l"].get()),
                            nozzle_position=float(self.motor_vars["nozzle_p_l"].get()),
                            burn_time=float(self.motor_vars["t_b"].get()),
                            coordinate_system_orientation="nozzle_to_combustion_chamber",
                        )
                        motor.add_tank(tank=oxidizer_tank, position=float(self.motor_vars["position_o_tank"].get()))
                        motor.add_tank(tank=fuel_tank, position=float(self.motor_vars["position_f_tank"].get()))
                        motor.info()
            
            
            # # Crear cohete
            if self.motor_type_combo.get() == "Calisto":
                cohete = get_calisto()
            else:
                cohete = Rocket(
                    radius=float(self.rocket_vars["radius"].get()),
                    mass=float(self.rocket_vars["dry_mass"].get()),
                    inertia=list(map(float, self.rocket_vars["inertia"].get().split(","))),
                    power_off_drag= float(self.rocket_vars["drag_coefficient"].get()),
                    power_on_drag=0,
                    center_of_mass_without_motor=float(self.rocket_vars["motor_cm"].get()),
                    coordinate_system_orientation="tail_to_nose",
                )
                
                cohete.add_motor(motor, position=-1.255)
                
                rail_buttons = cohete.set_rail_buttons(
                    upper_button_position=0.0818,
                    lower_button_position=-0.6182,
                    angular_position=45,
                )
                
                nose_cone = cohete.add_nose(
                    length=0.55829, kind="von karman", position=1.278)

                fin_set = cohete.add_trapezoidal_fins(
                    n=4,
                    root_chord=0.120,
                    tip_chord=0.060,
                    span=0.110,
                    position=-1.04956,
                    cant_angle=0.5,
                    airfoil=None,
                )

                tail = cohete.add_tail(
                    top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
                    )
                
                main = cohete.add_parachute(
                                name="main",
                                cd_s=10.0,
                                trigger=800,      # ejection altitude in meters
                                sampling_rate=105,
                                lag=1.5,
                                noise=(0, 8.3, 0.5),
                            )

                drogue = cohete.add_parachute(
                    name="drogue",
                    cd_s=1.0,
                    trigger="apogee",  # ejection at apogee
                    sampling_rate=105,
                    lag=1.5,
                    noise=(0, 8.3, 0.5),
                )
                #cohete.draw()
            # Simular vuelo
            flight = Flight(rocket = cohete, environment=env, rail_length=5.2, inclination=85, heading=0, verbose=True,)  # Crear objeto de vuelo
            #flight.plots.trajectory_3d() 
            
            # Mostrar resultados en la interfaz gráfica:
            to = 0 # tiempo inicial desede el que quieres simular
            tf = 25 # tiempo final de la simulación
            dt = 0.1 # salto temporal
            Uo = array([0, 0, 1400, 0, 0, 0, 0.923, -0.040, 0.017, 0.3820, 0, 0, 0])  # Condición inicial
            t = arange(to, tf+dt, dt) # vector de tiempos
            N = len(t) - 1
            temp = Crank_Nicolson # Esquema temporal con el que se va a resolver
            U = MyIntegrator( flight , t , Uo , temp )
            Error = MyIntegratorError( flight , t , Uo , temp )
            
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 fila, 2 columnas de gráficos
            axs[0].plot(t, U[:, 2], label="Mi simulación ({temp.__name__})")  # Altura
            axs[0].plot(flight.time, flight.z[:, 1], label="Simulación rocketpy")
            axs[0].set_title("Comparación de Simulaciones de Altura", fontsize=14)
            axs[0].set_xlabel("Tiempo (s)")
            axs[0].set_ylabel("Altura (m)")
            axs[0].set_xlim(0, 25)
            
            axs[1].plot(t, Error[:, 2])  # Altura Rocketpy
            axs[1].set_title(f"Error de mi simulación por ({temp.__name__})", fontsize=14)
            axs[1].set_xlabel("Tiempo (s)")
            axs[1].set_ylabel("Altura (m)")
            fig.tight_layout()

            # Incrustar el gráfico en el contenedor de Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)  # Asociar el gráfico con Tkinter
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)  # Mostrar el gráfico

                
                          
        # Mensajes de error
        except ValueError:
            messagebox.showerror("Error", "Por favor, introduce valores válidos.")
        except Exception as e:
            messagebox.showerror("Error inesperado", str(e))

                

                
        













        
        

    