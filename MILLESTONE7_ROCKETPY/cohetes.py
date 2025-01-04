from rocketpy import SolidMotor, Rocket

def get_calisto():

    # Motor sólido
    Pro75M1670 = SolidMotor(
    thrust_source="Cesaroni_M1670.eng",
    dry_mass=1.815,
    dry_inertia=(0.125, 0.125, 0.002),
    center_of_dry_mass_position=0.317,
    grains_center_of_mass_position=0.397,
    burn_time=3.9,
    grain_number=5,
    grain_separation=0.005,
    grain_density=1815,
    grain_outer_radius=0.033,
    grain_initial_inner_radius=0.015,
    grain_initial_height=0.12,
    nozzle_radius=0.033,
    throat_radius=0.011,
    interpolation_method="linear",
    nozzle_position=0,
    coordinate_system_orientation="nozzle_to_combustion_chamber",
    )   

    calisto = Rocket(
        radius=0.0635,
        mass=14.426,  # without motor
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="Team13_CD.csv",
        power_on_drag="Team13_CD.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    calisto.set_rail_buttons(
        upper_button_position=0.0818,
        lower_button_position=-0.6182,
        angular_position=45,
    )

    calisto.add_motor(Pro75M1670, position=-1.255)

    calisto.add_nose(
        length=0.55829, kind="vonKarman", position=1.278
    )

    calisto.add_trapezoidal_fins(
        n=4,
        root_chord=0.120,
        tip_chord=0.040,
        span=0.100,
        sweep_length=None,
        cant_angle=0,
        position=-1.04956,
    )

    calisto.add_tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
    )
    #calisto.draw()
    
    return calisto 

