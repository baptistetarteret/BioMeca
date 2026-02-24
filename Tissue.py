import numpy as np


class Tissue:
    def __init__(self, name: str, Center: np.ndarray, Radius: float,
                 absorption_coefficients: float, refractive_index: float,
                 density: float = 1040.0,
                 specific_heat: float = 3700.0,
                 thermal_conductivity: float = 0.50):
        self.Center = Center
        self.Radius = Radius
        self.name = name
        self.absorption_coefficients = absorption_coefficients
        self.refractive_index = refractive_index
        # Propriétés thermiques (kg/m³, J/kg/K, W/m/K)
        self.density = density
        self.specific_heat = specific_heat
        self.thermal_conductivity = thermal_conductivity
