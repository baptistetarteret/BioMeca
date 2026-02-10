import numpy as np


class Tissue:
    def __init__(self, name: str, Center: np.ndarray, Radius: float, absorption_coefficients: float, refractive_index: float):
        self.Center = Center
        self.Radius = Radius
        self.name = name
        self.absorption_coefficients = absorption_coefficients
        self.refractive_index = refractive_index
