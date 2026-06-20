import numpy as np

def resolve_weight_initializer(initialization_name:str, n_in: int, n_out: int) -> np.ndarray:
    match initialization_name:
        case "xavier":
            return np.sqrt(2. / (n_in + n_out))
        case "he":
            return np.sqrt(2. / n_in)
        case "le_cun":
            return np.sqrt(1. / n_in)
        case _:
            raise ValueError(f"Initilization function for weights'{initialization_name}' not recognized.")
        
def resolve_bias_initializer(initialization_name:str, n_in: int) -> np.ndarray:
    match initialization_name:
        case "zeroes":
            return np.zeros((n_in, 1))
        case _:
            raise ValueError(f"Initilization function for bias '{initialization_name}' not recognized.")