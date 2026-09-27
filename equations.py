import numpy as np


N_PARAMS = 1_039_968
FP32_BYTES = 4


def flops(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    return B * (17_712 * S**2 + 313_344)


def memory(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    return FP32_BYTES * (N_PARAMS + B * (26 * S**2 + 868))


def bytes_moved(image_size, batch):
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    return FP32_BYTES * (N_PARAMS + B * (91 * S**2 + 2148))


def latency(image_size, batch, theta):
    F = flops(image_size, batch)
    Q = bytes_moved(image_size, batch)
    return theta["t0"] + np.maximum(
        F / theta["compute_flops_s"],
        Q / theta["bandwidth_bytes_s"],
    )


def energy(image_size, batch, theta_energy):
    F = flops(image_size, batch) / 1e9
    Q = bytes_moved(image_size, batch) / 1e9
    return (
        theta_energy["e0"]
        + theta_energy["e_per_gflop"] * F
        + theta_energy["e_per_gbyte"] * Q
    )
