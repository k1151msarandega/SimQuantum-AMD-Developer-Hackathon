"""Step 1 sanity check: does JAX see the MI300X over ROCm, and does QArray run?

Run this on the MI300X instance itself, after installing dependencies
(see README setup notes). This does NOT get imported by anything else --
it's a standalone diagnostic, not part of the package.
"""
import time

import jax

print("JAX devices:", jax.devices())
print("Default backend:", jax.default_backend())

from qarray import ChargeSensedDotArray
import numpy as np

model = ChargeSensedDotArray(
    Cdd=[[0.1, 0.05], [0.05, 0.1]],
    Cgd=[[1.0, 0.1], [0.1, 1.0]],
    Cds=[[0.1, 0.0]],
    Cgs=[[0.1, 0.0]],
    coulomb_peak_width=0.1,
    T=0.0,
)

t0 = time.time()
vx = np.linspace(-5, 5, 200)
vy = np.linspace(-5, 5, 200)
z, n = model.charge_sensor_open(vx, vy)
t1 = time.time()

print(f"Generated {z.shape} diagram in {t1 - t0:.4f}s")
print("Backend used for this call:", jax.default_backend())
