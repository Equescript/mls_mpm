from ctypes import CDLL, POINTER, c_int, c_double
import random
import taichi as ti
import numpy as np
from numpy.ctypeslib import ndpointer

nd_int_array_dim1 = ndpointer(dtype=np.int32, ndim=1, flags='C')
nd_double_array_dim2 = ndpointer(dtype=np.double, ndim=2, flags='C')

mls_mpm_dll = CDLL("./build/Release/mls_mpm.dll")
mls_mpm_dll.init.argtypes = [nd_double_array_dim2, nd_int_array_dim1, c_int]
mls_mpm_dll.step.argtypes = []

FLUID = 3000
JELLY = 3000
SNOW = 3000

N_PARTICLES = FLUID + JELLY + SNOW

P_x = np.zeros((N_PARTICLES, 2), dtype=np.double)
P_material = np.zeros((N_PARTICLES), dtype=np.int32)

for i in range(0, FLUID):
    P_x[i] = [
        random.random() * 0.2 + 0.3 + 0.10 * 1,
        random.random() * 0.2 + 0.05 + 0.32 * 0,
    ]
    P_material[i] = 0
for i in range(FLUID, FLUID+JELLY):
    P_x[i] = [
        random.random() * 0.2 + 0.3 + 0.10 * 0,
        random.random() * 0.2 + 0.05 + 0.32 * 1,
    ]
    P_material[i] = 1
for i in range(FLUID+JELLY, N_PARTICLES):
    P_x[i] = [
        random.random() * 0.2 + 0.3 + 0.10 * 1.5,
        random.random() * 0.2 + 0.05 + 0.32 * 2,
    ]
    P_material[i] = 2

mls_mpm_dll.init(P_x, P_material, N_PARTICLES)

gui = ti.GUI("MLS_MPM")

while gui.running and not gui.get_event(gui.ESCAPE):
    for s in range(20):
        mls_mpm_dll.step()
    gui.clear(0x112F41)
    gui.circles(
        P_x,
        radius=1.5,
        palette=[0x068587, 0xED553B, 0xEEEEF0],
        palette_indices=P_material,
    )
    gui.show()
