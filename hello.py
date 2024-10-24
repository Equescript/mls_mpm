from ctypes import CDLL, POINTER, c_int, c_double
import random
import taichi as ti
import numpy as np
from numpy.ctypeslib import ndpointer
import pickle

nd_int_array_dim1 = ndpointer(dtype=np.int32, ndim=1, flags='C')
nd_double_array_dim2 = ndpointer(dtype=np.double, ndim=2, flags='C')

mls_mpm_dll = CDLL("./build/Release/mls_mpm.dll")
mls_mpm_dll.init.argtypes = [nd_double_array_dim2, nd_int_array_dim1, c_int]
mls_mpm_dll.step.argtypes = []

N_PARTICLES = 10000

SET_P_X = False
SET_COLOR = False

if SET_P_X:
    P_x = np.zeros((N_PARTICLES, 2), dtype=np.double)
P_material = np.zeros((N_PARTICLES), dtype=np.int32)
if SET_COLOR:
    P_color = np.zeros((N_PARTICLES), dtype=np.int32)

for i in range(0, N_PARTICLES):
    if SET_P_X:
        P_x[i] = [
            random.random() * 0.6 + 0.2,
            random.random() * 0.6 + 0.2,
        ]
    P_material[i] = 0
    if SET_COLOR:
        P_color[i] = 0

if SET_P_X:
    with open("P_x.pkl", "wb") as P_x_file:
        pickle.dump(P_x, P_x_file)
else:
    with open("P_x.pkl", "rb") as P_x_file:
        P_x = pickle.load(P_x_file)

if not SET_COLOR:
    with open("P_color.pkl", "rb") as P_color_file:
        P_color = pickle.load(P_color_file)

def in_rectangle(x, min, max) -> bool:
    return x[0] > min[0] and x[0] < max[0] and x[1] > min[1] and x[1] < max[1]

def in_H(x, origin, scaling) -> bool:
    return in_rectangle(x, origin, [origin[0]+1*scaling, origin[1]+5*scaling]) or \
    in_rectangle(x, [origin[0]+1*scaling, origin[1]+2*scaling], [origin[0]+2*scaling, origin[1]+3*scaling]) or \
    in_rectangle(x, [origin[0]+2*scaling, origin[1]], [origin[0]+3*scaling, origin[1]+5*scaling])

def in_E(x, origin, scaling) -> bool:
    return in_rectangle(x, origin, [origin[0]+1*scaling, origin[1]+5*scaling]) or \
    in_rectangle(x, origin, [origin[0]+3*scaling, origin[1]+1*scaling]) or \
    in_rectangle(x, [origin[0], origin[1]+2*scaling], [origin[0]+3*scaling, origin[1]+3*scaling]) or \
    in_rectangle(x, [origin[0], origin[1]+4*scaling], [origin[0]+3*scaling, origin[1]+5*scaling])

def in_L(x, origin, scaling) -> bool:
    return in_rectangle(x, origin, [origin[0]+1*scaling, origin[1]+5*scaling]) or \
    in_rectangle(x, origin, [origin[0]+3*scaling, origin[1]+1*scaling])

def in_O(x, origin, scaling) -> bool:
    return in_rectangle(x, origin, [origin[0]+1*scaling, origin[1]+5*scaling]) or \
    in_rectangle(x, [origin[0]+1*scaling, origin[1]], [origin[0]+2*scaling, origin[1]+1*scaling]) or \
    in_rectangle(x, [origin[0]+1*scaling, origin[1]+4*scaling], [origin[0]+2*scaling, origin[1]+5*scaling]) or \
    in_rectangle(x, [origin[0]+2*scaling, origin[1]], [origin[0]+3*scaling, origin[1]+5*scaling])

SCALING = 0.03
TEXT_ORIGIN = [0.2, 0.1]

def set_color():
    for i in range(0, N_PARTICLES):
        if in_H(P_x[i], [TEXT_ORIGIN[0], TEXT_ORIGIN[1]], SCALING) or \
            in_E(P_x[i], [TEXT_ORIGIN[0]+0.12, TEXT_ORIGIN[1]], SCALING) or \
            in_L(P_x[i], [TEXT_ORIGIN[0]+0.24, TEXT_ORIGIN[1]], SCALING) or \
            in_L(P_x[i], [TEXT_ORIGIN[0]+0.36, TEXT_ORIGIN[1]], SCALING) or \
            in_O(P_x[i], [TEXT_ORIGIN[0]+0.48, TEXT_ORIGIN[1]], SCALING):
            P_color[i] = 1
    with open("P_color.pkl", "wb") as P_color_file:
        pickle.dump(P_color, P_color_file)


mls_mpm_dll.init(P_x, P_material, N_PARTICLES)

gui = ti.GUI("HELLO")

counter = 0
while gui.running and not gui.get_event(gui.ESCAPE):
    for s in range(20):
        mls_mpm_dll.step()
    gui.clear(0xFFFFFF)
    gui.circles(
        P_x,
        radius=2,
        palette=[0x6AE1FF, 0xFF7364],
        palette_indices=P_color,
    )
    gui.show(str(counter)+".png")
    if SET_COLOR and counter == 340:
        set_color()
    print(counter)
    counter += 1
