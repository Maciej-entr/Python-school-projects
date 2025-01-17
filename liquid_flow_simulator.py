import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 100
viscosity = 0.1
dt = 0.1
iterations = 50

u = np.zeros((N, N))
v = np.zeros((N, N))
p = np.zeros((N, N))

def add_force(u, v, strength=1.0):
    force_x = np.random.uniform(-1, 1, (N, N))
    force_y = np.random.uniform(-1, 1, (N, N))
    u += strength * force_x
    v += strength * force_y

def diffuse(u, v, viscosity, dt):
    for _ in range(iterations):
        u[1:-1, 1:-1] = (u[1:-1, 1:-1] + viscosity * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]) * dt) / (1 + 4 * viscosity * dt)
        v[1:-1, 1:-1] = (v[1:-1, 1:-1] + viscosity * (v[:-2, 1:-1] + v[2:, 1:-1] + v[1:-1, :-2] + v[1:-1, 2:]) * dt) / (1 + 4 * viscosity * dt)

def project(u, v, p):
    div = (u[1:-1, 2:] - u[1:-1, :-2] + v[2:, 1:-1] - v[:-2, 1:-1]) / 2.0
    p[1:-1, 1:-1] = (div[:-1, 1:-1] + div[1:, :-1]) / 4.0
    u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2])
    v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1])

def step(u, v, p, viscosity, dt):
    add_force(u, v)
    diffuse(u, v, viscosity, dt)
    project(u, v, p)

fig, ax = plt.subplots()
q = ax.quiver(u, v)

def update(frame):
    global u, v, p
    step(u, v, p, viscosity, dt)
    q.set_UVC(u, v)
    return q,

ani = FuncAnimation(fig, update, frames=200, interval=50)
plt.show()