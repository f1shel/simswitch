import numpy as np
import cv2

import taichi as ti
ti.init(arch=ti.gpu)

window_size = 255
grid_size = 255
δ = 1e-3
h = 10
ν = 1e-1
fluv = 20000000

face_u = ti.field(float, shape=(grid_size+1, grid_size+1))
face_v = ti.field(float, shape=(grid_size+1, grid_size+1))
cell_u = ti.Vector.field(2, float, shape=(grid_size, grid_size))
cell_p = ti.field(float, shape=(grid_size, grid_size))
cell_u_div = ti.field(float, shape=(grid_size, grid_size))
dye = ti.Vector.field(3, float, shape=(grid_size, grid_size))

u_tmp = ti.Vector.field(2, float, shape=(grid_size, grid_size))
p_tmp = ti.field(float, shape=(grid_size, grid_size))
dye_tmp = ti.Vector.field(3, float, shape=(grid_size, grid_size))

@ti.func
def lerp(l, r, frac):
    return l + frac * (r - l)

@ti.kernel
def start():
    face_u.fill(0)
    face_v.fill(0)

@ti.func
def sample_bc(field, u, v, U):
    I = ti.Vector([int(u), int(v)])
    I = ti.max(0, ti.min(U - 1, I))
    return field[I]

@ti.func
def bilerp(field, p, U):
    u, v = p
    iu, iv = ti.floor(u), ti.floor(v)
    fu, fv = u - iu, v - iv
    a = sample_bc(field, iu, iv, U)
    b = sample_bc(field, iu + 1, iv, U)
    c = sample_bc(field, iu, iv + 1, U)
    d = sample_bc(field, iu + 1, iv + 1, U)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)


@ti.kernel
def add_force(cur_frame: int):
    center = grid_size // 2
    for i, j in face_u:
        u = i / grid_size
        v = j / grid_size
        if cur_frame > 100 and cur_frame < 650:
            if u > 0.1 and u < 0.12 and j >= center - 5 and j <= center + 5:
                dye[i, j] = ti.Vector([1, 0, 0])
                face_u[i, j] += fluv * δ
            if u > 0.88 and u < 0.9 and j >= center - 5 and j <= center + 5:
                dye[i, j] = ti.Vector([0, 1, 0])
                face_u[i, j] -= fluv * δ
            if v > 0.1 and v < 0.12 and u > 0.45 and u < 0.55:
                dye[i, j] = ti.Vector([1, 1, 1])
                face_v[i, j] += fluv * δ
            if v > 0.88 and v < 0.9 and u > 0.45 and u < 0.55:
                dye[i, j] = ti.Vector([1, 1, 1])
                face_v[i, j] -= fluv * δ
        # if i != 0 and i != grid_size and j != 0 and j != grid_size:
        #     face_v[i, j] -= 980 * δ

@ti.kernel
def advect():
    for i, j in cell_u:
        q = ti.Vector([i, j])
        u_q = ti.Vector([
            bilerp(face_u, q + ti.Vector([0.5, 0.0]), grid_size+1),
            bilerp(face_v, q + ti.Vector([0.0, 0.5]), grid_size+1)
        ])
        s = q - (0.5 * δ * u_q) / h
        u_s = ti.Vector([
            bilerp(face_u, s + ti.Vector([0.5, 0.0]), grid_size+1),
            bilerp(face_v, s + ti.Vector([0.0, 0.5]), grid_size+1)
        ])
        t = s - (0.5 * δ * u_s) / h
        u_t = ti.Vector([
            bilerp(face_u, t + ti.Vector([0.5, 0.0]), grid_size+1),
            bilerp(face_v, t + ti.Vector([0.0, 0.5]), grid_size+1)
        ])
        cell_u[i, j] = u_t
        dye_tmp[i, j] = bilerp(dye, t, grid_size) * 1.0

    for i, j in cell_u:
        dye[i, j] = dye_tmp[i, j]

@ti.kernel
def diffusion():
    for i, j in cell_u:
        ul = sample_bc(cell_u, i - 1, j, grid_size)
        ur = sample_bc(cell_u, i + 1, j, grid_size)
        ub = sample_bc(cell_u, i, j - 1, grid_size)
        ut = sample_bc(cell_u, i, j + 1, grid_size)
        uc = cell_u[i, j]
        if i == 0:
            ul.x = -uc.x
        if i == grid_size - 1:
            ur.x = -uc.x
        if j == 0:
            ub.y = -uc.y
        if j == grid_size - 1:
            ut.y = -uc.y
        laplace = (ul + ur + ut + ub - 4 * uc) / (h ** 2)
        cell_u[i, j] = cell_u[i, j] + ν * laplace * δ

    for i, j in face_u:
        face_u[i, j] = bilerp(cell_u, ti.Vector([i-0.5, j]), grid_size).x
        face_v[i, j] = bilerp(cell_u, ti.Vector([i, j-0.5]), grid_size).y
        if i == 0 or i == grid_size:
            face_u[i, j] = 0
        if j == 0 or j == grid_size:
            face_v[i, j] = 0

@ti.kernel
def divergence():
    for i, j in cell_u_div:
        ul = sample_bc(face_u, i, j, grid_size+1)
        ur = sample_bc(face_u, i + 1, j, grid_size+1)
        vb = sample_bc(face_v, i, j, grid_size+1)
        vt = sample_bc(face_v, i, j + 1, grid_size+1)
        if i == 0:
            ul = 0
        if i == grid_size - 1:
            ur = 0
        if j == 0:
            vb = 0
        if j == grid_size - 1:
            vt = 0
        cell_u_div[i, j] = (ur - ul + vt - vb) / δ # divergence times h

@ti.kernel
def solve_possion_jacobian():
    for i, j in p_tmp:
        pl = sample_bc(cell_p, i - 1, j, grid_size)
        pr = sample_bc(cell_p, i + 1, j, grid_size)
        pb = sample_bc(cell_p, i, j - 1, grid_size)
        pt = sample_bc(cell_p, i, j + 1, grid_size)
        # if i == 0:
        #     pl = 0
        # if i == grid_size - 1:
        #     pr = 0
        # if j == 0:
        #     pb = 0
        # if j == grid_size - 1:
        #     pt = 0
        div = cell_u_div[i, j]
        p_tmp[i, j] = (pl + pr + pb + pt - div) * 0.25
    for i, j in cell_p:
        cell_p[i, j] = p_tmp[i, j]

@ti.kernel
def projection():
    for i, j in face_u:
        pl = sample_bc(cell_p, i - 1, j, grid_size)
        pr = sample_bc(cell_p, i, j, grid_size)
        pb = sample_bc(cell_p, i, j - 1, grid_size)
        pt = sample_bc(cell_p, i, j, grid_size)
        face_u[i, j] -= δ / h * (pr - pl)
        face_v[i, j] -= δ / h * (pt - pb)
        if i == 0 or i == grid_size:
            face_u[i, j] = 0
        if j == 0 or j == grid_size:
            face_v[i, j] = 0

def step(cur_frame):
    # naiver-stokes equation
    # (1) Momentum: Du/Dt = ∂u/∂t + u·▽u = -(1/ρ)▽p + ν△u + a
    # (2) Incompressbility: ▽·u = 0

    # step 1: external acceleration
    # ∂u/∂t = u + δ a
    add_force(cur_frame)

    # step 2: advection
    # ∂u/∂t = -u·▽u
    advect()

    # step 3: diffusion
    # ∂u/∂t = ν△u
    diffusion()

    # step 4: pressure projection
    # ∂u/∂t = -(1/ρ)▽p
    divergence()
    for _ in range(128):
        solve_possion_jacobian()
    projection()

def render(gui):
    gui.set_image(dye)
    gui.show()

def main():
    start()
    gui = ti.GUI("stable fluid", window_size)
    cur_frame = 0
    while gui.running and not gui.get_event(gui.ESCAPE):
        step(cur_frame)
        render(gui)
        cur_frame += 1

if __name__ == "__main__":
    main()