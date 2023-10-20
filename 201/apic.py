import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

n_grid = 256
n_particles = n_grid * n_grid
dx = 1 / n_grid
dt = 0.03 # 1e-4
p_mass = 1#(dx * 0.5) ** 2
bound = 3

x = ti.Vector.field(2, float, n_particles)
particle_velocity = ti.Vector.field(2, float, n_particles)
c_u = ti.Vector.field(2, float, n_particles)
c_v = ti.Vector.field(2, float, n_particles)
grid_u = ti.field(float, (n_grid + 1, n_grid))
grid_m_u = ti.field(float, (n_grid + 1, n_grid))
grid_v = ti.field(float, (n_grid, n_grid + 1))
grid_m_v = ti.field(float, (n_grid, n_grid + 1))
grid_p = ti.field(float, (n_grid, n_grid))
tmp_p = ti.field(float, (n_grid, n_grid))
cell_type = ti.field(int, (n_grid, n_grid))
grid_div = ti.field(float, (n_grid, n_grid))
color_idx = ti.field(int, shape=n_particles)
solid, fluid, air = 0, 1, 2

@ti.kernel
def init():
    block_w = 0.5
    block_h = 0.5
    block_width = int(block_w * n_grid)
    block_height = int(block_h * n_grid)
    n_cell_p = n_particles // (block_height * block_width)
    bb = bound

    n_group = 18 - 1
    for i in range(n_particles):
        cell_idx = i // n_cell_p
        x_ = cell_idx % block_width
        y_ = cell_idx // block_width
        u = (x_ + ti.random()) / block_width
        v = (y_ + ti.random()) / block_height
        x[i] = [n_grid * dx * (u * block_w + 1 / n_grid * bb + 0.0),
                n_grid * dx * (v * block_h + 1 / n_grid * bb + 0.0)]
        color_idx[i] = x_ // (block_width / n_group)
    particle_velocity.fill(ti.Vector([0, 0]))
    grid_p.fill(0.0)

@ti.kernel
def init_grid_velocity():
    grid_u.fill(0.0)
    grid_v.fill(0.0)
    grid_m_u.fill(0.0)
    grid_m_v.fill(0.0)
    cell_type.fill(air)
    grid_div.fill(0.0)

@ti.func
def in_field(i, j, li, ui, lj, uj):
    return i >= li and i <= ui and j >= lj and j <= uj

@ti.func
def biliner_coef(i0, i1, j0, j1, px, py):
    norm = (i1 - i0) * (j1 - j0)
    c00 = (i1 - px) * (j1 - py) / norm
    c10 = (px - i0) * (j1 - py) / norm
    c01 = (i1 - px) * (py - j0) / norm
    c11 = (px - i0) * (py - j0) / norm
    return ti.Vector([c00, c10, c01, c11])

@ti.func
def solid_cell(i, j):
    # return (not in_field(i, j, 0, n_grid - 1, 0, n_grid - 1))
    return i < bound or i > n_grid - bound - 1 or j < bound or j > n_grid - 1 - bound or cell_type[i, j] == solid

@ti.func
def fluid_cell(i, j):
    return (not solid_cell(i, j)) and cell_type[i, j] == fluid

@ti.func
def air_cell(i, j):
    return (not solid_cell(i, j)) and cell_type[i, j] == air

@ti.func
def togrid_apic(mass, velocity, cp, field_v, field_m, px, py, li, ui, lj, uj, is_u):
    par = ti.Vector([px, py])
    base = int(ti.round(par))
    # base = int(ti.floor(par))
    fx = par - base
    w = [0.5 * (1.5 - ti.abs(fx+1.0)) ** 2, 0.75 - fx ** 2, 0.5 * (1.5 - ti.abs(fx-1.0)) ** 2]
    # w = [1 - fx, fx]
    for i, j in ti.static(ti.ndrange(3, 3)):
        offset = ti.Vector([i, j]) - 1
        weight = w[i].x * w[j].y
        idx = base + offset
        xi_xp = dx * (idx - par)
        if in_field(idx.x, idx.y, li, ui, lj, uj):
            # print(ti.math.dot(cp, xi_xp))
            field_v[idx] += weight * mass * (velocity + cp.x * xi_xp.x + cp.y * xi_xp.y)
            field_m[idx] += weight * mass

@ti.func
def fromgrid(Xp):
    c_u = ti.Vector([0.0, 0.0])
    c_v = ti.Vector([0.0, 0.0])

    # u
    px, py = Xp.x / dx, Xp.y / dx - 0.5
    li, ui, lj, uj = 0, n_grid, 0, n_grid - 1
    wsum = 0.0
    u = 0.0
    par = ti.Vector([px, py])
    base = int(ti.round(par))
    # base = int(ti.floor(par))
    fx = par - base
    w1 = [0.5 * (1.5 - abs(fx+1.0)) ** 2, 0.75 - fx ** 2, 0.5 * (1.5 - abs(fx-1.0)) ** 2]
    # w1 = [1 - fx, fx]
    for i, j in ti.static(ti.ndrange(3, 3)):
        offset = ti.Vector([i, j]) - 1
        weight = w1[i].x * w1[j].y
        idx = base + offset
        xi_xp = dx * (idx - par)
        if in_field(idx.x, idx.y, li, ui, lj, uj):
            c_u += 4 * weight * grid_u[idx] * xi_xp / (dx**2)
            u += weight * grid_u[idx]
            wsum += weight
    u /= wsum
    c_u /= wsum

    # v
    px, py = Xp.x / dx - 0.5, Xp.y / dx
    li, ui, lj, uj = 0, n_grid - 1, 0, n_grid
    wsum = 0.0
    v = 0.0
    par = ti.Vector([px, py])
    base = int(ti.round(par))
    # base = int(ti.floor(par))
    fx = par - base
    w2 = [0.5 * (1.5 - abs(fx+1.0)) ** 2, 0.75 - fx ** 2, 0.5 * (1.5 - abs(fx-1.0)) ** 2]
    # w2 = [1 - fx, fx]
    for i, j in ti.static(ti.ndrange(3, 3)):
        offset = ti.Vector([i, j]) - 1
        weight = w2[i].x * w2[j].y
        idx = base + offset
        xi_xp = dx * (idx - par)
        if in_field(idx.x, idx.y, li, ui, lj, uj):
            c_v += 4 * weight * grid_v[idx] * xi_xp / (dx**2)
            v += weight * grid_v[idx]
            wsum += weight
    v /= wsum
    c_v /= wsum

    return ti.Vector([u, v]), c_u, c_v

@ti.kernel
def p2g():
    for p in x:
        u, v = particle_velocity[p]
        Xp = x[p]

        # u
        px, py = Xp.x / dx, (Xp.y - 0.5 * dx) / dx
        togrid_apic(p_mass, u, c_u[p], grid_u, grid_m_u, px, py, 0, n_grid, 0, n_grid - 1, True)

        # v
        px, py = (Xp.x - 0.5 * dx) / dx, Xp.y / dx
        togrid_apic(p_mass, v, c_v[p], grid_v, grid_m_v, px, py, 0, n_grid - 1, 0, n_grid, False)

    for i, j in grid_u:
        if grid_m_u[i, j] > 0:
            grid_u[i, j] /= grid_m_u[i, j]

    for i, j in grid_v:
        if grid_m_v[i, j] > 0:
            grid_v[i, j] /= grid_m_v[i, j]

@ti.kernel
def g2p(t: ti.f32) -> ti.i32:
    ret = 0
    bb = bound# + 1
    for p in x:
        Xp = x[p]
        vel, c_u[p], c_v[p] = fromgrid(Xp)
        x[p] += t * vel
        if x[p].x <= bb * dx and vel.x < 0:
            vel.x = 0#-vel.x
        if x[p].y <= bb * dx and vel.y < 0:
            vel.y = 0#-vel.y
        if x[p].x >= (n_grid - bb) * dx and vel.x > 0:
            vel.x = 0#-vel.x
        if x[p].y >= (n_grid - bb) * dx and vel.y < 0:
            vel.y = 0#-vel.y
        x[p] =  clamp(x[p], bb * dx, (n_grid - bb) * dx)
        # x[p] =  tmp_xp
        particle_velocity[p] = vel

    return ret

@ti.kernel
def add_force(dt: ti.f32):
    for i, j in grid_v:
        if solid_cell(i, j) or solid_cell(i, j - 1):
            continue
        # if fluid_cell(i, j) or fluid_cell(i, j - 1):
        grid_v[i, j] += -1 * dt

@ti.func
def clamp(val, l, r):
    return ti.min(ti.max(val, l), r)

@ti.kernel
def mark_cells():
    for p in x:
        Xp = x[p]
        i = clamp(int(Xp.x / dx), 0, n_grid - 1)
        j = clamp(int(Xp.y / dx), 0, n_grid - 1)
        if not solid_cell(i, j):
            cell_type[i, j] = fluid

@ti.kernel
def divergence():
    for i, j in grid_div:
        _dx, _dy = 0.0, 0.0
        if (not solid_cell(i - 1, j)) and (not solid_cell(i, j)):
            _dx += - grid_u[i, j]
        if (not solid_cell(i + 1, j)) and (not solid_cell(i, j)):
            _dx += grid_u[i + 1, j]
        if (not solid_cell(i, j - 1)) and (not solid_cell(i, j)):
            _dy += - grid_v[i, j]
        if (not solid_cell(i, j)) and (not solid_cell(i, j + 1)):
            _dy += grid_v[i, j + 1] 
        grid_div[i, j] = _dx + _dy

@ti.kernel
def solve_pressure_jacobian():
    damping = 0.67
    tmp_p.fill(0.0)
    for i, j in grid_p:
        if not fluid_cell(i, j):
            continue
        coef = 0.0
        coef += float(not solid_cell(i + 1, j))
        coef += float(not solid_cell(i - 1, j))
        coef += float(not solid_cell(i, j + 1))
        coef += float(not solid_cell(i, j - 1))
        P = 0.0
        if fluid_cell(i + 1, j):
            P += grid_p[i + 1, j]
        if fluid_cell(i - 1, j):
            P += grid_p[i - 1, j]
        if fluid_cell(i, j + 1):
            P += grid_p[i, j + 1]
        if fluid_cell(i, j - 1):
            P += grid_p[i, j - 1]
        new = (P - grid_div[i, j]) / coef
        last = grid_p[i, j]
        tmp_p[i, j] = (1.0 - damping) * last + damping * new

@ti.kernel
def projection():
    for i, j in grid_u:
        if (not solid_cell(i - 1, j)) and (not solid_cell(i, j)):
            if fluid_cell(i - 1, j) or fluid_cell(i, j):
                grid_u[i, j] -= (grid_p[i, j] - grid_p[i - 1, j])
    for i, j in grid_v:
        if (not solid_cell(i, j - 1)) and (not solid_cell(i, j)):
            if fluid_cell(i, j - 1) or fluid_cell(i, j):
                grid_v[i, j] -= (grid_p[i, j] - grid_p[i, j - 1])

@ti.kernel
def enforce_bc():
    for i, j in grid_u:
        if solid_cell(i - 1, j) and grid_u[i, j] < 0:
            grid_u[i, j] = 0#-grid_u[i, j]
        if solid_cell(i, j) and grid_u[i, j] > 0:
            grid_u[i, j] = 0#-grid_u[i, j]
    for i, j in grid_v:
        if solid_cell(i, j - 1) and grid_v[i, j] < 0:
            grid_v[i, j] = 0#-grid_v[i, j]
        if solid_cell(i, j) and grid_v[i, j] > 0:
            grid_v[i, j] = 0#-grid_v[i, j]

def substep(t):
    init_grid_velocity()
    mark_cells()
    p2g()
    add_force(t)
    divergence()
    for _ in range(10):
        solve_pressure_jacobian()
        grid_p.copy_from(tmp_p)
    projection()
    g2p(t)
    # for _ in range(2):
    #     g2p(0.5 * t)

import cv2
def toti(img):
    if len(img.shape) == 3:
        img = img.transpose(1,0,2)
    else:
        img = img.transpose(1,0)
    return cv2.flip(img, 0)

init()
gui = ti.GUI("fuck", 512)
fid = 0

color = np.array([(_ << (4 * 2)) + 0x0600ff for _ in np.random.randint(0x44, 0x195, size=18)])
print(color)

frame_cnt = 0
result_dir = "./results/apic"
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)
while gui.running and not gui.get_event(gui.ESCAPE):
    for _ in range(10):
        substep(dt / 10)
    # substep(dt)
    gui.clear(0x112F41)
    gui.circles(x.to_numpy()/(n_grid * dx), radius=1, palette=color, palette_indices=color_idx)
    # img = gui.get_image()
    # cv2.imwrite(f"f_{fid:04d}.png", cv2.cvtColor(toti(img), cv2.COLOR_RGB2BGR) * 255)
    fid += 1
    pixels_img = gui.get_image()
    gui.show()
    video_manager.write_frame(pixels_img)
    frame_cnt += 1
    if frame_cnt >= 240:
        break

print()
print('Exporting .mp4 and .gif videos...')
video_manager.make_video(gif=True, mp4=True)
print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')
print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')