import taichi as ti
ti.init(arch=ti.cuda)
import numpy as np

n_grid_x = 256
n_grid_y = 128
dx = 1.0 / n_grid_x
denorm = dx * np.array([[n_grid_x, n_grid_y]])
dt = 4e-3
rho = 1 # particle density
vol = (dx * 0.5) ** 2 # particle(material point) volume
mass = rho * vol # particle mass
gravity = 9.8
E = 400
bound = 2

n_p_max = 32768
n_p = ti.field(ti.f32, ()) # current number of particles
part_x = ti.Vector.field(2, ti.f32, n_p_max) # position of particles
part_v = ti.Vector.field(2, ti.f32, n_p_max) # velocity of particles
part_C = ti.Matrix.field(2, 2, ti.f32, n_p_max) # C matrix of particles
part_J = ti.field(ti.f32, n_p_max) # volume ratio of particles
grid_v = ti.Vector.field(2, ti.f32, (n_grid_x, n_grid_y)) # velocity of grid nodes
grid_m = ti.field(ti.f32, (n_grid_x, n_grid_y)) # mass of grid nodes

body_o = ti.Vector.field(2, ti.f32, ()) # center of body
body_v = ti.Vector.field(2, ti.f32, ())
body_r = 0.04 * n_grid_x * dx # radius of body
body_rho = 0.5
body_mass = body_rho * np.pi * body_r**2 
body_canvas_data = ti.Vector.field(2, ti.f32, 128) # visualize

@ti.kernel
def init():
    body_o[None] = ti.Vector([0.5 * denorm[0,0], 0.8 * denorm[0,1]]) 
    body_v[None] = ti.Vector([0, 0])
    len = body_canvas_data.shape[0]
    for i in body_canvas_data:
        theta = (i / len) * 2 * np.pi
        body_canvas_data[i] = body_r * ti.Vector([ti.cos(theta), ti.sin(theta)])

@ti.kernel
def update_jet():
    if n_p[None] + 50 < n_p_max:
        for i in range(n_p[None], n_p[None] + 50):
            part_x[i] = ti.Vector([ti.random() * 0.03 + 0.92, ti.random() * 0.07 + 0.5]) * dx * ti.Vector([n_grid_x, n_grid_y])
            part_v[i] = ti.Vector([-1.5, 0.0])
            part_J[i] = 1.0
            part_C[i] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        n_p[None] += 50

@ti.func
def in_body(x):
    ret = False
    if ti.math.length(x - body_o[None]) <= body_r:
        ret = True
    return ret

@ti.kernel
def p2g(dt: ti.f32):
    body_v[None].y += -dt * gravity
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in range(n_p[None]):
        part_node = part_x[p] / dx
        base_node = int(part_node - 0.5)
        fx = part_node - base_node
        # quadratic spline weights
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = -dt * 4 * E * vol * part_J[p] * (part_J[p] - 1) / dx**2
        affine = ti.Matrix([[stress, 0], [0, stress]]) + mass * part_C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base_node + offset] += weight * (mass * part_v[p] + affine @ dpos)
            grid_m[base_node + offset] += weight * mass
    body_impulse = ti.Vector([0.0, 0.0])
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j].y -= dt * gravity
        # grid_collision with border
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j].x = 0
        if i > n_grid_x - 1 - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = 0
        if j < bound and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0
        if j > n_grid_y - 1 - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0
        # grid collision with body
        x = ti.Vector([i, j]) * dx
        if in_body(x):
            g_v = grid_v[i, j]
            n = ti.math.normalize(x - body_o[None])
            vn_mag = ti.math.dot(g_v, n)
            vt = g_v - vn_mag * n
            bvn_mag = ti.math.dot(body_v[None], n)
            if vn_mag - bvn_mag < 0.0:
                vn_mag = -0.05 * (vn_mag - bvn_mag) + bvn_mag
            grid_v[i, j] = vt + vn_mag * n
            node_impulse = grid_m[i, j] * (grid_v[i, j] - g_v)
            body_impulse += -node_impulse
    body_v[None] += body_impulse / body_mass

@ti.kernel
def g2p(dt: ti.f32):
    body_impulse = ti.Vector([0.0, 0.0])
    for p in range(n_p[None]):
        part_node = part_x[p] / dx
        base_node = int(part_node - 0.5)
        fx = part_node - base_node
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        # wsum = 0.0
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base_node + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        part_J[p] *= 1 + dt * new_C.trace()
        part_C[p] = new_C
        part_v[p] = new_v
        # particle collision with border
        p_x, p_v = part_x[p] / dx, part_v[p]
        i, j = p_x
        if i < bound and p_v.x < 0:
            p_v.x = 0
        if i > n_grid_x - 1 - bound and p_v.x > 0:
            p_v.x = 0
        if j < bound and p_v.y < 0:
            p_v.y = 0
        if j > n_grid_y - 1 - bound and p_v.y > 0:
            p_v.y = 0
        part_v[p] = p_v
        # particle collision with body
        if in_body(part_x[p]):
            p_v = part_v[p]
            n = ti.math.normalize(part_x[p] - body_o[None])
            vn_mag = ti.math.dot(p_v, n)
            vt = p_v - vn_mag * n
            bvn_mag = ti.math.dot(body_v[None], n)
            if vn_mag - bvn_mag < 0.0:
                vn_mag = -0.05 * (vn_mag - bvn_mag) + bvn_mag
            part_v[p] = vt + vn_mag * n
            particle_impulse = mass * (part_v[p] - p_v)
            body_impulse -= particle_impulse
        part_x[p] += dt * part_v[p]
    body_v[None] += body_impulse / body_mass
    # body collision with border
    if (body_o[None].y - body_r) / dx < bound - 0.3 and body_v[None].y < 0:
        body_v[None].y *= -0.6
    if (body_o[None].y + body_r) / dx > n_grid_y - 1 - bound and body_v[None].y > 0:
        body_v[None].y *= -0.6
    if (body_o[None].x - body_r) / dx < bound - 0.3 and body_v[None].x < 0:
        body_v[None].x *= -0.6
    if (body_o[None].x + body_r) / dx > n_grid_x - 1 - bound and body_v[None].x > 0:
        body_v[None].x *= -0.6
    body_o[None] += dt * body_v[None]

def substep(dt):
    p2g(dt)
    g2p(dt)

init()
gui = ti.GUI("mpm two-way coupling", (512, 512 * n_grid_y // n_grid_x))
while gui.running and not gui.get_event(gui.ESCAPE):
    update_jet()
    for _ in range(50):
        substep(dt / 50)
    gui.clear(0x000000)
    gui.circles(part_x.to_numpy() / denorm, color=0x068587)
    bcd = body_canvas_data.to_numpy()
    bo = body_o.to_numpy()
    gui.lines(begin=(bcd[:-1] + bo) / denorm, end=(bcd[1:] + bo) / denorm, color=0xffffff)
    # gui.circles(body_o.to_numpy().reshape(-1,2), radius=body_r * 512, color=0xffffff)
    gui.rect(
        np.array([bound * dx / (n_grid_x*dx), (n_grid_y - bound)*dx / (n_grid_y*dx)]),
        np.array([(n_grid_x - bound)*dx / (n_grid_x*dx), bound * dx / (n_grid_y*dx)]))
    gui.show()