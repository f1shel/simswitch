import taichi as ti
ti.init(arch=ti.cuda)
import numpy as np

n_grid_x = 256
n_grid_y = 128
dx = 1.0 / n_grid_x
denorm = dx * np.array([[n_grid_x, n_grid_y]])
dt = 4e-3
rho = 1 # particle density
vol = (dx * 0.75) ** 2 # particle(material point) volume
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

n_body_seg = 32
body_vert0 = ti.Vector.field(2, ti.f32, (n_body_seg**2)) # initial vertices of body
body_vert = ti.Vector.field(2, ti.f32, (n_body_seg**2)) # current vertices of body, used in body-border collision
body_T = ti.Vector.field(2, ti.f32, ()) # position of body center
body_v = ti.Vector.field(2, ti.f32, ()) # linear velocity of body center
body_inertia0 = ti.Matrix.field(3, 3, ti.f32, ()) # inertia tensor of body of initial state
body_R = ti.Matrix.field(3, 3, ti.f32, ()) # orientation of body
body_theta = ti.field(ti.f32, ()) # orientation of body
body_w = ti.Vector.field(3, ti.f32, ()) # angular velocity of body center
body_S = 0.1 # scale
body_rho = 0.7 # density
body_mass = body_S**2 * body_rho # mass
body_border_res = 0.3
body_particle_res = 0.3

@ti.func
def getR():
    M = ti.Matrix([
        [ti.cos(body_theta[None]), -ti.sin(body_theta[None]), 0.0],
        [ti.sin(body_theta[None]), ti.cos(body_theta[None]), 0.0],
        [0.0, 0.0, 1.0],
    ])

    return M

@ti.func
def v3(v2):
    return ti.Vector([v2.x, v2.y, 0.0])
@ti.func
def v2(v3):
    return ti.Vector([v3.x, v3.y])

@ti.kernel
def init():
    n_p[None] = 0
    body_R[None] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    body_theta[None] = 0.0
    body_T[None] = ti.Vector([0.6 * n_grid_x * dx, 0.75 * n_grid_y * dx])
    body_v[None] = ti.Vector([0.0, 0.0])
    body_w[None] = ti.Vector([0.0, 0.0, 0.0])
    for i in body_vert0:
        row_idx = i // n_body_seg
        col_idx = i - row_idx * n_body_seg
        body_vert0[i] = body_S * ti.Vector([col_idx / (n_body_seg - 1), row_idx / (n_body_seg - 1)])
        body_vert0[i] -= body_S * ti.Vector([0.5, 0.5])
        # calculating inertia tensor of initial/reference state
        m = body_mass / (n_body_seg**2)
        v3d = v3(body_vert0[i])
        body_inertia0[None][0, 0] += m * ti.math.dot(v3d, v3d)
        body_inertia0[None][1, 1] += m * ti.math.dot(v3d, v3d)
        body_inertia0[None][2, 2] += m * ti.math.dot(v3d, v3d)
        body_inertia0[None][0, 0] -= m * v3d.x * v3d.x
        body_inertia0[None][0, 1] -= m * v3d.x * v3d.y
        body_inertia0[None][0, 2] -= m * v3d.x * v3d.z
        body_inertia0[None][1, 0] -= m * v3d.y * v3d.x
        body_inertia0[None][1, 1] -= m * v3d.y * v3d.y
        body_inertia0[None][1, 2] -= m * v3d.y * v3d.z
        body_inertia0[None][2, 0] -= m * v3d.z * v3d.x
        body_inertia0[None][2, 1] -= m * v3d.z * v3d.y
        body_inertia0[None][2, 2] -= m * v3d.z * v3d.z
        body_vert[i] = v2(body_R[None] @ v3(body_vert0[i])) + body_T[None]

@ti.kernel
def update_jet():
    if n_p[None] + 32 < n_p_max:
        for i in range(n_p[None], n_p[None] + 32):
            part_x[i] = ti.Vector([ti.random() * 0.02 + 0.95, ti.random() * 0.1 + 0.5]) * dx * ti.Vector([n_grid_x, n_grid_y])
            part_v[i] = ti.Vector([-1, 0.0])
            part_J[i] = 1.0
            part_C[i] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        n_p[None] += 32
    if n_p[None] + 32 < n_p_max:
        for i in range(n_p[None], n_p[None] + 32):
            part_x[i] = ti.Vector([ti.random() * 0.02 + 0.05, ti.random() * 0.1 + 0.5]) * dx * ti.Vector([n_grid_x, n_grid_y])
            part_v[i] = ti.Vector([1, 0.0])
            part_J[i] = 1.0
            part_C[i] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        n_p[None] += 32

@ti.func
def dist2plane(x, o, n):
    dist = ti.math.dot(x - o, n)
    return dist

@ti.func
def in_body(x):
    # body top
    row_idx, col_idx = n_body_seg - 1, n_body_seg // 2
    idx = row_idx * n_body_seg + col_idx
    vo_top = body_vert[idx]
    vn_top = v2(body_R[None] @ ti.Vector([0.0, 1.0, 0.0]))
    dit = dist2plane(x, vo_top, vn_top)

    # body bottom
    row_idx, col_idx = 0, n_body_seg // 2
    idx = row_idx * n_body_seg + col_idx
    vo_bottom = body_vert[idx]
    vn_bottom = v2(body_R[None] @ ti.Vector([0.0, -1.0, 0.0]))
    dib = dist2plane(x, vo_bottom, vn_bottom)

    # body left
    row_idx, col_idx = n_body_seg // 2, 0
    idx = row_idx * n_body_seg + col_idx
    vo_left = body_vert[idx]
    vn_left = v2(body_R[None] @ ti.Vector([-1.0, 0.0, 0.0]))
    dil = dist2plane(x, vo_left, vn_left)

    # body right
    row_idx, col_idx = n_body_seg // 2, n_body_seg - 1
    idx = row_idx * n_body_seg + col_idx
    vo_right = body_vert[idx]
    vn_right = v2(body_R[None] @ ti.Vector([1.0, 0.0, 0.0]))
    dir = dist2plane(x, vo_right, vn_right)

    inbody = (dit <= 0.0 and dib <= 0.0 and dil <= 0.0 and dir <= 0.0)
    nearest_di = ti.max(dit, dib, dil, dir)
    nearest_bn = ti.Vector([0.0, 0.0]) # nearest body normal
    if nearest_di == dit:
        nearest_bn = vn_top
    elif nearest_di == dib:
        nearest_bn = vn_bottom
    elif nearest_di == dil:
        nearest_bn = vn_left
    elif nearest_di == dir:
        nearest_bn = vn_right
    nearest_bp = x - nearest_di * nearest_bn

    return inbody, nearest_di, nearest_bp, nearest_bn

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
    body_angular_impulse = ti.Vector([0.0, 0.0, 0.0])
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j].y -= dt * gravity
        # grid collision with border
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
        collision, nearest_di, nearest_bp, nearest_bn = in_body(x)
        if collision:
            g_v = grid_v[i, j]
            n = nearest_bn
            vn_mag = ti.math.dot(g_v, n)
            vt = g_v - vn_mag * n
            bv = body_v[None] + v2(ti.math.cross(body_w[None], v3(nearest_bp - body_T[None])))
            bvn_mag = ti.math.dot(bv, n)
            if vn_mag - bvn_mag < 0.0:
                vn_mag = -body_particle_res * (vn_mag - bvn_mag) + bvn_mag
            grid_v[i, j] = vt + vn_mag * n
            node_impulse = grid_m[i, j] * (grid_v[i, j] - g_v)
            body_impulse += -node_impulse
            body_angular_impulse += ti.math.cross(v3(x - body_T[None]), -v3(node_impulse))
    body_v[None] += body_impulse / body_mass
    body_inertia = body_R[None] @ body_inertia0[None] @ body_R[None].transpose()
    body_w[None].z += 1.0 / body_inertia[2, 2] * body_angular_impulse.z

@ti.kernel
def g2p(dt: ti.f32):
    body_impulse = ti.Vector([0.0, 0.0])
    body_angular_impulse = ti.Vector([0.0, 0.0, 0.0])
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
        collision, nearest_di, nearest_bp, nearest_bn = in_body(part_x[p])
        if collision:
            p_v = part_v[p]
            n = nearest_bn
            vn_mag = ti.math.dot(p_v, n)
            vt = p_v - vn_mag * n
            bv = body_v[None] + v2(ti.math.cross(body_w[None], v3(nearest_bp - body_T[None])))
            bvn_mag = ti.math.dot(bv, n)
            if vn_mag - bvn_mag < 0.0:
                vn_mag = -body_particle_res * (vn_mag - bvn_mag) + bvn_mag
            part_v[p] = vt + vn_mag * n
            particle_impulse = mass * (part_v[p] - p_v)
            body_impulse -= particle_impulse
            body_angular_impulse += ti.math.cross(v3(part_x[p] - body_T[None]), -v3(particle_impulse))
        part_x[p] += dt * part_v[p]
    body_v[None] += body_impulse / body_mass
    body_inertia = body_R[None] @ body_inertia0[None] @ (body_R[None].transpose())
    # print("body_angular_impulse", body_angular_impulse, 1.0 / body_inertia[2, 2] * body_angular_impulse.z)
    body_w[None].z += 1.0 / body_inertia[2, 2] * body_angular_impulse.z
    # body collision with border
    # body_r = body_S * 0.5
    # if (body_T[None].y - body_r) / dx < bound - 0.3 and body_v[None].y < 0:
    #     body_v[None].y *= -0.8
    # if (body_T[None].y + body_r) / dx > n_grid_y - 1 - bound and body_v[None].y > 0:
    #     body_v[None].y *= -0.8
    # if (body_T[None].x - body_r) / dx < bound - 0.3 and body_v[None].x < 0:
    #     body_v[None].x *= -0.8
    # if (body_T[None].x + body_r) / dx > n_grid_x - 1 - bound and body_v[None].x > 0:
    #     body_v[None].x *= -0.8

@ti.func
def cross_matrix(v):
    M = ti.Matrix([
        [0.0,-v.z,v.y],
        [v.z,0.0,-v.x],
        [-v.y,v.x,0.0],
    ])
    return M

@ti.kernel
def body_border_collision(o: ti.math.vec2, n: ti.math.vec2):
    n_collid = 0.0
    avg_r = ti.Vector([0.0, 0.0])
    for i in body_vert:
        xi = body_vert[i]
        ri = xi - body_T[None]
        vi = body_v[None] + v2(ti.math.cross(body_w[None], v3(ri)))
        dist = dist2plane(xi, o, n)
        if dist < 0 and ti.math.dot(vi, n) < 0:
            n_collid += 1.0
            avg_r += ri
    if n_collid > 0.0:
        avg_r = avg_r / n_collid
        v = body_v[None] + v2(ti.math.cross(body_w[None], v3(avg_r)))
        vn = ti.math.dot(v, n) * n
        vt = v - vn;
        vn_new = -body_border_res * vn

        v_new = vn_new + vt


        cr = cross_matrix(v3(avg_r))
        body_inertia = body_R[None] @ body_inertia0[None] @ (body_R[None].transpose())
        I_inv = body_inertia.inverse()
        K = -cr @ I_inv @ cr
        K[0, 0] += 1.0 / body_mass
        K[1, 1] += 1.0 / body_mass
        K[2, 2] += 1.0 / body_mass

        J = K.inverse() @ v3(v_new - v)
        body_v[None] += v2(J / body_mass)
        tmp = ti.math.cross(v3(avg_r), J)
        # print("tmp", tmp)
        # body_w[None] += I_inv @ tmp

@ti.kernel
def update_body(dt: ti.f32):
    body_T[None] += dt * body_v[None]
    w = body_w[None]
    body_theta[None] += dt * w.z
    body_R[None] = getR()
    for i in body_vert:
        body_vert[i] = v2(body_R[None] @ v3(body_vert0[i])) + body_T[None]

def substep(dt):
    p2g(dt)
    g2p(dt)
    body_border_collision(ti.Vector([0.5 * n_grid_x * dx, (bound - 0.1) * dx]), ti.Vector([0.0, 1.0]))
    body_border_collision(ti.Vector([(bound - 0.1) * dx, 0.5 * n_grid_y * dx]), ti.Vector([1.0, 0.0]))
    body_border_collision(ti.Vector([(n_grid_x - 1 - bound) * dx, 0.5 * n_grid_y * dx]), ti.Vector([-1.0, 0.0]))
    update_body(dt)

result_dir = "./results"
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

init()

gui = ti.GUI("mpm two-way coupling", (512, 512 * n_grid_y // n_grid_x))
frame_cnt = 0
while gui.running and not gui.get_event(gui.ESCAPE):
    for _ in range(300):
        if _ % 50 == 0:
            update_jet()
        substep(dt / 50)
    # print(body_theta[None], body_w[None])
    gui.clear(0x000000)
    gui.circles(part_x.to_numpy() / denorm, radius=1.15, color=0x068587)
    bvert = body_vert.to_numpy().reshape(n_body_seg*n_body_seg, 2) / denorm
    gui.lines(begin=bvert[:-1], end=bvert[1:], radius=1.5)
    # gui.rect(
    #     np.array([bound * dx / (n_grid_x*dx), (n_grid_y - bound)*dx / (n_grid_y*dx)]),
    #     np.array([(n_grid_x - bound)*dx / (n_grid_x*dx), bound * dx / (n_grid_y*dx)]))
    pixels_img = gui.get_image()
    gui.show()
    video_manager.write_frame(pixels_img)
    frame_cnt += 1
    if frame_cnt >= 24 * 8:
        break

print()
print('Exporting .mp4 and .gif videos...')
video_manager.make_video(gif=True, mp4=True)
print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')
print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')