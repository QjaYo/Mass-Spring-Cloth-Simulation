import taichi as ti

# ti.init(arch=ti.cpu)
ti.init(arch=ti.gpu)

n = 256 # number of mass points along one dimension
quad_size = 1.0 / n
dt = 4e-2 / n
substeps = int(1 / 60 // dt)
reset_time = 4.0
paused = False

gravity = ti.Vector([0, -9.8, 0])

spring_Y = ti.field(dtype=float, shape=())
dashpot_damping = ti.field(dtype=float, shape=())
drag_damping = ti.field(dtype=float, shape=())
friction = ti.field(dtype=float, shape=())
restitution = ti.field(dtype=float, shape=())

spring_Y[None] = 3e4
dashpot_damping[None] = 1e4
drag_damping[None] = 1
friction[None] = 0.01
restitution[None] = 0.0

ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 0, 0]

x = ti.Vector.field(3, dtype=float, shape=(n, n)) # position
v = ti.Vector.field(3, dtype=float, shape=(n, n)) # velocity

vertices = ti.Vector.field(3, dtype=float, shape=(n * n))
indices = ti.field(dtype=int, shape=((n - 1) * (n - 1) * 6))
colors = ti.Vector.field(3, dtype=float, shape=(n * n))

bending_springs = False

spring_offsets = []
if bending_springs == False:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([i, j]))
else:
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                spring_offsets.append(ti.Vector([i, j]))

@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1

    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 
            0.6,
            j * quad_size - 0.5 + random_offset[1]
        ]
        v[i, j] = [0, 0, 0]

@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # First triangle of the square (clockwise order)
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # Second triangle of the square (clockwise order)
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

@ti.kernel
def initialize_mesh_colors():
    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = ti.Vector([59, 179, 96]) / 255.0
        else:
            colors[i * n + j] = ti.Vector([14, 65, 29]) / 255.0

@ti.kernel
def substep():
    for i in ti.grouped(x):
        v[i] += gravity * dt
    
    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset

            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]

                d = x_ij.normalized() # direction from j to i
                current_dist = x_ij.norm() # current distance between i and j
                original_dist = quad_size * float(spring_offset).norm() # original distance between i and j

                force += -spring_Y[None] * (current_dist / original_dist - 1) * d # spring force
                force += -v_ij.dot(d) * (dashpot_damping[None] * quad_size) * d # dashpot damping force

        v[i] += force * dt # mass = 1.0

    for i in ti.grouped(x):
        v[i] *= ti.exp(-drag_damping[None] * dt) # drag damping independent of dt

        # Collision with the ball
        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            normal = offset_to_center.normalized()
            v_normal = v[i].dot(normal)

            v[i] -= (1 + restitution[None]) * min(v_normal, 0) * normal # restitution
            v[i] -= friction[None] * (v[i] - v_normal * normal) # friction
            x[i] = ball_center[0] + ball_radius * normal # position projection

        x[i] += dt * v[i]

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]

if __name__ == "__main__":
    initialize_mass_points()
    initialize_mesh_indices()
    initialize_mesh_colors()

    window = ti.ui.Window("Taichi Mass-Spring Cloth Simulation", (1024, 1024), vsync=True)

    canvas = window.get_canvas()
    canvas.set_background_color((0.1, 0.1, 0.1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    current_t = 0.0
    
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
            if window.event.key == ti.ui.SPACE:
                paused = not paused

        gui = window.get_gui()
        with gui.sub_window("Settings", x=0.05, y=0.05, width=0.4, height=0.15):

            spring_Y[None] = gui.slider_float("spring_Y", spring_Y[None], 0, 43500)
            dashpot_damping[None] = gui.slider_float("dashpot_damping", dashpot_damping[None], 0, 200000)
            drag_damping[None] = gui.slider_float("drag_damping", drag_damping[None], 0, 5)
            friction[None] = gui.slider_float("friction", friction[None], 0, 0.025)
            restitution[None] = gui.slider_float("restitution", restitution[None], 0, 1)

        if not paused:
            if current_t > reset_time:
                initialize_mass_points()
                current_t = 0.0
        
            for i in range(substeps):
                substep()
                current_t += dt
            update_vertices()

        camera.position(0.0, 0.0, 3)
        camera.lookat(0.0, 0.0, 0.0)
        scene.set_camera(camera)

        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.mesh(vertices=vertices, indices=indices, per_vertex_color=colors, two_sided=True)

        scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.5, 0.5))
        canvas.scene(scene)
        window.show()

    window.destroy()