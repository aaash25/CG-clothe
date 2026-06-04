import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

GRID_SIZE = 16
GRID_WIDTH = 1.5
GRID_HEIGHT = 2.0
MASS = 0.1
GRAVITY = ti.Vector([0, -9.8, 0])
DT = 2e-3
MAX_VELOCITY = 10.0
SUBSTEPS = 5
MAX_STRETCH = 1.5

spring_k = ti.field(dtype=ti.f32, shape=())
damping_k = ti.field(dtype=ti.f32, shape=())
bending_k = ti.field(dtype=ti.f32, shape=())
spring_k[None] = 150.0
damping_k[None] = 5.0
bending_k[None] = 30.0

positions = ti.Vector.field(3, dtype=ti.f32, shape=(GRID_SIZE, GRID_SIZE))
velocities = ti.Vector.field(3, dtype=ti.f32, shape=(GRID_SIZE, GRID_SIZE))
accumulated_forces = ti.Vector.field(3, dtype=ti.f32, shape=(GRID_SIZE, GRID_SIZE))
old_positions = ti.Vector.field(3, dtype=ti.f32, shape=(GRID_SIZE, GRID_SIZE))

structural_springs_a = ti.Vector.field(2, dtype=ti.i32, shape=(2 * GRID_SIZE * (GRID_SIZE - 1)))
structural_springs_b = ti.Vector.field(2, dtype=ti.i32, shape=(2 * GRID_SIZE * (GRID_SIZE - 1)))
structural_springs_rest_length = ti.field(dtype=ti.f32, shape=(2 * GRID_SIZE * (GRID_SIZE - 1)))
num_structural = ti.field(ti.i32, shape=())

shear_springs_a = ti.Vector.field(2, dtype=ti.i32, shape=(2 * (GRID_SIZE - 1) * (GRID_SIZE - 1)))
shear_springs_b = ti.Vector.field(2, dtype=ti.i32, shape=(2 * (GRID_SIZE - 1) * (GRID_SIZE - 1)))
shear_springs_rest_length = ti.field(dtype=ti.f32, shape=(2 * (GRID_SIZE - 1) * (GRID_SIZE - 1)))
num_shear = ti.field(ti.i32, shape=())

bending_springs_a = ti.Vector.field(2, dtype=ti.i32, shape=(2 * GRID_SIZE * (GRID_SIZE - 2)))
bending_springs_b = ti.Vector.field(2, dtype=ti.i32, shape=(2 * GRID_SIZE * (GRID_SIZE - 2)))
bending_springs_rest_length = ti.field(dtype=ti.f32, shape=(2 * GRID_SIZE * (GRID_SIZE - 2)))
num_bending = ti.field(ti.i32, shape=())

current_method = ti.field(ti.i32, shape=())
is_paused = ti.field(ti.i32, shape=())

UNIT_WIDTH = GRID_WIDTH / (GRID_SIZE - 1)
UNIT_HEIGHT = GRID_HEIGHT / (GRID_SIZE - 1)
DIAGONAL_REST = ti.sqrt(UNIT_WIDTH * UNIT_WIDTH + UNIT_HEIGHT * UNIT_HEIGHT)

@ti.kernel
def init_positions():
    for i, j in positions:
        x = (j / (GRID_SIZE - 1)) * GRID_WIDTH - GRID_WIDTH / 2
        y = 1.0
        z = -(i / (GRID_SIZE - 1)) * GRID_HEIGHT
        positions[i, j] = ti.Vector([x, y, z])
        velocities[i, j] = ti.Vector([0, 0, 0])
        accumulated_forces[i, j] = ti.Vector([0, 0, 0])
        old_positions[i, j] = ti.Vector([x, y, z])

@ti.kernel
def init_structural_springs():
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE - 1):
            idx = i * (GRID_SIZE - 1) + j
            structural_springs_a[idx] = ti.Vector([i, j])
            structural_springs_b[idx] = ti.Vector([i, j + 1])
            structural_springs_rest_length[idx] = UNIT_WIDTH

    offset = GRID_SIZE * (GRID_SIZE - 1)
    for i in range(GRID_SIZE - 1):
        for j in range(GRID_SIZE):
            idx = offset + i * GRID_SIZE + j
            structural_springs_a[idx] = ti.Vector([i, j])
            structural_springs_b[idx] = ti.Vector([i + 1, j])
            structural_springs_rest_length[idx] = UNIT_HEIGHT
    num_structural[None] = 2 * GRID_SIZE * (GRID_SIZE - 1)

@ti.kernel
def init_shear_springs():
    n = (GRID_SIZE - 1) * (GRID_SIZE - 1)
    for i in range(GRID_SIZE - 1):
        for j in range(GRID_SIZE - 1):
            idx = i * (GRID_SIZE - 1) + j
            shear_springs_a[idx] = ti.Vector([i, j])
            shear_springs_b[idx] = ti.Vector([i + 1, j + 1])
            shear_springs_rest_length[idx] = DIAGONAL_REST

            idx2 = n + i * (GRID_SIZE - 1) + j
            shear_springs_a[idx2] = ti.Vector([i, j + 1])
            shear_springs_b[idx2] = ti.Vector([i + 1, j])
            shear_springs_rest_length[idx2] = DIAGONAL_REST
    num_shear[None] = 2 * (GRID_SIZE - 1) * (GRID_SIZE - 1)

@ti.kernel
def init_bending_springs():
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE - 2):
            idx = i * (GRID_SIZE - 2) + j
            bending_springs_a[idx] = ti.Vector([i, j])
            bending_springs_b[idx] = ti.Vector([i, j + 2])
            bending_springs_rest_length[idx] = 2.0 * UNIT_WIDTH

    offset = GRID_SIZE * (GRID_SIZE - 2)
    for i in range(GRID_SIZE - 2):
        for j in range(GRID_SIZE):
            idx = offset + i * GRID_SIZE + j
            bending_springs_a[idx] = ti.Vector([i, j])
            bending_springs_b[idx] = ti.Vector([i + 2, j])
            bending_springs_rest_length[idx] = 2.0 * UNIT_HEIGHT
    num_bending[None] = 2 * GRID_SIZE * (GRID_SIZE - 2)

@ti.kernel
def clear_forces():
    for i, j in accumulated_forces:
        accumulated_forces[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def add_gravity_and_damping():
    for i, j in positions:
        accumulated_forces[i, j] += MASS * GRAVITY - damping_k[None] * velocities[i, j]

@ti.kernel
def compute_structural_forces():
    for idx in range(num_structural[None]):
        ia, ja = structural_springs_a[idx][0], structural_springs_a[idx][1]
        ib, jb = structural_springs_b[idx][0], structural_springs_b[idx][1]
        rest_len = structural_springs_rest_length[idx]
        
        dx = positions[ib, jb] - positions[ia, ja]
        length = dx.norm()
        
        if length > 0:
            max_len = rest_len * MAX_STRETCH
            effective_length = ti.min(length, max_len)
            force = spring_k[None] * dx.normalized() * (effective_length - rest_len)
            
            accumulated_forces[ia, ja] += force
            accumulated_forces[ib, jb] -= force

@ti.kernel
def compute_shear_forces():
    for idx in range(num_shear[None]):
        ia, ja = shear_springs_a[idx][0], shear_springs_a[idx][1]
        ib, jb = shear_springs_b[idx][0], shear_springs_b[idx][1]
        rest_len = shear_springs_rest_length[idx]
        
        dx = positions[ib, jb] - positions[ia, ja]
        length = dx.norm()
        
        if length > 0:
            max_len = rest_len * MAX_STRETCH
            effective_length = ti.min(length, max_len)
            force = spring_k[None] * 0.5 * dx.normalized() * (effective_length - rest_len)
            
            accumulated_forces[ia, ja] += force
            accumulated_forces[ib, jb] -= force

@ti.kernel
def compute_bending_forces():
    for idx in range(num_bending[None]):
        ia, ja = bending_springs_a[idx][0], bending_springs_a[idx][1]
        ib, jb = bending_springs_b[idx][0], bending_springs_b[idx][1]
        rest_len = bending_springs_rest_length[idx]
        
        dx = positions[ib, jb] - positions[ia, ja]
        length = dx.norm()
        
        if length > 0:
            max_len = rest_len * MAX_STRETCH
            effective_length = ti.min(length, max_len)
            force = bending_k[None] * dx.normalized() * (effective_length - rest_len)
            
            accumulated_forces[ia, ja] += force
            accumulated_forces[ib, jb] -= force

@ti.func
def clamp_velocity(v):
    speed = v.norm()
    result = v
    if speed > MAX_VELOCITY:
        result = v * (MAX_VELOCITY / speed)
    return result

@ti.kernel
def apply_stretch_constraints():
    for idx in range(num_structural[None]):
        ia, ja = structural_springs_a[idx][0], structural_springs_a[idx][1]
        ib, jb = structural_springs_b[idx][0], structural_springs_b[idx][1]
        rest_len = structural_springs_rest_length[idx]
        max_len = rest_len * MAX_STRETCH

        dx = positions[ib, jb] - positions[ia, ja]
        length = dx.norm()
        if length > max_len and length > 0.0:
            correction = dx * (length - max_len) / length
            movable_a = ti.cast(ia > 0, ti.i32)
            movable_b = ti.cast(ib > 0, ti.i32)
            movable_count = movable_a + movable_b
            weight = 1.0 / ti.cast(ti.max(movable_count, 1), ti.f32)
            if ia > 0:
                positions[ia, ja] += correction * weight
            if ib > 0:
                positions[ib, jb] -= correction * weight

@ti.kernel
def step_explicit():
    for i, j in positions:
        if i == 0:
            continue
        
        acceleration = accumulated_forces[i, j] / MASS
        
        positions[i, j] += velocities[i, j] * DT
        velocities[i, j] += acceleration * DT
        velocities[i, j] = clamp_velocity(velocities[i, j])

@ti.kernel
def step_semi_implicit():
    for i, j in positions:
        if i == 0:
            continue
        
        acceleration = accumulated_forces[i, j] / MASS
        
        velocities[i, j] += acceleration * DT
        velocities[i, j] = clamp_velocity(velocities[i, j])
        positions[i, j] += velocities[i, j] * DT

@ti.kernel
def step_implicit_iter():
    for i, j in positions:
        old_positions[i, j] = positions[i, j]
    
    for i, j in positions:
        if i == 0:
            continue
        
        acceleration = accumulated_forces[i, j] / MASS
        
        velocities[i, j] += acceleration * DT
        positions[i, j] = old_positions[i, j] + velocities[i, j] * DT
        velocities[i, j] = clamp_velocity(velocities[i, j])

@ti.kernel
def reset_simulation():
    for i, j in positions:
        x = (j / (GRID_SIZE - 1)) * GRID_WIDTH - GRID_WIDTH / 2
        y = 1.0
        z = -(i / (GRID_SIZE - 1)) * GRID_HEIGHT
        positions[i, j] = ti.Vector([x, y, z])
        velocities[i, j] = ti.Vector([0, 0, 0])
        accumulated_forces[i, j] = ti.Vector([0, 0, 0])
        old_positions[i, j] = ti.Vector([x, y, z])

def compute_all_forces():
    clear_forces()
    compute_structural_forces()
    compute_shear_forces()
    compute_bending_forces()
    add_gravity_and_damping()

def initialize():
    init_positions()
    init_structural_springs()
    init_shear_springs()
    init_bending_springs()
    current_method[None] = 1
    is_paused[None] = 0

def main():
    initialize()
    
    window = ti.ui.Window("Cloth Simulation", (1024, 768), vsync=True)
    canvas = window.get_canvas()
    
    camera = ti.ui.Camera()
    camera.position(0.0, 2.5, 1.0)
    camera.lookat(0.0, 0.8, -1.0)
    camera.fov(55)
    
    method_names = ['Explicit Euler', 'Semi-Implicit', 'Implicit Euler']
    
    all_line_indices = []
    
    # 水平线
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE - 1):
            all_line_indices.append(i * GRID_SIZE + j)
            all_line_indices.append(i * GRID_SIZE + j + 1)
    
    # 垂直线
    for j in range(GRID_SIZE):
        for i in range(GRID_SIZE - 1):
            all_line_indices.append(i * GRID_SIZE + j)
            all_line_indices.append((i + 1) * GRID_SIZE + j)
    
    # 对角线 - 修正：统一对角线方向，与弹簧定义一致
    for i in range(GRID_SIZE - 1):
        for j in range(GRID_SIZE - 1):
            # 主对角线 \ (i,j) -> (i+1,j+1)
            all_line_indices.append(i * GRID_SIZE + j)
            all_line_indices.append((i + 1) * GRID_SIZE + (j + 1))
            # 反对角线 / (i,j+1) -> (i+1,j)
            all_line_indices.append(i * GRID_SIZE + (j + 1))
            all_line_indices.append((i + 1) * GRID_SIZE + j)
    
    all_line_indices = np.array(all_line_indices, dtype=np.int32)
    
    while window.running:
        camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.RMB)
        
        scene = window.get_scene()
        scene.set_camera(camera)
        
        scene.point_light(pos=(5.0, 5.0, 5.0), color=(1.0, 1.0, 1.0))
        scene.ambient_light((0.3, 0.3, 0.3))
        
        positions_np = positions.to_numpy().reshape(-1, 3)
        scene.lines(positions_np, indices=all_line_indices, color=(0.8, 0.6, 0.4), width=1.2)
        
        canvas.scene(scene)
        
        window.GUI.begin("Control Panel", 0.05, 0.05, 0.2, 0.35)
        
        window.GUI.text(f"Method: {method_names[current_method[None]]}")
        
        if window.GUI.button("Explicit"):
            current_method[None] = 0
        if window.GUI.button("Semi-Implicit"):
            current_method[None] = 1
        if window.GUI.button("Implicit"):
            current_method[None] = 2
        
        damping_k[None] = window.GUI.slider_float("Damping", damping_k[None], 0.0, 30.0)
        
        spring_k[None] = window.GUI.slider_float("Stiffness", spring_k[None], 50.0, 500.0)
        
        bending_k[None] = window.GUI.slider_float("Bending", bending_k[None], 0.0, 100.0)
        
        if window.GUI.button("Pause" if is_paused[None] == 0 else "Resume"):
            is_paused[None] = 1 - is_paused[None]
        
        if window.GUI.button("Reset"):
            reset_simulation()
        
        window.GUI.end()
        
        if is_paused[None] == 0:
            for _ in range(SUBSTEPS):
                compute_all_forces()
                if current_method[None] == 0:
                    step_explicit()
                elif current_method[None] == 1:
                    step_semi_implicit()
                elif current_method[None] == 2:
                    step_implicit_iter()
                for _ in range(5):
                    apply_stretch_constraints()
        
        window.show()

if __name__ == "__main__":
    main()