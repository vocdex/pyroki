import time
import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
import pyroki_snippets as pks


def generate_shape_path(shape="trapezoid", num_points_per_edge=25):
    """
    Generate a Cartesian path for an arbitrary shape.
    Currently supports: trapezoid, triangle, cube corners.
    Returns list of positions (np.array).
    """
    if shape == "trapezoid":
        # Define trapezoid corners in XY plane at fixed Z
        z = 0.5
        corners = np.array([
            [0.5, -0.2, z],
            [0.7, -0.2, z],
            [0.8, 0.2, z],
            [0.4, 0.2, z],
        ])
    elif shape == "triangle":
        z = 0.5
        corners = np.array([
            [0.5, 0.0, z],
            [0.7, -0.3, z],
            [0.7, 0.3, z],
        ])
    elif shape == "cube":
        # 8 corners of a cube centered at 0.6,0,0.5 side length 0.2
        c = 0.6
        s = 0.1
        corners = np.array([
            [c - s, -s, 0.5 - s],
            [c + s, -s, 0.5 - s],
            [c + s, s, 0.5 - s],
            [c - s, s, 0.5 - s],
            [c - s, -s, 0.5 + s],
            [c + s, -s, 0.5 + s],
            [c + s, s, 0.5 + s],
            [c - s, s, 0.5 + s],
        ])
    else:
        raise ValueError("Shape not supported")

    # Interpolate linearly between corners to get smooth path
    path = []
    for i in range(len(corners)):
        start = corners[i]
        end = corners[(i + 1) % len(corners)]
        for alpha in np.linspace(0, 1, num_points_per_edge, endpoint=False):
            pos = start * (1 - alpha) + end * alpha
            path.append(pos)
    return path


def time_parametrize_path(path_points, velocity=0.1):
    """
    Assign timestamps to path points assuming constant velocity (m/s).
    Returns list of (t, position) tuples.
    """
    times = [0]
    for i in range(1, len(path_points)):
        dist = np.linalg.norm(path_points[i] - path_points[i - 1])
        dt = dist / velocity
        times.append(times[-1] + dt)
    return list(zip(times, path_points))


def main():
    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"

    robot = pk.Robot.from_urdf(urdf)

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")


    # Generate shape path and time-parametrize it
    shape_path = generate_shape_path("cube", num_points_per_edge=25)
    timed_path = time_parametrize_path(shape_path, velocity=0.1)

    # Fixed orientation for all poses (end-effector pointing down Z)
    fixed_wxyz = np.array([0, 0, 1, 0])

    joint_trajectory = []
    timestamps = []

    print("Solving IK for path points...")
    for t, pos in timed_path:
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=pos,
            target_wxyz=fixed_wxyz,
            # initial_q=q_guess,  # Use previous solution as initial guess. Not currently supported in pks.solve_ik
        )
        if solution is None:
            print(f"IK failed at t={t:.2f}s, pos={pos}")
            continue
        joint_trajectory.append(solution)
        timestamps.append(t)
        # q_guess = solution  # Update initial guess

    # Playback loop
    print("Playing back trajectory...")
    t_prev = 0
    for q, t in zip(joint_trajectory, timestamps):
        urdf_vis.update_cfg(q)
        if t_prev > 0:
            time.sleep(float(t - t_prev))
        t_prev = t


def qks_forward_fk(robot, q, link_name):
    """
    Compute forward kinematics position of the link given joint angles q.
    Simple wrapper for pyroki FK.
    """
    robot.set_q(q)
    pose = robot.forward_kinematics(link_name)
    return pose.position


if __name__ == "__main__":
    main()
