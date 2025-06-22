

import time
import numpy as np
import jax.numpy as jnp
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

import pyroki_snippets as pks

# --- NEW HELPER FUNCTION ---
# This function converts a path of points into the segment format
# required by viser.scene.add_line_segments.
def path_to_segments(path_points: np.ndarray) -> np.ndarray:
    """
    Converts a path array of shape (N, 3) into a line segment array of
    shape (N-1, 2, 3).
    """
    if len(path_points) < 2:
        return np.zeros((0, 2, 3))
    
    # Segments are defined by [start_point, end_point]
    # Start points are all points except the last one.
    starts = path_points[:-1]
    # End points are all points except the first one.
    ends = path_points[1:]
    
    # Stack them along a new axis to get the (N-1, 2, 3) shape.
    return np.stack([starts, ends], axis=1)


def generate_shape_path(shape="trapezoid", num_points_per_edge=25):
    """
    Generate a Cartesian path for an arbitrary shape.
    Returns a numpy array of positions.
    """
    if shape == "trapezoid":
        z = 0.5
        corners = np.array([
            [0.5, -0.2, z], [0.7, -0.2, z], [0.8, 0.2, z], [0.4, 0.2, z]
        ])
    elif shape == "triangle":
        z = 0.5
        corners = np.array([
            [0.5, 0.0, z], [0.7, -0.3, z], [0.7, 0.3, z]
        ])
    elif shape == "cube":
        c, s = 0.6, 0.1
        corners = np.array([
            [c - s, -s, 0.5 - s], [c + s, -s, 0.5 - s], [c + s, s, 0.5 - s], [c - s, s, 0.5 - s],
            [c - s, -s, 0.5 + s], [c + s, -s, 0.5 + s], [c + s, s, 0.5 + s], [c - s, s, 0.5 + s]
        ])
    else:
        raise ValueError("Shape not supported")

    path_list = []
    for i in range(len(corners)):
        start, end = corners[i], corners[(i + 1) % len(corners)]
        for alpha in np.linspace(0, 1, num_points_per_edge, endpoint=False):
            path_list.append(start * (1 - alpha) + end * alpha)
            
    # Also connect the last point back to the first for a closed loop
    path_list.append(corners[0])
    return np.array(path_list)


def main():
    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"
    robot = pk.Robot.from_urdf(urdf)

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    shape_path_points = generate_shape_path("cube", num_points_per_edge=25)

    # --- MODIFIED VISUALIZATION: Use add_line_segments ---
    target_path_segments = path_to_segments(shape_path_points)
    server.scene.add_line_segments(
        "/target_path",
        points=target_path_segments,
        colors=(0, 128, 255),  # Blue
        line_width=2.0,
    )

    fixed_wxyz = np.array([0, 0, 1, 0])
    joint_trajectory = []

    print("Solving IK for path points...")
    for pos in shape_path_points:
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=pos,
            target_wxyz=fixed_wxyz,
        )
        joint_trajectory.append(solution)

    # Calculate and visualize the actual path
    solved_q = jnp.array(joint_trajectory)
    all_link_poses = robot.forward_kinematics(solved_q)
    target_link_index = robot.links.names.index(target_link_name)
    actual_positions = np.array(all_link_poses[:, target_link_index, 4:])

    # --- MODIFIED VISUALIZATION: Use add_line_segments for the actual path ---
    actual_path_segments = path_to_segments(actual_positions)
    server.scene.add_line_segments(
        "/actual_path",
        points=actual_path_segments,
        colors=(255, 0, 0),  # Red
        line_width=4.0,
    )

    # Interactive playback loop
    print("IK solving complete. Starting interactive playback...")
    num_timesteps = len(joint_trajectory)
    slider = server.gui.add_slider(
        "Timestep", min=0, max=num_timesteps - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("Playing", initial_value=True)

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % num_timesteps
        
        q = joint_trajectory[slider.value]
        urdf_vis.update_cfg(q)
        
        time.sleep(1.0 / 30.0)

if __name__ == "__main__":
    main()
