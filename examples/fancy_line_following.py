

import time
import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
import pyroki_snippets as pks
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize_scalar


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


def generate_velocity_profile(path_points, max_vel=0.1, max_accel=0.5):
    """
    Generate trapezoidal velocity profile for a path.
    
    Args:
        path_points: List of 3D positions
        max_vel: Maximum velocity (m/s)
        max_accel: Maximum acceleration (m/s²)
    
    Returns:
        timestamps: Array of time values
        velocities: Array of velocity magnitudes
        accelerations: Array of acceleration magnitudes
    """
    # Calculate distances between consecutive points
    distances = []
    for i in range(1, len(path_points)):
        dist = np.linalg.norm(path_points[i] - path_points[i-1])
        distances.append(dist)
    
    # Calculate cumulative distance
    cumulative_dist = np.cumsum([0] + distances)
    total_distance = cumulative_dist[-1]
    
    # Calculate time for trapezoidal profile
    # Time to reach max velocity
    t_accel = max_vel / max_accel
    # Distance during acceleration
    d_accel = 0.5 * max_accel * t_accel**2
    
    if 2 * d_accel >= total_distance:
        # Triangular profile (never reach max velocity)
        t_accel = np.sqrt(total_distance / max_accel)
        t_constant = 0
        t_decel = t_accel
        peak_velocity = max_accel * t_accel
    else:
        # Trapezoidal profile
        d_constant = total_distance - 2 * d_accel
        t_constant = d_constant / max_vel
        t_decel = t_accel
        peak_velocity = max_vel
    
    total_time = t_accel + t_constant + t_decel
    
    # Generate time vector
    dt = 0.01  # 10ms resolution
    time_vector = np.arange(0, total_time + dt, dt)
    
    # Generate velocity and acceleration profiles
    velocities = np.zeros_like(time_vector)
    accelerations = np.zeros_like(time_vector)
    distances_at_time = np.zeros_like(time_vector)
    
    for i, t in enumerate(time_vector):
        if t <= t_accel:
            # Acceleration phase
            v = max_accel * t
            a = max_accel
            d = 0.5 * max_accel * t**2
        elif t <= t_accel + t_constant:
            # Constant velocity phase
            v = peak_velocity
            a = 0
            d = d_accel + peak_velocity * (t - t_accel)
        elif t <= total_time:
            # Deceleration phase
            t_decel_elapsed = t - t_accel - t_constant
            v = peak_velocity - max_accel * t_decel_elapsed
            a = -max_accel
            d = d_accel + peak_velocity * t_constant + peak_velocity * t_decel_elapsed - 0.5 * max_accel * t_decel_elapsed**2
        else:
            v = 0
            a = 0
            d = total_distance
        
        velocities[i] = v
        accelerations[i] = a
        distances_at_time[i] = d
    
    # Map distances back to path parameter (0 to 1)
    path_parameters = distances_at_time / total_distance
    path_parameters = np.clip(path_parameters, 0, 1)
    
    return time_vector, velocities, accelerations, path_parameters


def interpolate_path_with_timing(path_points, time_vector, path_parameters):
    """
    Interpolate path points according to velocity profile timing.
    
    Args:
        path_points: Original path waypoints
        time_vector: Time stamps from velocity profile
        path_parameters: Parameter values (0 to 1) along path
    
    Returns:
        timed_path: List of (time, position) tuples
    """
    # Create parameter values for original path points
    original_params = np.linspace(0, 1, len(path_points))
    
    # Interpolate each coordinate
    path_array = np.array(path_points)
    x_interp = interp1d(original_params, path_array[:, 0], kind='cubic', bounds_error=False, fill_value='extrapolate')
    y_interp = interp1d(original_params, path_array[:, 1], kind='cubic', bounds_error=False, fill_value='extrapolate')
    z_interp = interp1d(original_params, path_array[:, 2], kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    # Generate time-parameterized path
    timed_path = []
    for t, param in zip(time_vector, path_parameters):
        pos = np.array([
            x_interp(param),
            y_interp(param),
            z_interp(param)
        ])
        timed_path.append((t, pos))
    
    return timed_path


def smooth_joint_trajectory(joint_configs, timestamps, smoothing_method="cubic_spline"):
    """
    Smooth joint trajectory to ensure velocity and acceleration continuity.
    
    Args:
        joint_configs: List of joint configurations
        timestamps: Corresponding timestamps
        smoothing_method: "cubic_spline" or "minimum_jerk"
    
    Returns:
        smooth_trajectory: List of (time, joint_config, joint_velocity, joint_acceleration)
    """
    if len(joint_configs) < 2:
        return [(timestamps[0], joint_configs[0], np.zeros_like(joint_configs[0]), np.zeros_like(joint_configs[0]))]
    
    joint_array = np.array(joint_configs)
    n_joints = joint_array.shape[1]
    
    if smoothing_method == "cubic_spline":
        # Use cubic spline interpolation for smooth trajectories
        dt = 0.01  # 10ms resolution
        smooth_times = np.arange(timestamps[0], timestamps[-1] + dt, dt)
        
        smooth_trajectory = []
        
        for joint_idx in range(n_joints):
            # Create spline for this joint
            spline = CubicSpline(timestamps, joint_array[:, joint_idx], bc_type='natural')
            
            # Evaluate position, velocity, acceleration
            joint_positions = spline(smooth_times)
            joint_velocities = spline(smooth_times, 1)  # First derivative
            joint_accelerations = spline(smooth_times, 2)  # Second derivative
            
            if joint_idx == 0:
                # Initialize arrays
                all_positions = np.zeros((len(smooth_times), n_joints))
                all_velocities = np.zeros((len(smooth_times), n_joints))
                all_accelerations = np.zeros((len(smooth_times), n_joints))
            
            all_positions[:, joint_idx] = joint_positions
            all_velocities[:, joint_idx] = joint_velocities
            all_accelerations[:, joint_idx] = joint_accelerations
        
        # Combine into trajectory
        for i, t in enumerate(smooth_times):
            smooth_trajectory.append((
                t,
                all_positions[i],
                all_velocities[i],
                all_accelerations[i]
            ))
        
        return smooth_trajectory
    
    elif smoothing_method == "minimum_jerk":
        # Minimum jerk trajectory between waypoints
        smooth_trajectory = []
        
        for i in range(len(joint_configs) - 1):
            start_q = joint_configs[i]
            end_q = joint_configs[i + 1]
            start_time = timestamps[i]
            end_time = timestamps[i + 1]
            duration = end_time - start_time
            
            # Generate minimum jerk trajectory segment
            dt = 0.01
            segment_times = np.arange(0, duration + dt, dt)
            
            for t_seg in segment_times:
                # Minimum jerk polynomial: s(τ) = 10τ³ - 15τ⁴ + 6τ⁵
                tau = t_seg / duration
                s = 10*tau**3 - 15*tau**4 + 6*tau**5
                s_dot = (30*tau**2 - 60*tau**3 + 30*tau**4) / duration
                s_ddot = (60*tau - 180*tau**2 + 120*tau**3) / duration**2
                
                # Interpolate joint values
                q = start_q + s * (end_q - start_q)
                q_dot = s_dot * (end_q - start_q)
                q_ddot = s_ddot * (end_q - start_q)
                
                smooth_trajectory.append((
                    start_time + t_seg,
                    q,
                    q_dot,
                    q_ddot
                ))
        
        return smooth_trajectory
    
    else:
        raise ValueError(f"Unknown smoothing method: {smoothing_method}")


def analyze_trajectory_quality(original_path, joint_trajectory, robot, target_link_name):
    """
    Analyze trajectory quality by comparing target vs actual end-effector positions.
    
    Returns:
        tracking_errors: List of position errors
        max_error: Maximum tracking error
        rms_error: RMS tracking error
        actual_positions: Realized end-effector positions
    """
    actual_positions = []
    tracking_errors = []
    
    for t, q, q_dot, q_ddot in joint_trajectory:
        # Forward kinematics
        robot.set_q(q)
        pose = robot.forward_kinematics(target_link_name)
        actual_pos = pose.position
        actual_positions.append(actual_pos)
    
    # Find closest original path point for each trajectory point
    for actual_pos in actual_positions:
        min_error = float('inf')
        for target_pos in original_path:
            error = np.linalg.norm(actual_pos - target_pos)
            min_error = min(min_error, error)
        tracking_errors.append(min_error)
    
    max_error = np.max(tracking_errors)
    rms_error = np.sqrt(np.mean(np.array(tracking_errors)**2))
    
    return tracking_errors, max_error, rms_error, actual_positions


def main():
    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"

    robot = pk.Robot.from_urdf(urdf)

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Generate shape path
    shape_path = generate_shape_path("triangle", num_points_per_edge=25)
    
    # Generate velocity profile
    print("Generating velocity profile...")
    time_vector, velocities, accelerations, path_parameters = generate_velocity_profile(
        shape_path, max_vel=0.15, max_accel=0.3
    )
    
    # Create time-parameterized path
    timed_path = interpolate_path_with_timing(shape_path, time_vector, path_parameters)
    
    print(f"Generated path with {len(timed_path)} points over {time_vector[-1]:.2f} seconds")
    print(f"Max velocity: {np.max(velocities):.3f} m/s")
    print(f"Max acceleration: {np.max(np.abs(accelerations)):.3f} m/s²")

    # Fixed orientation for all poses (end-effector pointing down Z)
    fixed_wxyz = np.array([0, 0, 1, 0])

    # Solve IK for key waypoints (subsample for efficiency)
    step_size = max(1, len(timed_path) // 50)  # ~50 IK solutions max
    key_waypoints = timed_path[::step_size]
    
    joint_configs = []
    waypoint_times = []

    print("Solving IK for key waypoints...")
    for t, pos in key_waypoints:
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=pos,
            target_wxyz=fixed_wxyz,
        )
        if solution is None:
            print(f"IK failed at t={t:.2f}s, pos={pos}")
            continue
        joint_configs.append(solution)
        waypoint_times.append(t)

    print(f"Successfully solved IK for {len(joint_configs)} waypoints")

    # Smooth joint trajectory
    print("Smoothing joint trajectory...")
    smooth_trajectory = smooth_joint_trajectory(
        joint_configs, waypoint_times, smoothing_method="cubic_spline"
    )
    
    # Analyze trajectory quality
    print("Analyzing trajectory quality...")
    # tracking_errors, max_error, rms_error, actual_positions = analyze_trajectory_quality(
        # shape_path, smooth_trajectory, robot, target_link_name
    # )
    

    # Playback loop with smooth timing
    print("Playing back smooth trajectory...")
    start_time = time.time()
    
    for t, q, q_dot, q_ddot in smooth_trajectory:
        # Update robot visualization
        urdf_vis.update_cfg(q)
        
        # Maintain real-time playback
        elapsed = time.time() - start_time
        sleep_time = t - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # Optional: Print debug info periodically
        if int(t * 10) % 50 == 0:  # Every 5 seconds
            print(f"Time: {t:.2f}s, Joint velocities: {np.max(np.abs(q_dot)):.3f} rad/s")


if __name__ == "__main__":
    main()