"""
This script generates and visualizes a lawnmower path with turns for a given domain.

The lawnmower path is a common strategy used in autonomous underwater vehicle (AUV) missions, 
land surveying, and agricultural applications to ensure complete coverage of an area. The script
calculates waypoints for the path, including handling turns with cloverleaf and half-circle maneuvers.

It then visualizes the path by plotting the domain and the path with waypoints using Matplotlib, 
including the ability to label the waypoints and customize the plot's appearance.

Key features include:
- Customizable path width and minimum turn radius
- Directional control for the path
- Optional LaTeX-style font settings for publication-quality plots
- Legends and labels for enhanced plot readability
"""

import matplotlib.pyplot as plt
import numpy as np

# Update plot parameters for nice formatting (comment out if not necessary)
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['Computer Modern Roman'],
#     'text.usetex': True,
#     'font.size': 14
# })


def make_lawnmower_path(x_data, y_data, width, min_turn_radius, direction='y'):
    """
    Generate a lawnmower path covering the specified area.

    Parameters
    ----------
    x_data : array-like
        x-coordinates of the area to cover.
    y_data : array-like
        y-coordinates of the area to cover.
    width : float
        Width of each pass.
    min_turn_radius : float
        Minimum turn radius.
    direction : str, optional
        Direction of the passes, either 'x' or 'y' (default is 'y').

    Returns
    -------
    waypoints : list of tuple
        List of waypoints as (x, y) coordinates.
    """
    # Determine if cloverleaf pattern is used
    use_cloverleaf = width < min_turn_radius

    # Calculate the buffer depending on the pattern used and direction
    if use_cloverleaf:
        if direction == 'x':
            buffer_x = min_turn_radius * 2
            buffer_y = min_turn_radius
        elif direction == 'y':
            buffer_x = min_turn_radius
            buffer_y = min_turn_radius * 2
    else:
        buffer_x = width/2
        buffer_y = width/2

    # Define the boundaries of the area after applying the buffer
    x_min, x_max = np.min(x_data) + buffer_x, np.max(x_data) - buffer_x
    y_min, y_max = np.min(y_data) + buffer_y, np.max(y_data) - buffer_y

    waypoints = []

    # Create the lawnmower path based on the selected direction
    if direction == 'x':
        num_passes = int((x_max - x_min) / width) + 1

        for i in range(num_passes):
            x_start = x_min + i * width
            if x_start > x_max:
                break

            # Alternate between bottom-to-top and top-to-bottom passes
            if i % 2 == 0:
                waypoints.extend([(x_start, y_min), (x_start, y_max)])
            else:
                waypoints.extend([(x_start, y_max), (x_start, y_min)])

    elif direction == 'y':
        num_passes = int((y_max - y_min) / width) + 1

        for i in range(num_passes):
            y_start = y_min + i * width
            if y_start > y_max:
                break

            # Alternate between left-to-right and right-to-left passes
            if i % 2 == 0:
                waypoints.extend([(x_min, y_start), (x_max, y_start)])
            else:
                waypoints.extend([(x_max, y_start), (x_min, y_start)])

    return waypoints


def generate_cloverleaf_points(center1, center2, radius, mode='up', direction='counterclockwise'):
    """
    Generate points for a cloverleaf turn.

    Parameters
    ----------
    center1, center2 : tuple of float
        Centers of the two arcs forming the cloverleaf turn.
    radius : float
        Radius of the arcs.
    mode : str, optional
        Orientation of the cloverleaf ('up', 'down', 'left', 'right'), by default 'up'.
    direction : str, optional
        Direction of the cloverleaf turn ('clockwise', 'counterclockwise'), by default 'counterclockwise'.

    Returns
    -------
    turn_points : list of tuple
        List of (x, y) points forming the cloverleaf turn.
    """
    # Generate arc points based on the mode and orientation
    if mode == 'up':
        theta = np.linspace(-np.pi/2, np.pi, 100)
        arc2_x = center2[0] + radius * np.sin(theta)
        arc2_y = center2[1] + radius * np.cos(theta)
        arc1_x = center1[0] + radius * np.sin(theta + 3 * np.pi / 2)
        arc1_y = center1[1] + radius * np.cos(theta + 3 * np.pi / 2)
    elif mode == 'down':
        theta = np.linspace(-np.pi/2, np.pi, 100)
        arc2_x = center2[0] + radius * np.sin(theta)
        arc2_y = center2[1] + radius * -np.cos(theta)
        arc1_x = center1[0] + radius * np.sin(theta + 3 * np.pi / 2)
        arc1_y = center1[1] + radius * -np.cos(theta + 3 * np.pi / 2)
    elif mode == 'left':
        theta = np.linspace(np.pi/2, -np.pi, 100)
        arc2_x = center2[0] + radius * np.cos(theta)
        arc2_y = center2[1] + radius * -np.sin(theta)
        arc1_x = center1[0] + radius * -np.cos(theta + 3 * np.pi / 2)
        arc1_y = center1[1] + radius * np.sin(theta + 3 * np.pi / 2)
    elif mode == 'right':
        theta = np.linspace(-np.pi/2, np.pi, 100)
        arc2_x = center2[0] + radius * -np.cos(theta)
        arc2_y = center2[1] + radius * np.sin(theta)
        arc1_x = center1[0] + radius * -np.cos(theta + 3 * np.pi / 2)
        arc1_y = center1[1] + radius * np.sin(-theta + 3 * np.pi / 2)
    else:
        raise ValueError("Mode must be one of 'up', 'down', 'left', or 'right'.")

    turn_points = []
    turn_points.extend(zip(arc2_x, arc2_y))
    turn_points.extend(zip(arc1_x, arc1_y))
    
    # Reverse the points if the turn direction is counterclockwise
    if direction == 'counterclockwise':
        turn_points.reverse()
    
    return turn_points


def add_half_circle_turn(x_center, y_center, radius, start_angle, end_angle, num_points, x_sign=1, y_sign=1):
    """
    Generate a half-circle turn.

    Parameters
    ----------
    x_center, y_center : float
        Center coordinates of the half-circle.
    radius : float
        Radius of the half-circle.
    start_angle, end_angle : float
        Start and end angles in radians.
    num_points : int
        Number of points to generate along the half-circle.
    x_sign, y_sign : int, optional
        Sign multipliers to control the direction of the arc, by default 1.

    Returns
    -------
    x_arc, y_arc : ndarray
        x and y coordinates of the arc points.
    """
    angles = np.linspace(start_angle, end_angle, num_points)
    x_arc = x_center + radius * x_sign * np.cos(angles)
    y_arc = y_center + radius * y_sign * np.sin(angles)
    return x_arc, y_arc


def add_turns_to_waypoints(waypoints, width, min_turn_radius, direction='y'):
    """
    Add turns to a set of waypoints for a lawnmower pattern.

    Parameters
    ----------
    waypoints : list of tuple
        List of (x, y) waypoints.
    width : float
        Width of each pass.
    min_turn_radius : float
        Minimum turn radius.
    direction : str, optional
        Direction of the turns ('x' or 'y'), by default 'y'.

    Returns
    -------
    waypoints_with_turns : list of tuple
        List of (x, y) waypoints with turns included.
    """
    waypoints_with_turns = []
    mid_point = width / 2
    use_cloverleaf = width < min_turn_radius
    mode_toggle = 1  # Used to alternate turn direction

    for i, waypoint in enumerate(waypoints):
        waypoints_with_turns.append(waypoint)

        # Add turns at the end of each pass except the last one
        if i % 2 != 0 and i < (len(waypoints) - 1):
            if direction == 'x':
                if use_cloverleaf:
                    # Generate cloverleaf turns when width < min_turn_radius
                    center1 = (waypoints[i][0] - min_turn_radius, waypoints[i][1])
                    center2 = (waypoints[i + 1][0] + min_turn_radius, waypoints[i + 1][1])
                    mode = 'up' if mode_toggle > 0 else 'down'
                    turn_points = generate_cloverleaf_points(center1, center2, min_turn_radius, mode=mode)
                else:
                    # Generate half-circle turns when cloverleaf is not used
                    x_sign = 1
                    y_sign = 1 if mode_toggle > 0 else -1
                    x_center = waypoints[i][0]
                    y_max = waypoints[i][1]
                    x_arc, y_arc = add_half_circle_turn(x_center + mid_point, y_max, mid_point, np.pi, 0, 10, x_sign, y_sign)
                    turn_points = list(zip(x_arc, y_arc))
                    
            elif direction == 'y':
                if use_cloverleaf:
                    # Generate cloverleaf turns when width < min_turn_radius
                    center1 = (waypoints[i][0], waypoints[i][1] - min_turn_radius)
                    center2 = (waypoints[i + 1][0], waypoints[i + 1][1] + min_turn_radius)
                    mode = 'left' if mode_toggle > 0 else 'right'
                    turn_points = generate_cloverleaf_points(center1, center2, min_turn_radius, mode=mode)
                else:
                    # Generate half-circle turns when cloverleaf is not used
                    x_sign = 1 if mode_toggle > 0 else -1
                    y_sign = 1
                    y_center = waypoints[i][1]
                    x_max = waypoints[i][0]
                    x_arc, y_arc = add_half_circle_turn(x_max, y_center + mid_point, mid_point, -np.pi / 2, np.pi / 2, 10, x_sign, y_sign)
                    turn_points = list(zip(x_arc, y_arc))

            # Add generated turn points to the waypoints with turns
            waypoints_with_turns.extend(turn_points)
            mode_toggle *= -1  # Toggle the direction for the next turn

    return waypoints_with_turns


def scatter_plot_points_and_path(points, x_min, x_max, y_min, y_max):
    """
    Create a scatter plot of given points with a colormap indicating progression,
    plot lines connecting the points to visualize the path, and shade the domain
    in the background.

    Parameters
    ----------
    points : list of tuple
        List of (x, y) coordinates to plot.
    x_min, x_max, y_min, y_max : float
        The boundaries of the domain.
    """
    x, y = zip(*points)
    colors = np.linspace(0, 1, len(points))

    plt.figure(figsize=(10, 6))
    
    # Plot the domain background
    plt.fill_between([x_min, x_max], y_min, y_max, color='lightgray', alpha=0.5, label='Domain')
    
    # Plot the lines connecting the points
    plt.plot(x, y, color='black', linestyle='-', linewidth=1, alpha=0.7, label='AUV Path')
    
    # Plot the scatter points
    scatter = plt.scatter(x, y, c=colors, cmap='jet', label='Waypoints')
    
    plt.colorbar(scatter, label='Path progression')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Lawnmower Path with Turns')
    plt.grid(True)
    plt.axis('equal')
    
    # Set the limits of the plot to match the domain
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    
    # Place the legend centered below the plot in two columns
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.show()


def generate_lawnmower_waypoints(x_data, y_data, width, min_turn_radius, siglay, direction):
    """
    Generate waypoints for a lawnmower path with turns.

    Parameters
    ----------
    x_data, y_data : array-like
        x and y coordinates of the area to cover.
    width : float
        Width of each pass.
    min_turn_radius : float
        Minimum turn radius.
    siglay : float
        Depth or layer of the waypoints.

    Returns
    -------
    waypoints_with_turns : ndarray
        Array of waypoints with x, y, and z coordinates.
    x_coords, y_coords, z_coords : ndarray
        x, y, and z coordinates of the waypoints.
    """
    # Generate the basic lawnmower path
    waypoints = make_lawnmower_path(x_data, y_data, width, min_turn_radius, direction=direction)
    
    # Add turns to the path based on the direction and turn radius
    waypoints_with_turns = add_turns_to_waypoints(waypoints, width, min_turn_radius, direction=direction)
    
    # Convert waypoints to numpy array if not already
    waypoints_with_turns = np.array(waypoints_with_turns)

    # Extract x and y coordinates
    x_coords = waypoints_with_turns[:, 0]
    y_coords = waypoints_with_turns[:, 1]

    # Create z coordinates, filled with the siglay value
    z_coords = np.full_like(x_coords, siglay)

    # Stack x, y, and z coordinates together
    waypoints_with_turns = np.column_stack((waypoints_with_turns, z_coords))

    return waypoints_with_turns, x_coords, y_coords, z_coords

def remove_consecutive_duplicate_wps(arr, tol=1e-8):
    """
    Remove consecutive duplicate rows from a NumPy array within a given tolerance.
    
    Parameters:
    - arr: numpy.ndarray of shape (n, m)
    - tol: float, tolerance for considering elements as equal
    
    Returns:
    - new_arr: numpy.ndarray with consecutive duplicates removed
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if arr.ndim != 2:
        raise ValueError("Input array must be 2-dimensional.")
    if len(arr) == 0:
        return arr  # Return empty array if input is empty
    
    # Compute differences between consecutive rows
    diffs = np.abs(arr[1:] - arr[:-1])
    
    # Check if differences are within tolerance for all elements in the row
    within_tol = np.all(diffs <= tol, axis=1)
    
    # Initialize the mask with True values
    mask = np.ones(len(arr), dtype=bool)
    
    # Update the mask for rows to keep
    mask[1:] = ~within_tol
    
    # Apply the mask to the array to get the result
    new_arr = arr[mask]
    
    return new_arr

if __name__ == '__main__':
    # Dummy domain for generating example waypoints
    x_data = np.linspace(0, 500, 3000)
    y_data = np.linspace(0, 500, 3000)

    # Generate waypoints for the lawnmower path with specified parameters
    waypoints_with_turns, x_coords, y_coords, z_coords = generate_lawnmower_waypoints(
        x_data, y_data, width=50, min_turn_radius=50, siglay=1, direction='y'
    )

    # Domain boundaries for the plot
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_data), np.max(y_data)

    # Visualize the path with the domain background and labeled waypoints
    scatter_plot_points_and_path(waypoints_with_turns, x_min, x_max, y_min, y_max)
