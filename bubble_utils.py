import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors

from generate_colormap import get_continuous_cmap
# pylint: disable=pointless-string-statement

import os
import pickle


# Set global font settings for Matplotlib
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],  # Use Computer Modern
    'text.usetex': True,  # Use LaTeX to render the text with Computer Modern
    'font.size': 14  # Set the default font size
})


def load_bubble_dataset(bubble_data_path, processed_data_path="bubble_data_processed.pkl"):
    """
    Loads bubble data from a specified file path, with the option to load from or save to a preprocessed file.

    This function reads bubble data from a file, processes it, and returns a DataFrame
    containing the bubble data. If a preprocessed file exists, it loads the data from that
    file instead of processing the raw data.

    Parameters
    ----------
    bubble_data_path : str
        The file path to the raw bubble data file.
    processed_data_path : str, optional
        The file path to the preprocessed bubble data file (default is "bubble_data_processed.pkl").

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the processed bubble data.
    """
    # Check if the processed data file exists
    if os.path.exists(processed_data_path):
        print(f"Loading processed data from {processed_data_path}...")
        with open(processed_data_path, 'rb') as file:
            bubble_data_df = pickle.load(file)
    else:
        print(f"Processing raw data from {bubble_data_path}...")
        bubble_records = []

        with open(bubble_data_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        i = 0
        while i < len(lines):
            metadata_line = lines[i].strip()
            i += 1

            if metadata_line == "":
                continue

            metadata_parts = metadata_line.split(",")
            num_bubbles = int(metadata_parts[0].strip())
            timestamp = metadata_parts[1].strip()
            dissolved_mass = (
                float(metadata_parts[2].strip()) if metadata_parts[2].strip() else None
            )

            for _ in range(num_bubbles):
                bubble_line = lines[i].strip()
                i += 1
                bubble_parts = bubble_line.split(",")
                bubble_records.append(
                    {
                        "datetime": timestamp,
                        "dissolved_mass": dissolved_mass,
                        "longitude": float(bubble_parts[0]),
                        "latitude": float(bubble_parts[1]),
                        "x_coordinate": float(bubble_parts[2]),
                        "y_coordinate": float(bubble_parts[3]),
                        "depth": -float(bubble_parts[4]),
                        "mass": float(bubble_parts[5]),
                        "num_bubbles": 1,
                        "size": float(bubble_parts[7]),
                        "density": float(bubble_parts[8]),
                        "vertical_velocity": float(bubble_parts[9]),
                        "gas_type": bubble_parts[10].strip(),
                        "hydro_grid_proc": int(bubble_parts[11]),
                        "bubble_proc": int(bubble_parts[12]),
                    }
                )

        # Convert list of dictionaries to DataFrame
        bubble_data_df = pd.DataFrame(bubble_records)
        bubble_data_df["datetime"] = pd.to_datetime(bubble_data_df["datetime"])

        # Save the processed data for future use
        with open(processed_data_path, 'wb') as file:
            pickle.dump(bubble_data_df, file)
        print(f"Processed data saved to {processed_data_path}.")

    return bubble_data_df


def compute_tetrahedron_matrix(vertex1, vertex2, vertex3, vertex4):
    """
    Computes the transformation matrix for a tetrahedron.

    Parameters
    ----------
    vertex1, vertex2, vertex3, vertex4 : numpy.ndarray
        Vertices of the tetrahedron.

    Returns
    -------
    np.ndarray
        The transformation matrix from orthogonal coordinates to tetrahedral coordinates.
    """
    mat = np.array([vertex2 - vertex1, vertex3 - vertex1, vertex4 - vertex1]).T
    return np.linalg.inv(mat)


def is_point_in_tetrahedron(vertex1, vertex2, vertex3, vertex4, point):
    """
    Determines if a point is inside a tetrahedron.

    Parameters
    ----------
    vertex1, vertex2, vertex3, vertex4 : np.ndarray
        Vertices of the tetrahedron.
    point : np.ndarray
        Point to check.

    Returns
    -------
    bool
        True if the point is inside the tetrahedron, False otherwise.
    """
    transformation_matrix = compute_tetrahedron_matrix(vertex1, vertex2, vertex3, vertex4)
    transformed_point = transformation_matrix.dot(point - vertex1)
    return np.all(transformed_point >= 0) and np.all(transformed_point <= 1) and np.sum(transformed_point) <= 1


def is_point_in_prism(point, prism_vertices):
    """
    Checks if a point is inside any of the tetrahedra forming a trapezoidal prism.

    Parameters
    ----------
    point : np.ndarray
        The point to check.
    prism_vertices : np.ndarray
        Vertices of the trapezoidal prism.

    Returns
    -------
    bool
        True if the point is inside the prism, False otherwise.
    """
    tetrahedra = [
        (prism_vertices[0], prism_vertices[1], prism_vertices[2], prism_vertices[5]),
        (prism_vertices[0], prism_vertices[2], prism_vertices[3], prism_vertices[7]),
        (prism_vertices[0], prism_vertices[2], prism_vertices[5], prism_vertices[7]),
        (prism_vertices[2], prism_vertices[5], prism_vertices[6], prism_vertices[7]),
        (prism_vertices[0], prism_vertices[5], prism_vertices[7], prism_vertices[4]),
    ]

    for tetrahedron in tetrahedra:
        if is_point_in_tetrahedron(*tetrahedron, point):
            return True

    return False

# Reference:
# The functions compute_tetrahedron_matrix, is_point_in_tetrahedron and is_point_in_prism 
# are adapted and modified from a solution by Dorian on Stack Overflow.
# https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not



def compute_trapezoidal_prisms(
    origin, theta_degrees, phi_degree, line_length, num_planes, prism_width_angle_deg, prism_height_angle_deg
):
    """
    Computes vertices of multiple trapezoidal prisms to simulate sonar beams, ensuring consistent prism size,
    parallel alignment, and proper handling of angles for sonar beam simulation.

    Parameters
    ----------
    origin : array_like
        The starting point (x, y, z) of the line along which the prisms are placed.
    theta_degrees : float
        The polar angle in degrees from the z-axis.
    phi_degree : float
        The azimuthal angle in degrees from the x-axis in the xy-plane.
    line_length : float
        The length of the line along which the prisms are placed.
    num_planes : int
        The number of planes defining the prisms along the line.
    prism_width_angle_deg : float
        The width angle of the prisms in degrees.
    prism_height_angle_deg : float
        The height angle of the prisms in degrees.

    Returns
    -------
    np.ndarray
        A 3D array (num_planes - 1, 8, 3) containing the vertices of the trapezoidal prisms for the beam.
    """
    
    # Convert angles from degrees to radians
    theta_radians = np.radians(theta_degrees)
    phi_radians = np.radians(phi_degree)
    width_angle = np.radians(prism_width_angle_deg)
    height_angle = np.radians(prism_height_angle_deg)
    
    # Function to compute the direction vector from theta and phi
    def get_direction_vector(theta, phi):
        return np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
    
    # Get the direction vector for the current beam
    direction_vector = get_direction_vector(theta_radians, phi_radians)
    direction_vector /= np.linalg.norm(direction_vector)  # Normalize
    
    # Define consistent global right and up vectors (independent of the direction vector)
    global_up = np.array([0, 0, 1])
    
    if np.allclose(direction_vector, global_up) or np.allclose(direction_vector, -global_up):
        global_right = np.array([0, 1, 0])
    else:
        global_right = np.cross(global_up, direction_vector)
        global_right /= np.linalg.norm(global_right)
    
    global_up = np.cross(direction_vector, global_right)
    global_up /= np.linalg.norm(global_up)

    # Compute distances along the direction vector
    t_values = np.linspace(0, line_length, num_planes)
    
    prisms = []

    for i in range(num_planes - 1):
        # Calculate near and far distances
        near_distance = t_values[i]
        far_distance = t_values[i + 1]

        # Compute centers of near and far planes
        near_center = origin + near_distance * direction_vector
        far_center = origin + far_distance * direction_vector

        # Compute widths and heights at near and far planes (consistent for all beams)
        near_width = 2 * near_distance * np.tan(width_angle / 2)
        near_height = 2 * near_distance * np.tan(height_angle / 2)

        far_width = 2 * far_distance * np.tan(width_angle / 2)
        far_height = 2 * far_distance * np.tan(height_angle / 2)

        # Define corners for near plane
        near_corners = [
            near_center - (near_width / 2) * global_right - (near_height / 2) * global_up,
            near_center + (near_width / 2) * global_right - (near_height / 2) * global_up,
            near_center + (near_width / 2) * global_right + (near_height / 2) * global_up,
            near_center - (near_width / 2) * global_right + (near_height / 2) * global_up,
        ]

        # Define corners for far plane
        far_corners = [
            far_center - (far_width / 2) * global_right - (far_height / 2) * global_up,
            far_center + (far_width / 2) * global_right - (far_height / 2) * global_up,
            far_center + (far_width / 2) * global_right + (far_height / 2) * global_up,
            far_center - (far_width / 2) * global_right + (far_height / 2) * global_up,
        ]

        # Combine corners to form prism
        prism_vertices = np.array(near_corners + far_corners)
        prisms.append(prism_vertices)

    return np.array(prisms)


def plot_bubbles_and_prisms(prisms_array, inside_bubble_points, outside_bubble_points, origin, bubble_counts, all_bubble_points, plot_hugin=False):
    """
    Plots trapezoidal prisms and bubbles, with different colors for bubbles inside and outside the prisms.

    This function visualizes multiple trapezoidal prisms and the bubbles within a dataset, differentiating 
    the bubbles inside and outside the prisms using different colors.

    Parameters
    ----------
    prisms_array : numpy.ndarray
        An array of shape (num_beams, num_planes-1, 8, 3) defining the vertices of the trapezoidal prisms.
    inside_bubble_points : numpy.ndarray
        Array of points (x, y, depth) that are inside the prisms.
    outside_bubble_points : numpy.ndarray
        Array of points (x, y, depth) that are outside the prisms.
    origin : tuple
        The origin point (x, y, z) for plotting the HUGIN Superior.
    bubble_counts : np.ndarray
        A 2D array with shape (num_beams, num_planes-1) containing the counts of bubbles within each prism.
    all_bubble_points : numpy.ndarray
        Array of all bubble points (x, y, depth) in the dataset.

    Returns
    -------
    None

    Notes
    -----
    The `get_continuous_cmap` function used in this function is adapted from the following GitHub repository:
    https://github.com/KerryHalupka/custom_colormap/blob/master/generate_colormap.py

    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(True, alpha=0.1)

    num_beams, num_depths = prisms_array.shape[:2]

    # Normalize bubble counts to range [0, 1] for colormap
    max_count = np.max(bubble_counts)

    hex_colors = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#FEFEFD']
    shifted_cmap = get_continuous_cmap(hex_colors, reverse=True)

    # Reference for get_continuous_cmap:
    # This function is adapted from the following GitHub repository:
    # https://github.com/KerryHalupka/custom_colormap/blob/master/generate_colormap.py

    norm = colors.Normalize(vmin=0, vmax=max_count)

    for beam_index in range(num_beams):
        for depth_index in range(num_depths):
            prism_vertices = prisms_array[beam_index, depth_index]
            color = shifted_cmap(norm(bubble_counts[beam_index, depth_index]))
            faces = [
                [prism_vertices[0], prism_vertices[1], prism_vertices[2], prism_vertices[3]],  # bottom face
                [prism_vertices[4], prism_vertices[5], prism_vertices[6], prism_vertices[7]],  # top face
                [prism_vertices[0], prism_vertices[1], prism_vertices[5], prism_vertices[4]],  # front face
                [prism_vertices[1], prism_vertices[2], prism_vertices[6], prism_vertices[5]],  # right face
                [prism_vertices[2], prism_vertices[3], prism_vertices[7], prism_vertices[6]],  # back face
                [prism_vertices[3], prism_vertices[0], prism_vertices[4], prism_vertices[7]],  # left face
            ]

            poly3d = [[tuple(point) for point in face] for face in faces]
            ax.add_collection3d(Poly3DCollection(poly3d, facecolors=color, linewidths=1, edgecolors=(0.45, 0.45, 0.45, 0.8), alpha=0.20))

    # Plot bubbles inside the prism in blue
    if len(inside_bubble_points) > 0:
        ax.scatter(*inside_bubble_points.T, color="dodgerblue", label="Inside Bubbles")

    # Plot bubbles outside the prism in red
    if len(outside_bubble_points) > 0:
        ax.scatter(*outside_bubble_points.T, color="orangered", label="Outside Bubbles")
    
    if plot_hugin:
        plot_hugin_superior(ax, origin)

    ax.set_title("Bubble Distribution Inside and Outside Trapezoidal Prisms", fontsize=24)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Depth")
    set_plot_limits(ax, prisms_array, all_bubble_points)

    ax.view_init(elev=37, azim=40)
    ax.legend(bbox_to_anchor=(0.8, -0.05), ncol=2)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=shifted_cmap, norm=norm), ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Number of Bubbles')
    plt.savefig('bubbles_and_beam.png', dpi=300)
    plt.show()


def plot_hugin_superior(ax, origin):
    """
    Plots the HUGIN Superior at the given origin on the provided axes.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D subplot axes to plot on.
    origin : tuple
        The origin point (x, y, z) for plotting the HUGIN Superior.
    """
    a, b, c = 0.075, 0.05, 2  # Semi-axis lengths for the ellipsoid
    phi, theta = np.meshgrid(np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 20))

    x = a * np.sin(theta) * np.cos(phi) + origin[0]
    y = b * np.sin(theta) * np.sin(phi) + origin[1]
    z = c * np.cos(theta) + origin[2] + c

    ax.plot_surface(x, y, z, color='darkorange', alpha=1.0)

    
def set_plot_limits(ax, prisms_array, all_bubble_points):
    """
    Sets the limits for the 3D plot based on the provided prism vertices and bubble points.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D subplot axes to set limits on.
    prisms_array : np.ndarray
        The vertices of the trapezoidal prisms.
    all_bubble_points : np.ndarray
        All bubble points in the dataset.
    """
    all_x = np.concatenate([prisms_array[:, :, :, 0].flatten(), all_bubble_points[:, 0]])
    all_y = np.concatenate([prisms_array[:, :, :, 1].flatten(), all_bubble_points[:, 1]])
    all_z = np.concatenate([prisms_array[:, :, :, 2].flatten(), all_bubble_points[:, 2]])

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    z_min, z_max = np.min(all_z), np.max(all_z)

    buffer_x = (x_max - x_min) * 0.1
    buffer_y = (y_max - y_min) * 0.1
    buffer_z = (z_max - z_min) * 0.1

    ax.set_xlim([x_min - buffer_x, x_max + buffer_x])
    ax.set_ylim([y_min - buffer_y, y_max + buffer_y])
    ax.set_zlim([z_min - buffer_z, z_max + buffer_z])


def get_bubble_counts_for_prisms(dataset, prisms_array, specific_time):
    """
    Calculate bubble counts for given prisms at a specific time and classify points.

    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset containing bubble data points with 'datetime', 'x_coordinate', 'y_coordinate', and 'depth' columns.
    prisms_array : numpy.ndarray
        A 3D array where each element is a prism defined by its vertices.
    specific_time : datetime
        The specific time to filter the dataset for processing.

    Returns
    -------
    result_df : pandas.DataFrame
        DataFrame containing rows from the dataset with additional 'beam' and 'depth_index' columns for points inside any prism.
    inside_bubble_points : numpy.ndarray
        Array of points (x, y, depth) that are inside the prisms.
    outside_bubble_points : numpy.ndarray
        Array of points (x, y, depth) that are outside the prisms.
    bubble_counts : numpy.ndarray
        2D array with counts of bubbles for each prism at different beams and depths.
    """
    filtered_dataset = dataset[dataset["datetime"] == specific_time]  # Filter dataset by specific time

    num_beams, num_depths = prisms_array.shape[:2]  # Get dimensions of prisms array
    bubble_counts = np.zeros((num_beams, num_depths), dtype=int)  # Initialize bubble counts array

    inside_bubble_points = []
    outside_bubble_points = []
    result = []

    for _, row in filtered_dataset.iterrows():
        point = (row["x_coordinate"], row["y_coordinate"], row["depth"])

        for beam_index in range(num_beams):
            for depth_index in range(1, num_depths):
                prism_vertices = prisms_array[beam_index, depth_index]
                if is_point_in_prism(point, prism_vertices):
                    bubble_counts[beam_index, depth_index] += 1
                    row["beam"] = beam_index
                    row["depth_index"] = depth_index
                    result.append(row)
                    inside_bubble_points.append(point)
                    break
            else:
                continue
            break
        else:
            outside_bubble_points.append(point)

    return pd.DataFrame(result), np.array(inside_bubble_points), np.array(outside_bubble_points), bubble_counts



def get_bubbles_from_beam(bubble_dataset, specific_timestamp, origin, theta_degrees):
    """
    Get bubbles inside and outside prisms for a specific timestamp and set of beams.

    Parameters
    ----------
    bubble_dataset : pd.DataFrame
        The dataset containing bubble data points.
    specific_timestamp : str
        The timestamp to filter the bubble data.
    origin : tuple
        The origin point (x, y, z) for the prisms.
    theta_degrees : list of float
        List of polar angles for the beams.

    Returns
    -------
    np.ndarray
        Array of points (x, y, depth) that are inside the prisms.
    np.ndarray
        2D array with counts of bubbles for each prism at different beams and depths.
    np.ndarray
        Array defining the vertices of the trapezoidal prisms.
    """
    phi_degree = 180
    line_length = 32
    num_planes = 8
    prism_width_angle_deg = 0.5
    prism_height_angle_deg = 1

    all_prisms = np.array([
        compute_trapezoidal_prisms(
            origin, theta_degree, phi_degree, line_length, num_planes, prism_width_angle_deg, prism_height_angle_deg
        ) for theta_degree in theta_degrees
    ])

    result_df, inside_bubble_points, outside_bubble_points, bubble_counts = get_bubble_counts_for_prisms(
        bubble_dataset, all_prisms, specific_timestamp)

    return inside_bubble_points, bubble_counts, all_prisms



if __name__ == "__main__":
    # Multi-beam tests:
    theta_degrees = [179, 180, 181]
    origin = (0, 0, -40)

    # theta_degrees = [269, 270, 271]
    # origin = (-20, 0, -69.3)

    # theta_degrees = [89, 90, 91]
    # origin = (-20, 0, -69.3)
    

    bubble_file_path = "1-plume.dat"
    bubble_dataset = load_bubble_dataset(bubble_file_path)

    specific_timestamp = "2019-12-31T13:09:58"
    inside_bubble_points, bubble_counts, all_prisms = get_bubbles_from_beam(
        bubble_dataset, specific_timestamp, origin, theta_degrees
    )

    if inside_bubble_points.size > 0:
        result_df = pd.DataFrame({
            'x': inside_bubble_points[:, 0],
            'y': inside_bubble_points[:, 1],
            'depth': inside_bubble_points[:, 2]
        })
    else:
        result_df = pd.DataFrame(columns=['x', 'y', 'depth'])

    result_df.to_csv('bubble_data_09_58.csv', index=False)


    all_bubble_points = bubble_dataset[bubble_dataset["datetime"] == specific_timestamp][["x_coordinate", "y_coordinate", "depth"]].values

    plot_bubbles_and_prisms(
        all_prisms, inside_bubble_points, [], origin, bubble_counts, all_bubble_points
    )
