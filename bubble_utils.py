import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors

from generate_colormap import get_continuous_cmap
# pylint: disable=pointless-string-statement



def load_bubble_dataset(bubble_data_path):
    """
    Loads bubble data from a specified file path.

    This function reads bubble data from a file, processes it, and returns a DataFrame
    containing the bubble data.

    Parameters
    ----------
    bubble_data_path : str
        The file path to the bubble data file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the processed bubble data.
    """
    bubble_records = []

    with open(bubble_data_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        # Read metadata line
        metadata_line = lines[i].strip()
        i += 1

        if metadata_line == "":
            continue

        # Parse metadata
        metadata_parts = metadata_line.split(",")
        num_bubbles = int(metadata_parts[0].strip())
        timestamp = metadata_parts[1].strip()
        dissolved_mass = (
            float(metadata_parts[2].strip()) if metadata_parts[2].strip() else None
        )

        # Read bubble data lines
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
                    "depth": -float(bubble_parts[4]),  # Depth is stored as negative value
                    "mass": float(bubble_parts[5]),
                    "num_bubbles": 1,  # Each line represents one bubble
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
    transformation_matrix : numpy.ndarray
        The transformation matrix from orthogonal coordinates to tetrahedral coordinates.
    """
    v1 = vertex2 - vertex1
    v2 = vertex3 - vertex1
    v3 = vertex4 - vertex1
    mat = np.array((v1, v2, v3)).T
    transformation_matrix = np.linalg.inv(mat)
    return transformation_matrix

def is_point_in_tetrahedron(vertex1, vertex2, vertex3, vertex4, point):
    """
    Determines if a point is inside a tetrahedron.

    Parameters
    ----------
    vertex1, vertex2, vertex3, vertex4 : numpy.ndarray
        Vertices of the tetrahedron.
    point : numpy.ndarray
        Point to check.

    Returns
    -------
    bool
        True if the point is inside the tetrahedron, False otherwise.
    """
    transformation_matrix = compute_tetrahedron_matrix(vertex1, vertex2, vertex3, vertex4)
    transformed_point = transformation_matrix.dot(point - vertex1)
    return (np.all(transformed_point >= 0) and np.all(transformed_point <= 1) and np.sum(transformed_point) <= 1)

def is_point_in_prism(point, prism_vertices):
    """
    Checks if a point is inside any of the tetrahedra forming a trapezoidal prism.

    Parameters
    ----------
    point : numpy.ndarray
        The point to check.
    prism_vertices : numpy.ndarray
        Vertices of the trapezoidal prism.

    Returns
    -------
    bool
        True if the point is inside the prism, False otherwise.
    """
    # Define the tetrahedra forming the trapezoidal prism
    tetrahedra = [
        (prism_vertices[0], prism_vertices[1], prism_vertices[2], prism_vertices[5]),  # (a, b, c, f)
        (prism_vertices[0], prism_vertices[2], prism_vertices[3], prism_vertices[7]),  # (a, c, d, h)
        (prism_vertices[0], prism_vertices[2], prism_vertices[5], prism_vertices[7]),  # (a, c, f, h)
        (prism_vertices[2], prism_vertices[5], prism_vertices[6], prism_vertices[7]),  # (c, f, g, h)
        (prism_vertices[0], prism_vertices[5], prism_vertices[7], prism_vertices[4]),  # (a, f, h, e)
    ]

    point = np.array(point, dtype='d')

    # Check each tetrahedron to see if the point is inside any of them
    for tetrahedron in tetrahedra:
        vertex1, vertex2, vertex3, vertex4 = tetrahedron
        if is_point_in_tetrahedron(vertex1, vertex2, vertex3, vertex4, point):
            return True

    return False

# Reference:
# This functions compute_tetrahedron_matrix, is_point_in_tetrahedron and is_point_in_prism 
# are adapted and modified from a solution by Dorian on Stack Overflow.
# https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not





def compute_trapezoidal_prisms(origin, direction_theta_deg, direction_phi_deg, line_length, num_planes, prism_angle_width_deg, prism_angle_height_deg):
    """
    Computes vertices of multiple trapezoidal prisms along a specified direction.

    Parameters
    ----------
    origin : array_like
        The starting point (x, y, z) of the line along which the prisms are placed.
    direction_theta_deg : float
        The polar angle in degrees from the z-axis.
    direction_phi_deg : float
        The azimuthal angle in degrees from the x-axis in the xy-plane.
    line_length : float
        The length of the line along which the prisms are placed.
    num_planes : int
        The number of planes defining the prisms along the line.
    prism_angle_width_deg : float
        The width angle of the prisms in degrees.
    prism_angle_height_deg : float
        The height angle of the prisms in degrees.

    Returns
    -------
    prisms_array : numpy.ndarray
        An array of shape (num_planes-1, 8, 3) containing the vertices of the trapezoidal prisms.
    """
    # Convert angles to radians
    phi = np.radians(direction_phi_deg)
    theta = np.radians(direction_theta_deg)
    angle_width = np.radians(prism_angle_width_deg)
    angle_height = np.radians(prism_angle_height_deg)
    
    # Direction vector from spherical coordinates
    vx = np.sin(theta) * np.cos(phi)
    vy = np.sin(theta) * np.sin(phi)
    vz = np.cos(theta)
    direction_vector = np.array([vx, vy, vz])

    # Calculate distance values and orthogonal vectors
    t_values = np.linspace(0, line_length, num_planes)
    distances = t_values

    # Define orthogonal vectors a and b
    if np.isclose(vx, 0) and np.isclose(vy, 0):
        orthogonal_vector_a = np.array([1, 0, 0], dtype=float)
    else:
        orthogonal_vector_a = np.array([1, 0, -vx/vz], dtype=float) if vz != 0 else np.array([0, 1, -vy/vx], dtype=float)
    orthogonal_vector_a /= np.linalg.norm(orthogonal_vector_a)
    orthogonal_vector_b = np.cross(direction_vector, orthogonal_vector_a)
    orthogonal_vector_b /= np.linalg.norm(orthogonal_vector_b)

    prisms = []

    # Compute vertices for each pair of planes and create prisms
    for i in range(1, len(distances)):
        dw1 = np.tan(angle_width) * distances[i-1]
        dh1 = np.tan(angle_height) * distances[i-1]
        center1 = np.array(origin) + t_values[i-1] * direction_vector
        vertices1 = np.array([
            center1 - dh1 * orthogonal_vector_a - dw1 * orthogonal_vector_b,  # Bottom left
            center1 - dh1 * orthogonal_vector_a + dw1 * orthogonal_vector_b,  # Bottom right
            center1 + dh1 * orthogonal_vector_a + dw1 * orthogonal_vector_b,  # Top right
            center1 + dh1 * orthogonal_vector_a - dw1 * orthogonal_vector_b   # Top left
        ])

        dw2 = np.tan(angle_width) * distances[i]
        dh2 = np.tan(angle_height) * distances[i]
        center2 = np.array(origin) + t_values[i] * direction_vector
        vertices2 = np.array([
            center2 - dh2 * orthogonal_vector_a - dw2 * orthogonal_vector_b,  # Bottom left
            center2 - dh2 * orthogonal_vector_a + dw2 * orthogonal_vector_b,  # Bottom right
            center2 + dh2 * orthogonal_vector_a + dw2 * orthogonal_vector_b,  # Top right
            center2 + dh2 * orthogonal_vector_a - dw2 * orthogonal_vector_b   # Top left
        ])

        prism_vertices = np.vstack([vertices1, vertices2])
        prisms.append(prism_vertices)

    return np.array(prisms)




def plot_bubbles_and_prisms(prisms_array, inside_bubble_points, outside_bubble_points, origin, bubble_counts, all_bubble_points):
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
    fig = plt.figure(figsize=(12, 10))
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
        ax.scatter(
            inside_bubble_points[:, 0],
            inside_bubble_points[:, 1],
            inside_bubble_points[:, 2],
            color="dodgerblue",
            label="Inside Bubbles",
        )

    # Plot bubbles outside the prism in red
    if len(outside_bubble_points) > 0:
        ax.scatter(
            outside_bubble_points[:, 0],
            outside_bubble_points[:, 1],
            outside_bubble_points[:, 2],
            color="orangered",
            label="Outside Bubbles",
        )

    # Plot HUGIN Superior
    a = 0.075  # Semi-axis along x
    b = 0.05  # Semi-axis along y
    c = 2  # Semi-axis along z

    phi = np.linspace(0, 2 * np.pi, 20)
    theta = np.linspace(0, np.pi, 20)
    phi, theta = np.meshgrid(phi, theta)

    center_x = origin[0]
    center_y = origin[1]
    center_z = origin[2] + c

    x = a * np.sin(theta) * np.cos(phi) + center_x
    y = b * np.sin(theta) * np.sin(phi) + center_y
    z = c * np.cos(theta) + center_z

    ax.plot_surface(x, y, z, color='darkorange', alpha=1.0)

    # Set plot title, labels and limits
    ax.set_title("Bubble Distribution Inside and Outside Trapezoidal Prisms")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Depth")

    # Determine plot limits based on prism vertices and bubble points
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

    ax.legend(bbox_to_anchor=(0.8, -0.05), ncol=2)

    # Add colorbar with adjusted midpoint
    sm = plt.cm.ScalarMappable(cmap=shifted_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Number of Bubbles')

    plt.show()

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

    for _, row in filtered_dataset.iterrows():  # Iterate over each row in the filtered dataset
        point = (row["x_coordinate"], row["y_coordinate"], row["depth"])

        inside_prism = False
        for beam_index in range(num_beams):
            for depth_index in range(1, num_depths):  # Start from 1 to avoid the surface level
                prism_vertices = prisms_array[beam_index, depth_index]
                
                if is_point_in_prism(point, prism_vertices):  # Check if the point is inside the prism
                    bubble_counts[beam_index, depth_index] += 1
                    
                    row["beam"] = beam_index
                    row["depth_index"] = depth_index
                    result.append(row)

                    inside_bubble_points.append(point)
                    inside_prism = True
                    break
            if inside_prism:
                break
        if not inside_prism:
            outside_bubble_points.append(point)

    # Convert results to DataFrame and points lists to numpy arrays
    return pd.DataFrame(result), np.array(inside_bubble_points), np.array(outside_bubble_points), bubble_counts





if __name__ == "__main__":
    # Define parameters for the prisms
    theta_degrees = [179.8, 180, 180.2]
    origin = (0, 0, -40) 
    phi_degree = 0
    line_length = 32
    num_planes = 8
    prism_width_angle_deg = 1
    prism_height_angle_deg = 0.1

    # Initialize array to accumulate all prisms from different angles
    all_prisms = np.empty((len(theta_degrees), num_planes - 1, 8, 3), dtype=np.float64)
    for i, theta_degree in enumerate(theta_degrees):
        prisms = compute_trapezoidal_prisms(
            origin, theta_degree, phi_degree, line_length, num_planes, prism_width_angle_deg, prism_height_angle_deg
        )
        all_prisms[i, :, :, :] = prisms

    # Load bubble data
    bubble_file_path = "bubble/1-plume.dat"
    bubble_dataset = load_bubble_dataset(bubble_file_path)

    # Define the timestamp at which the bubbles should be checked
    specific_timestamp = "2019-12-31T13:00:01"

    # Get bubble counts and classification of bubbles inside and outside the prisms
    result_df, inside_bubble_points, outside_bubble_points, bubble_counts = get_bubble_counts_for_prisms(
        bubble_dataset, all_prisms, specific_timestamp
    )
    
    print(result_df, '\n\n')
    print(bubble_counts.T)

    # Extract all bubble points for plotting
    all_bubble_points = bubble_dataset[bubble_dataset["datetime"] == specific_timestamp][["x_coordinate", "y_coordinate", "depth"]].values

    # Plot the bubbles and prisms
    plot_bubbles_and_prisms(
        all_prisms, inside_bubble_points, outside_bubble_points, origin, bubble_counts, all_bubble_points
    )



    # result_df_to_save_ll = result_df[['lon', 'lat', 'depth', 'size']]
    # result_df_to_save_ll.to_csv(f"bubble_ll_{timestamp.replace('-', '_').replace(':', '_')}.csv", index=False)

    # result_df_to_save_xy = result_df[['x', 'y', 'depth', 'size']]

    # x_shift = 923324.3
    # y_shift = 6614355.5

    # result_df_to_save_xy.loc[:, 'x'] += x_shift
    # result_df_to_save_xy.loc[:, 'y'] += y_shift

    # result_df_to_save_xy.to_csv(f"bubble_xy_{timestamp.replace('-', '_').replace(':', '_')}.csv", index=False)

