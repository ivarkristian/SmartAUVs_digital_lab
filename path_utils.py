import numpy as np

def rotate_points(coordinates, angle_deg):
    """
    Rotates a set of 2D coordinates by a given angle in degrees.

    Parameters:
    - coordinates: numpy array of shape (n, 2), where each row is a point [x, y].
    - angle_deg: float, the angle by which to rotate the points, in degrees.

    Returns:
    - rotated_coords: numpy array of shape (n, 2), the rotated coordinates.
    """
    # Convert angle from degrees to radians
    angle_rad = np.deg2rad(angle_deg)

    # Create the rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])

    # Rotate the coordinates
    rotated_coords = coordinates @ rotation_matrix.T

    return rotated_coords