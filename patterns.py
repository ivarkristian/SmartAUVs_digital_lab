import numpy as np

def bowtie(center, a=10, steps=30):
    """
    Generates a bowtie pattern centered at a specified point.

    Parameters
    ----------
    center : array-like
        The center of the bowtie.
    a : float, optional
        The amplitude of the bowtie pattern. Defaults to 10.
    steps : int, optional
        Number of points in the pattern. Defaults to 30.

    Returns
    -------
    numpy.ndarray
        Coordinates of the bowtie pattern.
    """
    t = np.linspace(0, 2 * np.pi, steps)
    x = a * np.sin(t)
    y = a * np.sin(t) * np.cos(t)
    z = np.zeros(steps)
    return np.column_stack((x, y, z)) + center

def cross(center, length=10):
    """
    Generates a cross pattern centered at a specified point.

    Parameters
    ----------
    center : array-like
        The center of the cross.
    length : float, optional
        The length of each line in the cross. Defaults to 10.

    Returns
    -------
    numpy.ndarray
        Coordinates of the cross pattern.
    """
    lines = [
        [[-length/2, 0, 0], [length/2, 0, 0]],
        [[0, -length/2, 0], [0, length/2, 0]]
    ]
    return np.array([line + center for line in lines])

def crisscross(center, length=10):
    """
    Generates a crisscross pattern centered at a specified point.

    Parameters
    ----------
    center : array-like
        The center of the crisscross.
    length : float, optional
        The length of each line in the crisscross. Defaults to 10.

    Returns
    -------
    numpy.ndarray
        Coordinates of the crisscross pattern.
    """
    lines = [
        [[-length/2, 0, 0], [length/2, 0, 0]],
        [[0, -length/2, 0], [0, length/2, 0]],
        [[-length/2, -length/2, 0], [length/2, length/2, 0]],
        [[-length/2, length/2, 0], [length/2, -length/2, 0]]
    ]
    return np.array([line + center for line in lines])

def drifting_circle(center, radius=10, drift=1, steps=30):
    """
    Generates a drifting circle pattern centered at a specified point.

    Parameters
    ----------
    center : array-like
        The center of the drifting circle.
    radius : float, optional
        The radius of the circle. Defaults to 10.
    drift : float, optional
        The drift along the z-axis. Defaults to 1.
    steps : int, optional
        Number of points in the pattern. Defaults to 30.

    Returns
    -------
    numpy.ndarray
        Coordinates of the drifting circle pattern.
    """
    t = np.linspace(0, 2 * np.pi, steps)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = drift * t / (2 * np.pi)
    return np.column_stack((x, y, z)) + center

def square(center, length=10):
    """
    Generates a square pattern centered at a specified point.

    Parameters
    ----------
    center : array-like
        The center of the square.
    length : float, optional
        The side length of the square. Defaults to 10.

    Returns
    -------
    numpy.ndarray
        Coordinates of the square pattern.
    """
    half = length / 2
    lines = [
        [[-half, -half, 0], [half, -half, 0]],
        [[half, -half, 0], [half, half, 0]],
        [[half, half, 0], [-half, half, 0]],
        [[-half, half, 0], [-half, -half, 0]]
    ]
    return np.array([line + center for line in lines])

def leaf_clover(center, radius=10, steps=30):
    """
    Generates a leaf clover pattern centered at a specified point.

    Parameters
    ----------
    center : array-like
        The center of the leaf clover.
    radius : float, optional
        The radius of each leaf. Defaults to 10.
    steps : int, optional
        Number of points in each leaf. Defaults to 30.

    Returns
    -------
    numpy.ndarray
        Coordinates of the leaf clover pattern.
    """
    patterns = []
    for i in range(4):
        angle = np.pi/2 * i
        x = radius * np.cos(np.linspace(angle, angle + np.pi, steps))
        y = radius * np.sin(np.linspace(angle, angle + np.pi, steps))
        z = np.zeros(steps)
        patterns.append(np.column_stack((x, y, z)) + center)
    return np.vstack(patterns)

def spiral(center, b=1, steps=30):
    """
    Generates a spiral pattern centered at a specified point.

    Parameters
    ----------
    center : array-like
        The center of the spiral.
    b : float, optional
        The coefficient controlling the spiral tightness. Defaults to 1.
    steps : int, optional
        Number of points in the spiral. Defaults to 30.

    Returns
    -------
    numpy.ndarray
        Coordinates of the spiral pattern.
    """
    t = np.linspace(0, 4 * np.pi, steps)
    r = b * t
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = np.zeros(steps)
    return np.column_stack((x, y, z)) + center
