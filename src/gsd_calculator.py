import numpy as np

def calculate_gsd(altitude, sensor_size, focal_length, image_height):
    """
    Calculate Ground Sample Distance (GSD) in meters per pixel.
    Formula: GSD = (altitude * sensor_size) / (focal_length * image_height)
    All units: altitude in m, sensor_size and focal_length in mm, image_height in pixels.
    """
    if not all([altitude, sensor_size, focal_length, image_height]):
        raise ValueError("All parameters must be provided for GSD calculation.")
    return (altitude * sensor_size) / (focal_length * image_height)

def pixel_to_real_distance(pixel_dist, gsd):
    """
    Convert pixel distance to real-world distance.
    """
    return pixel_dist * gsd

def euclidean_distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))