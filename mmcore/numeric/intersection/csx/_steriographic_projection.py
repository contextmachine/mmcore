import numpy as np


def stereographic_projection(points):
    """
    Performs a stereographic projection of 3D points (all on the unit sphere)
    onto the plane z = 0 from the north pole.

    Parameters:
        points (np.ndarray): Array of shape (N, 3) with points on the unit sphere.

    Returns:
        np.ndarray: Array of shape (N, 2) with the projected 2D points.
    """
    # Separate the coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Compute the denominator for the projection
    denominator = 1 - z  # Ensure z != 1 to avoid division by zero

    # Compute the projected 2D coordinates
    projected_x = x / denominator
    projected_y = y / denominator

    # Stack the results into an (N, 2) array
    return np.stack((projected_x, projected_y), axis=-1)


# Example usage:
if __name__ == "__main__":
    # Create a grid of points on the unit sphere using spherical coordinates
    theta = np.linspace(0.01, np.pi - 0.01, 50)  # avoid the poles to prevent division by zero
    phi = np.linspace(0, 2 * np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    theta = theta.flatten()
    phi = phi.flatten()

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    points = np.stack((x, y, z), axis=-1)

    projected_points = stereographic_projection(points)
    print("Projected points:\n", projected_points)
