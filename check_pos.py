import numpy as np

def get_camera_position_from_extrinsic(extrinsic_matrix):
    # Extract the rotation matrix (R) and translation vector (t) from the extrinsic matrix
    R = extrinsic_matrix[0:3, 0:3]
    t = extrinsic_matrix[0:3, 3]

    # Compute the inverse of the rotation matrix
    R_inv = np.linalg.inv(R)

    # Compute the camera position in the world coordinate system
    camera_position = -np.dot(R_inv, t)

    return camera_position

# Example extrinsic matrix (replace with your actual extrinsic matrix)
extrinsic_matrix = np.array([[-4.16752896e-03, 0.00000000e+00, 9.99991316e-01, -1.37981884e-15],
                  [-1.94035192e-01, 9.80994236e-01, -8.08654302e-04, -7.90433426e-14],
                  [-9.80985717e-01, -1.94036877e-01, -4.08832188e-03, 2.12048274e+03]])

# Get the camera position
camera_position = get_camera_position_from_extrinsic(extrinsic_matrix)
print("Camera Position (x, y, z):", camera_position)