import pandas as pd
import ast
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import math
np.set_printoptions(precision=4, suppress=True)

def read_txt_to_dataframe(file_path):
    # Read the text file into a DataFrame
    df = pd.read_csv(file_path, sep=',')
    
    return df

def read_csv_to_dataframe(file_path):
    # Define a conversion function to interpret string representation of lists
    def convert_to_list(value):
        return ast.literal_eval(value)
    
    def convert_to_array_of_arrays(value):
        # Remove the word 'array' from the string
        value = re.sub(r'array\(', '', value)
        value = re.sub(r'\)', '', value)
        
        # Use ast.literal_eval to safely evaluate the string
        list_of_arrays = ast.literal_eval(value)
        
        # Convert each element to a numpy array
        return [np.array(arr) for arr in list_of_arrays]
    
    # Read the csv file into a DataFrame, applying the conversion function to relevant columns
    df = pd.read_csv(file_path, converters={
        'Crater Indices': convert_to_list,
        'Centre points 2D coord': convert_to_array_of_arrays
    })

    return df

def get_intrinsics(fov, im_width, im_height):
    fov = fov*math.pi/180
    fx = im_width/(2*math.tan(fov/2)) # Conversion from fov to focal length
    fy = im_height/(2*math.tan(fov/2)) # Conversion from fov to focal length
    cx = im_width/2
    cy = im_height/2
    return (np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]]))

def get_projection(camera_matrix):
    extrinsics = [[-4.16752896e-03, 0.00000000e+00, 9.99991316e-01, -1.37981884e-15],
                  [-1.94035192e-01, 9.80994236e-01, -8.08654302e-04, -7.90433426e-14],
                  [-9.80985717e-01, -1.94036877e-01, -4.08832188e-03, 2.12048274e+03]]
    
    return np.dot(camera_matrix, extrinsics)

def get_extrinsic_matrix(rvec, tvec):
    # Convert rvec to rotation matrix
    rotation_matrix, _ = cv.Rodrigues(rvec)
    # Concatenate rotation matrix and translation vector to form the extrinsic matrix
    extrinsic_matrix = np.hstack((rotation_matrix, tvec))
    return extrinsic_matrix

def get_camera_position_from_extrinsic(extrinsic_matrix):
    # Extract the rotation matrix (R) and translation vector (t) from the extrinsic matrix
    R = extrinsic_matrix[0:3, 0:3]
    t = extrinsic_matrix[0:3, 3]

    # Compute the inverse of the rotation matrix
    R_inv = np.linalg.inv(R)

    # Compute the camera position in the world coordinate system
    camera_position = -np.dot(R_inv, t)

    return camera_position

def get_intrinsics_ref(fov, image_width, image_height):
    fov = fov*np.pi/180
    fx = image_width/(2*np.tan(fov/2)) # Conversion from fov to focal length
    fy = image_height/(2*np.tan(fov/2)) # Conversion from fov to focal length
    cx = image_width/2
    cy = image_height/2
    print(np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]]))


# Image dimensions and FOV
image_width = 1024
image_height = 1024
fov = 30  # Field of view in degrees

# Calculate the focal length
focal_length = image_width / (2 * np.tan(np.radians(fov) / 2))

# Create the initial intrinsic matrix
camera_matrix = get_intrinsics(fov, image_width, image_height)

camera_matrix = [[886.81001348, 0., 512.],
                 [0., 886.81001348, 512.],
                 [0., 0., 1.]]

camera_matrix = np.array(camera_matrix)

projection_matrix = get_projection(camera_matrix)

robbins = read_txt_to_dataframe('ORBIT/data/robbins_navigation_dataset_christians_all.txt')
t_inst = read_csv_to_dataframe('ORBIT/output/testing_instance.csv')

# print(np.vstack(t_inst['Centre points 2D coord'][0]))

w_points_list = []
i_points_list = []

for i in range(3):

    i_points_list.append(np.vstack(t_inst['Centre points 2D coord'][i]))
    indices = t_inst['Crater Indices'][i]

    x_vals = robbins.iloc[indices][' X']
    y_vals = robbins.iloc[indices][' Y']
    z_vals = robbins.iloc[indices][' Z']

    w_points = robbins.iloc[indices][[' X', ' Y', ' Z']].to_numpy()
    w_points_list.append(w_points)

# print(w_points_list[0])
# print(i_points_list)


# Manual projection
all_projections = []
for j in range(3):
    projections = []
    for i in range(len(w_points_list[j])):
        first_point = w_points_list[j][i]
        object_point = np.array([first_point[0], first_point[1], first_point[2], 1])
        projection = np.dot(projection_matrix, object_point)
        projection = np.array([projection[0]/projection[2], projection[1]/projection[2]])
        projections.append(projection.tolist())

    all_projections.append(np.array(projections))

# print(w_points_list[0][0])
# print(all_projections[0][0])
# print(i_points_list[0][0])

# Graph projections on 2d grid
GRAPH = False
if GRAPH:
    projections = np.array(projections)
    fig1, ax1 = plt.subplots()
    scatter = ax1.scatter(projections[:, 0], projections[:, 1])

    data_points = i_points_list[0]
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(data_points[:, 0], data_points[:, 1])

    # Event handler to display coordinates
    def on_click1(event):
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            print(f"Clicked coordinates: x={x}, y={y}")
            # Optionally, you can display the coordinates on the plot
            ax1.annotate(f"({x:.2f}, {y:.2f})", (x, y), textcoords="offset points", xytext=(10, 10), ha='center', color='red')
            fig1.canvas.draw()

    def on_click2(event):
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            print(f"Clicked coordinates: x={x}, y={y}")
            # Optionally, you can display the coordinates on the plot
            ax2.annotate(f"({x:.2f}, {y:.2f})", (x, y), textcoords="offset points", xytext=(10, 10), ha='center', color='red')
            fig2.canvas.draw()

    # Connect the event handler to the figure
    fig1.canvas.mpl_connect('button_press_event', on_click1)
    fig2.canvas.mpl_connect('button_press_event', on_click2)

    plt.show()

# Camera calibration

# print(all_projections[0])
# print(i_points_list[0])

object_points = [obj.reshape(-1, 1, 3).astype(np.float32) for obj in w_points_list]
image_points = [img.reshape(-1, 1, 2).astype(np.float32) for img in i_points_list]
image_points_projected = [img.reshape(-1, 1, 2).astype(np.float32) for img in all_projections]



flags = (cv.CALIB_FIX_K1 | cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3 |
         cv.CALIB_FIX_K4 | cv.CALIB_FIX_K5 | cv.CALIB_FIX_K6 |
         cv.CALIB_FIX_TANGENT_DIST | cv.CALIB_USE_INTRINSIC_GUESS)

ret, camera_matrix, dist_coeffs, rvecs, tvecs, stdDeviationIntrinsics, stdDeviationExtrinsics, perViewErrors = cv.calibrateCameraExtended(
    object_points, image_points, (1024, 1024), camera_matrix, None, flags=flags)


# [print(tvec, "\n") for tvec in tvecs]

for i in range(len(rvecs)):
    a = get_extrinsic_matrix(rvecs[i], tvecs[i])
    pos = get_camera_position_from_extrinsic(a)
    pos = pos/1000
    print(pos)

exit()

# extra_point = [-0.3371, 0.4923, 2120493.3015]
ref_point = np.array([[2080.16328486, 411.45184895, 8.66921601],
             [2078.67469492, 418.92982014, 8.8552222],
             [2077.15938335, 426.40240594,9.04111456]])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points on a 3D graph
ax.scatter(w_points_list[0][:, 0], w_points_list[0][:, 1], w_points_list[0][:, 2], color='blue')
for i in ref_point:
    ax.scatter(i[0], i[1], i[2], color='red')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot of X, Y, and Z')

# Set equal scaling for all axes
x_limits = [w_points_list[0][:, 0].min(), w_points_list[0][:, 0].max()]
y_limits = [w_points_list[0][:, 1].min(), w_points_list[0][:, 1].max()]
z_limits = [w_points_list[0][:, 2].min(), w_points_list[0][:, 2].max()]

# Find the overall range
all_limits = np.array([x_limits, y_limits, z_limits])
min_limit = all_limits.min()
max_limit = all_limits.max()

# Set the same limits for all axes
ax.set_xlim(min_limit, max_limit)
ax.set_ylim(min_limit, max_limit)
ax.set_zlim(min_limit, max_limit)

# Show the plot
plt.show()
