import pandas as pd
import ast
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
import cv2 as cv
import math
import scipy.interpolate as interp
np.set_printoptions(precision=4, suppress=True)
np.random.seed(1)

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
    
    def convert_space_separated_list(value):
        # Split the string by spaces and convert to a list of floats
        value = value.strip('[]')
        return [float(x) for x in value.split()]
    
    # Read the csv file into a DataFrame, applying the conversion function to relevant columns
    df = pd.read_csv(file_path, converters={
        'Crater Indices': convert_to_list,
        'Centre points 2D coord': convert_to_array_of_arrays,
        'Camera Position': convert_space_separated_list
    })

    return df

def get_intrinsics(fov, im_width, im_height):
    fov = fov*math.pi/180
    fx = im_width/(2*math.tan(fov)) # Conversion from fov to focal length
    fy = im_height/(2*math.tan(fov)) # Conversion from fov to focal length
    cx = im_width/2
    cy = im_height/2
    return (np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]]))

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

def percentage_difference(vec1, vec2):
    """
    Calculate the percentage difference between two 2D arrays based on the sum of their components.

    Parameters:
    vec1 (list of lists or np.array): First 2D array.
    vec2 (list of lists or np.array): Second 2D array.

    Returns:
    float: Percentage difference between the sums of the components of the two 2D arrays.
    """
    # Calculate the sum of the components of each 2D array
    sum_vec1 = sum(sum(sublist) for sublist in vec1)
    sum_vec2 = sum(sum(sublist) for sublist in vec2)
    
    # Calculate the absolute difference between the sums
    abs_diff = abs(sum_vec1 - sum_vec2)
    
    # Calculate the average of the sums
    avg_sum = (sum_vec1 + sum_vec2) / 2
    
    # Calculate the percentage difference
    percent_diff = (abs_diff / avg_sum) * 100
    
    return percent_diff

def estimation_offset(all_positions, ref_points):
    # Convert lists to NumPy arrays for easier manipulation
    all_positions = np.array(all_positions)
    ref_points = np.array(ref_points)
    
    # Calculate the differences between corresponding points
    differences = all_positions - ref_points
    
    # Calculate the Euclidean distance for each set of points
    distances = np.linalg.norm(differences, axis=1)
    # print(distances)


    # Calculate the average distance
    average_distance = np.mean(distances)
    
    return average_distance

def add_noise(i_points, stdev):
    for i in range(len(i_points)):
        noise = np.random.normal(0, stdev, i_points[i].shape)
        i_points[i] += noise

    return i_points

def graph_world(robbins, points):
    x_values = robbins[' X'][::20]
    y_values = robbins[' Y'][::20]
    z_values = robbins[' Z'][::20]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_values, y_values, z_values, c='b', marker='o')

    # Extract the additional points
    extra_x = [point[0] for point in points]
    extra_y = [point[1] for point in points]
    extra_z = [point[2] for point in points]

    # Plot the additional points in red
    ax.scatter(extra_x, extra_y, extra_z, c='r', marker='^', label='Extra Points')

    # Add title and labels
    ax.set_title('3D Scatter Plot of Robbins Data')
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values')
    ax.set_zlabel('Z Values')

    # Set equal aspect ratio
    max_range = max(x_values.max() - x_values.min(), 
                    y_values.max() - y_values.min(), 
                    z_values.max() - z_values.min()) / 2.0

    mid_x = (x_values.max() + x_values.min()) * 0.5
    mid_y = (y_values.max() + y_values.min()) * 0.5
    mid_z = (z_values.max() + z_values.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Show the plot
    ax.set_box_aspect([1, 1, 1])
    plt.show()

def graph_moon_and_camera_points(radius_moon_km, points):
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate the sphere representing the Moon
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius_moon_km * np.outer(np.cos(u), np.sin(v))
    y = radius_moon_km * np.outer(np.sin(u), np.sin(v))
    z = radius_moon_km * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the sphere
    ax.plot_surface(x, y, z, color='grey', alpha=0.6, rstride=5, cstride=5)

    # Extract the additional points (camera points)
    extra_x = [point[0] for point in points]
    extra_y = [point[1] for point in points]
    extra_z = [point[2] for point in points]

    # Plot the additional points in red
    ax.scatter(extra_x, extra_y, extra_z, c='r', marker='^', s=100, label='Spacecraft Orbit')

    # Add title and labels
    ax.set_title('3D Plot of the Moon and Camera Orbit')
    ax.set_xlabel('X Values (km x 10^3)')
    ax.set_ylabel('Y Values (km x 10^3)')
    ax.set_zlabel('Z Values (km x 10^3)')

    # Set equal aspect ratio
    max_range = max(x.max() - x.min(), 
                    y.max() - y.min(), 
                    z.max() - z.min()) / 2.0

    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set the box aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

# Image dimensions and FOV
image_width = 1024
image_height = 1024
fov = 30  # Field of view in degrees

# Calculate the focal length
focal_length = image_width / (2 * np.tan(np.radians(fov) / 2))

# Create the initial intrinsic matrix
camera_matrix = get_intrinsics(fov, image_width, image_height)

camera_matrix = np.array(camera_matrix)
ground_truth_matrix = camera_matrix

robbins = read_txt_to_dataframe('data/robbins_navigation_dataset_christians_all.txt')
t_inst = read_csv_to_dataframe('output/testing_instance.csv')

w_points_list = []
i_points_list = []
ref_vals = []

# Extract the points from the dataframes
for i, (index, row) in enumerate(t_inst.iterrows()):

    i_points_list.append(np.vstack(t_inst['Centre points 2D coord'][i]))
    ref_vals.append(t_inst['Camera Position'][i])
    indices = t_inst['Crater Indices'][i]

    x_vals = robbins.iloc[indices][' X']
    y_vals = robbins.iloc[indices][' Y']
    z_vals = robbins.iloc[indices][' Z']

    w_points = robbins.iloc[indices][[' X', ' Y', ' Z']].to_numpy()
    w_points_list.append(w_points)


# Camera calibration
object_points = [obj.reshape(-1, 1, 3).astype(np.float32) for obj in w_points_list]
image_points = [img.reshape(-1, 1, 2).astype(np.float32) for img in i_points_list]

# Calibration flags, NO distortion coefficients
flags = (cv.CALIB_FIX_K1 | cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3 |
         cv.CALIB_FIX_K4 | cv.CALIB_FIX_K5 | cv.CALIB_FIX_K6 |
         cv.CALIB_FIX_TANGENT_DIST | cv.CALIB_USE_INTRINSIC_GUESS)

# Experiment 1: Frequency vs Error with variable number of images images
Experiment1 = False
if Experiment1:
    # Frequency *10 for current testing instance (10-1000)
    MIN_FREQ = 1    # 10 second imaging interval
    MAX_FREQ = 100  # 1000 second imaging interval
    NUM_IMAGES = 5  # 5 images used for calibration per frequency
    NOISE = 0       # Standard deviation of noise added to image points

    frequencies = [i*10 for i in range(MIN_FREQ, MAX_FREQ+1)]
    freq_error = []
    for freq in range(MIN_FREQ, MAX_FREQ+1):

        object_points_i = [object_points[i] for i in range(0, freq * NUM_IMAGES, freq)]
        image_points_i = [image_points[i] for i in range(0, freq * NUM_IMAGES, freq)]
        ref_points = [ref_vals[i] for i in range(0, freq * NUM_IMAGES, freq)]

        # Add noise to the image points (if needed)
        if NOISE > 0:
            image_points_i = add_noise(image_points_i, NOISE)

        camera_matrix = np.array(get_intrinsics(fov, image_width, image_height))
        ret, calc_cam_matrix, dist_coeffs, rvecs, tvecs, stdDeviationIntrinsics, stdDeviationExtrinsics, perViewErrors = cv.calibrateCameraExtended(
            object_points_i, image_points_i, (1024, 1024), camera_matrix, None, flags=flags)
        
        all_positions = []
        for i in range(len(rvecs)):
            a = get_extrinsic_matrix(rvecs[i], tvecs[i])
            pos = get_camera_position_from_extrinsic(a)
            pos = pos/1000
            all_positions.append(pos)

        # Use either %difference or absolute offset for error calculation
        # err = percentage_difference(all_positions, ref_points)
        err = estimation_offset(all_positions, ref_points)
        freq_error.append(err)

    # Graph
    plt.plot(frequencies, freq_error, marker='o')
    plt.title('Frequency vs Error')
    plt.xlabel('Imaging Time Interval (seconds)')
    plt.ylabel('Average positional error (km)')
    plt.grid(True)
    plt.show()

# Experiment 2: Noise vs Error, 3D plots (Not used in the report)
Experiment2 = False
if Experiment2:
    MIN_FREQ = 15
    MAX_FREQ = 105
    MIN_IMAGES = 3
    MAX_IMAGES = 7
    JUMP = 10   # Frequency jump (will consider every 10th frequency in the range)
    NOISE = 0

    frequencies = [i * 10 for i in range(MIN_FREQ, MAX_FREQ + 1, JUMP)]
    num_images_range = range(MIN_IMAGES, MAX_IMAGES + 1)

    # Collect data for 3D plot
    data = []

    for num_images in num_images_range:
        freq_error = []
        for freq in range(MIN_FREQ, MAX_FREQ + 1, JUMP):
            object_points_i = [object_points[i] for i in range(0, freq * num_images, freq)]
            image_points_i = [image_points[i] for i in range(0, freq * num_images, freq)]
            ref_points = [ref_vals[i] for i in range(0, freq * num_images, freq)]

            # Add noise to the image points (if needed)
            if NOISE > 0:
                image_points_i = add_noise(image_points_i, NOISE)

            camera_matrix = np.array(get_intrinsics(fov, image_width, image_height))
            ret, calc_cam_matrix, dist_coeffs, rvecs, tvecs, stdDeviationIntrinsics, stdDeviationExtrinsics, perViewErrors = cv.calibrateCameraExtended(
                object_points_i, image_points_i, (1024, 1024), camera_matrix, None, flags=flags)

            all_positions = []
            for i in range(len(rvecs)):
                a = get_extrinsic_matrix(rvecs[i], tvecs[i])
                pos = get_camera_position_from_extrinsic(a)
                pos = pos / 1000
                all_positions.append(pos)

            err = estimation_offset(all_positions, ref_points)
            freq_error.append(err)

        data.append((num_images, frequencies, freq_error))

    # Plotting the 3D graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for num_images, freqs, errors in data:
        ax.scatter(freqs, [num_images] * len(freqs), errors, label=f'{num_images} Images')
        ax.plot(freqs, [num_images] * len(freqs), errors)

    ax.set_xlabel('Frequency')
    ax.set_ylabel('Number of Images')
    ax.set_zlabel('Positional Error (km)')
    ax.legend()
    ax.grid(True)
    plt.show()

    # Plotting the 2D connected scatter plot
    fig, ax = plt.subplots()

    for num_images, freqs, errors in data:
        # Plot scatter points
        ax.scatter(freqs, errors, label=f'{num_images} Images')
        
        # Plot connecting lines
        ax.plot(freqs, errors)

    ax.set_xlabel('Frequency')
    ax.set_ylabel('Positional Error (km)')
    ax.legend()
    ax.grid(True)
    plt.show()

    # Prepare data for surface plot
    num_images_vals = np.array([d[0] for d in data])
    freq_vals = np.array(data[0][1])
    error_vals = np.array([d[2] for d in data])

    X, Y = np.meshgrid(freq_vals, num_images_vals)
    Z = np.array(error_vals)

    # Plotting the 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Frequency')
    ax.set_ylabel('Number of Images')
    ax.set_zlabel('Positional Error (km)')
    ax.grid(True)
    plt.show()

# Experiment 3: Number of calibration images vs Error, heatmap of frequency
Experiment3 = False
if Experiment3:
    MIN_FREQ = 10
    MAX_FREQ = 100
    MIN_IMAGES = 3
    MAX_IMAGES = 7
    JUMP = 10       # Frequency jump (will consider every 10th frequency in the range)
    NOISE = 0

    frequencies = [i * 10 for i in range(MIN_FREQ, MAX_FREQ + 1, JUMP)]
    num_images_range = range(MIN_IMAGES, MAX_IMAGES + 1)

    # Collect data for 2D plot
    num_images_list = []
    freq_list = []
    error_list = []

    for num_images in num_images_range:
        for freq in range(MIN_FREQ, MAX_FREQ + 1, JUMP):
            object_points_i = [object_points[i] for i in range(0, freq * num_images, freq)]
            image_points_i = [image_points[i] for i in range(0, freq * num_images, freq)]
            ref_points = [ref_vals[i] for i in range(0, freq * num_images, freq)]

            # Add noise to the image points (if needed)
            if NOISE > 0:
                image_points_i = add_noise(image_points_i, NOISE)

            camera_matrix = np.array(get_intrinsics(fov, image_width, image_height))
            ret, calc_cam_matrix, dist_coeffs, rvecs, tvecs, stdDeviationIntrinsics, stdDeviationExtrinsics, perViewErrors = cv.calibrateCameraExtended(
                object_points_i, image_points_i, (1024, 1024), camera_matrix, None, flags=flags)

            all_positions = []
            for i in range(len(rvecs)):
                a = get_extrinsic_matrix(rvecs[i], tvecs[i])
                pos = get_camera_position_from_extrinsic(a)
                pos = pos / 1000
                all_positions.append(pos)

            err = estimation_offset(all_positions, ref_points)
            num_images_list.append(num_images)
            freq_list.append(freq*JUMP)
            error_list.append(err)

    # Plotting the 2D scatter plot with color representing frequency
    fig, ax = plt.subplots()

    # Plot scatter points with zorder to ensure they are on top of the grid
    scatter = ax.scatter(num_images_list, error_list, c=freq_list, cmap='plasma', edgecolors='grey', zorder=3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Imaging Time Interval (seconds)')

    # Set x-axis interval to 1
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # Set labels and grid
    ax.set_xlabel('Number of Images')
    ax.set_ylabel('Positional Error (km)')
    ax.grid(True, zorder=0)  # Ensure grid is below the scatter points

    plt.show()

# Experiment 4: Noise vs Error, heatmap of frequency
Experiment4 = True
if Experiment4:
    MIN_FREQ = 10
    MAX_FREQ = 100
    NUM_IMAGES = 5
    MIN_NOISE = 0    # MIN Standard deviation of noise added to image points
    MAX_NOISE = 8    # MAX Standard deviation of noise added to image points
    JUMP = 10

    frequencies = [i * 100 for i in range(MIN_FREQ, MAX_FREQ + 1, JUMP)]
    noise_range = range(MIN_NOISE, MAX_NOISE + 1)

    # Collect data for 2D plot
    noise_list = []
    freq_list = []
    error_list = []

    for noise in noise_range:
        for freq in range(MIN_FREQ, MAX_FREQ + 1, JUMP):
            object_points_i = [object_points[i] for i in range(0, freq * NUM_IMAGES, freq)]
            image_points_i = [image_points[i] for i in range(0, freq * NUM_IMAGES, freq)]
            ref_points = [ref_vals[i] for i in range(0, freq * NUM_IMAGES, freq)]

            image_points_i = add_noise(image_points_i, noise)

            camera_matrix = np.array(get_intrinsics(fov, image_width, image_height))
            ret, calc_cam_matrix, dist_coeffs, rvecs, tvecs, stdDeviationIntrinsics, stdDeviationExtrinsics, perViewErrors = cv.calibrateCameraExtended(
                object_points_i, image_points_i, (1024, 1024), camera_matrix, None, flags=flags)

            all_positions = []
            for i in range(len(rvecs)):
                a = get_extrinsic_matrix(rvecs[i], tvecs[i])
                pos = get_camera_position_from_extrinsic(a)
                pos = pos / 1000
                all_positions.append(pos)

            err = estimation_offset(all_positions, ref_points)
            noise_list.append(noise)
            freq_list.append(freq*10)
            error_list.append(err)

    # Plotting the 2D scatter plot with color representing frequency
    fig, ax = plt.subplots()

    # Plot scatter points with zorder to ensure they are on top of the grid
    scatter = ax.scatter(noise_list, error_list, c=freq_list, cmap='viridis', edgecolors='grey', zorder=3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Imaging Time Interval (seconds)')

    # Set x-axis interval to 1
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # Set labels and grid
    ax.set_xlabel('Noise (standard deviation)')
    ax.set_ylabel('Positional Error (km)')
    ax.grid(True, zorder=0) 

    plt.show()

# Graph moon and camera points
Graph_Moon = False
if Graph_Moon:
    GRAPH_FREQUENCY = 20
    GRAPH_NUM_IMAGES = 5
    poses = [ref_vals[i] for i in range(0, GRAPH_FREQUENCY * GRAPH_NUM_IMAGES, GRAPH_FREQUENCY)]
    graph_moon_and_camera_points(1737.1*10**3, [[i[0]*1000, i[1]*1000, i[2]*1000] for i in poses])
