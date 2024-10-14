from numba import njit
import numpy as np
from orbdtools import ArcObs, OrbeleTrans, FrameTrans, KeprvTrans
import pyvista as pv
import astropy
from scipy.spatial.transform import Rotation
import cv2
import csv

mu_km_s = 4902.800118 # gravity constant of moon in km/s

# mu_km_s = GM_EARTH * (1 / 1000000000) # converting m^3/s^2 to km^3/s^2
# mu_km_min = GM_EARTH * (3600 / 1000000000) # converting m^3/s^2 to km^3/minute^2

@njit
def two_body(mu, tau, ri, vi):
    """
    :param mu: gravitational constant(km ** 3 / sec ** 2)
    :param tau: propagation time interval(seconds)
    :param ri: initial eci position vector(kilometers)
    :param vi: initial eci velocity vector(kilometers / second)
    :return:
    rf = final eci position vector(kilometers)
    vf = final eci velocity vector(kilometers / second)
    """
    tolerance = 1.0e-10
    u = np.float64(0.0)
    uold = 100
    dtold = 100
    # imax = 20
    imax = 100

    # umax = sys.float_info.max
    # umin = -sys.float_info.max
    umax = np.float64(1.7976931348623157e+308)
    umin = np.float64(-umax)

    orbits = 0

    tdesired = tau

    threshold = tolerance * abs(tdesired)

    r0 = np.linalg.norm(ri)

    n0 = np.dot(ri, vi)

    beta = 2 * (mu / r0) - np.dot(vi, vi)

    if (beta != 0):
        umax = +1 / np.sqrt(abs(beta))
        umin = -1 / np.sqrt(abs(beta))

    if (beta > 0):
        orbits = beta * tau - 2 * n0
        orbits = 1 + (orbits * np.sqrt(beta)) / (np.pi * mu)
        orbits = np.floor(orbits / 2)

    for i in range(imax):
        q = beta * u * u
        q = q / (1 + q)
        n = 0
        r = 1
        l = 1
        s = 1
        d = 3
        gcf = 1
        k = -5

        gold = 0

        while (gcf != gold):
            k = -k
            l = l + 2
            d = d + 4 * l
            n = n + (1 + k) * l
            r = d / (d - n * r * q)
            s = (r - 1) * s
            gold = gcf
            gcf = gold + s

        h0 = 1 - 2 * q
        h1 = 2 * u * (1 - q)
        u0 = 2 * h0 * h0 - 1
        u1 = 2 * h0 * h1
        u2 = 2 * h1 * h1
        u3 = 2 * h1 * u2 * gcf / 3

        if (orbits != 0):
            u3 = u3 + 2 * np.pi * orbits / (beta * np.sqrt(beta))

        r1 = r0 * u0 + n0 * u1 + mu * u2
        dt = r0 * u1 + n0 * u2 + mu * u3
        slope = 4 * r1 / (1 + beta * u * u)
        terror = tdesired - dt

        if (abs(terror) < threshold):
            break

        if ((i > 1) and (u == uold)):
            break

        if ((i > 1) and (dt == dtold)):
            break

        uold = u
        dtold = dt
        ustep = terror / slope

        if (ustep > 0):
            umin = u
            u = u + ustep
            if (u > umax):
                u = (umin + umax) / 2
        else:
            umax = u
            u = u + ustep
            if (u < umin):
                u = (umin + umax) / 2

        if (i == imax):
            print('max iterations in twobody2 function')

    # usaved = u
    f = 1.0 - (mu / r0) * u2
    gg = 1.0 - (mu / r1) * u2
    g = r0 * u1 + n0 * u2
    ff = -mu * u1 / (r0 * r1)

    rf = f * ri + g * vi
    vf = ff * ri + gg * vi
    return rf, vf


@njit
def create_rotation_matrix(axis, angle):
    # Normalize the axis
    axis = axis / np.linalg.norm(axis)
    kx, ky, kz = axis

    # Compute sine and cosine of the angle
    c = np.cos(angle)
    s = np.sin(angle)

    # Compute the cross-product matrix K
    K = np.array([[0, -kz, ky],
                  [kz, 0, -kx],
                  [-ky, kx, 0]])

    # Compute the rotation matrix using Rodrigues' formula
    R = np.eye(3) + s * K + (1 - c) * np.dot(K, K)

    return R

@njit
def get_craters_world_numba(lines):
    # Initialize the matrices
    N = len(lines)
    crater_param = np.zeros((N, 6))
    crater_conic = np.zeros((N, 3, 3))
    crater_conic_inv = np.zeros((N, 3, 3))
    Hmi_k = np.zeros((N, 4, 3))

    ENU = np.zeros((N, 3, 3))
    L_prime = np.zeros((N, 3, 3))
    S = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    # Populate the matrices
    k = np.array([0, 0, 1])

    for idx, line in enumerate(lines):
        X, Y, Z, a, b, phi = line
        a = (a * 1000) / 2  # converting diameter to meter
        b = (b * 1000) / 2
        X = X / 1000
        Y = Y / 1000
        Z = Z / 1000
        a = a / 1000
        b = b / 1000
        phi = phi

        # Populate crater_param
        crater_param[idx] = [X, Y, Z, a, b, phi]

        # Calculate conic matrix
        A = a ** 2 * (np.sin(phi) ** 2) + b ** 2 * (np.cos(phi) ** 2)
        B = 2 * (b ** 2 - a ** 2) * np.cos(phi) * np.sin(phi)
        C = a ** 2 * (np.cos(phi) ** 2) + b ** 2 * (np.sin(phi) ** 2)
        D = -2 * A * 0 - B * 0
        E = -B * 0 - 2 * C * 0
        F = A * 0 ** 2 + B * 0 * 0 + C * 0 ** 2 - a ** 2 * b ** 2

        # Populate crater_conic
        # crater_conic[idx] = [[A, B / 2, D / 2], [B / 2, C, E / 2], [D / 2, E / 2, F]]
        crater_conic[idx] = np.array([[A, B / 2, D / 2], [B / 2, C, E / 2], [D / 2, E / 2, F]])

        crater_conic_inv[idx] = np.linalg.inv(crater_conic[idx])
        # get ENU coordinate
        Pc_M = np.array([X, Y, Z])

        u = Pc_M / np.linalg.norm(Pc_M)
        e = np.cross(k, u) / np.linalg.norm(np.cross(k, u))
        n = np.cross(u, e) / np.linalg.norm(np.cross(u, e))

        R_on_the_plane = create_rotation_matrix(u, phi)
        e_prime = R_on_the_plane @ e
        n_prime = R_on_the_plane @ n
        curr_L_prime = np.empty((3, 3), dtype=np.float64)
        curr_L_prime[:, 0] = e_prime
        curr_L_prime[:, 1] = n_prime
        curr_L_prime[:, 2] = u

        TE_M = np.empty((3, 3), dtype=np.float64)
        TE_M[:, 0] = e
        TE_M[:, 1] = n
        TE_M[:, 2] = u

        ENU[idx] = TE_M
        # compute Hmi

        Hmi = np.hstack((TE_M.dot(S), Pc_M.reshape(-1, 1)))
        Hmi_k[idx] = np.vstack((Hmi, k.reshape(1, 3)))
        L_prime[idx] = curr_L_prime

    return crater_param, crater_conic, crater_conic_inv, ENU, Hmi_k, L_prime


def read_crater_database(craters_database_text_dir):
    with open(craters_database_text_dir, "r") as f:
        lines = f.readlines()[1:]  # ignore the first line
    lines = [i.split(',') for i in lines]
    lines = np.array(lines)

    ID = lines[:, 0]
    lines = np.float64(lines[:, 1:])

    # convert all to conics
    db_CW_params, db_CW_conic, db_CW_conic_inv, db_CW_ENU, db_CW_Hmi_k, db_L_prime = get_craters_world_numba(lines)


    return db_CW_params, db_CW_conic, db_CW_conic_inv, db_CW_ENU, db_CW_Hmi_k, ID
    


def visible_points_on_sphere(points, sphere_center, sphere_radius, camera_position, valid_indices):
    """Return the subset of the 3D points on the sphere that are visible to the camera."""
    visible_points = []
    visible_indices = []
    visible_len_P_cam = []
    non_visible_len_P_cam = []

    for idx in valid_indices:
        point = points[idx, :]

        # 1. Translate the origin to the camera
        P_cam = point - camera_position

        # 2. Normalize the translated point
        P_normalized = P_cam / np.linalg.norm(P_cam)

        # 3 & 4. Solve for the real roots
        # Coefficients for the quadratic equation
        a = np.dot(P_normalized, P_normalized)
        b = 2 * np.dot(P_normalized, camera_position - sphere_center)
        c = np.dot(camera_position - sphere_center, camera_position - sphere_center) - sphere_radius ** 2

        discriminant = b ** 2 - 4 * a * c
        root1 = (-b + np.sqrt(discriminant)) / (2 * a)
        root2 = (-b - np.sqrt(discriminant)) / (2 * a)

        min_root = np.minimum(root1, root2)
        # 5. Check which real root matches the length of P_cam
        length_P_cam = np.linalg.norm(P_cam)

        # 6 & 7. Check visibility
        if (np.abs(min_root - length_P_cam) < 1): # min_root should be equivalent to length_p_cam. 
            visible_points.append(point)
            visible_indices.append(idx)
            visible_len_P_cam.append(length_P_cam)

    return visible_points, visible_indices

def get_intrinsic(calibration_file):
    f = open(calibration_file, 'r')
    lines = f.readlines()
    calibration = lines[1].split(',')
    fov = int(calibration[0])
    # fx = int(calibration[1])
    # fy = int(calibration[2])
    image_width = int(calibration[3])
    image_height = int(calibration[4])

    fov = fov*np.pi/180
    fx = image_width/(2*np.tan(fov/2)) # Conversion from fov to focal length
    fy = image_height/(2*np.tan(fov/2)) # Conversion from fov to focal length
    cx = image_width/2
    cy = image_height/2

    return (np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]]))

@njit
def matrix_multiply(A, B, A_rows, A_cols, B_cols):
    C = np.zeros((A_rows, B_cols))
    for i in range(A_rows):
        for j in range(B_cols):
            C[i, j] = 0.0
            for k in range(A_cols):
                C[i, j] += A[i, k] * B[k, j]

    return C

@njit
def conic_from_crater(C_conic, Hmi_k, Pm_c):
    '''
    :param C_conic_inv: [3x3]
    :param Hmi_k: [4x3]
    :param Pm_c: [3x4]
    :param A: [3x3]
    :return:
    '''
    # Hci = np.dot(Pm_c, Hmi_k)
    Hci = matrix_multiply(Pm_c, Hmi_k, 3, 4, 3)
    Hci_inv = np.linalg.inv(Hci)
    # Astar = np.dot(np.dot(Hci, C_conic_inv), Hci.T)
    Astar = matrix_multiply(Hci_inv.T, C_conic, 3, 3, 3)
    A = matrix_multiply(Astar, Hci_inv, 3, 3, 3)

    # A_ = Hci.T @ A @ Hci
    # A = np.linalg.inv(Astar)
    # A = inverse_3x3_cpu(Astar)
    return A

def create_extrinsic_matrix(plane_normal, radius, rot_angle_range, rotate=False, att_noise_range=None):
    # Ensure the plane normal is a unit vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Camera's z-axis is the opposite of the plane normal
    z_axis = -plane_normal

    # Determine an up vector. If the z-axis is not parallel to [0, 1, 0], use [0, 1, 0] as the up vector.
    # Otherwise, use [1, 0, 0].
    if np.abs(np.dot(z_axis, [0, 1, 0])) != 1:
        up_vector = [0, 1, 0]
    else:
        up_vector = [1, 0, 0]

    # Camera's x-axis
    x_axis = np.cross(up_vector, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Camera's y-axis
    y_axis = np.cross(z_axis, x_axis)

    rotation_angle = 0
    if rotate:
        # Compute a random rotation angle between 0 and 60 degrees
        rotation_angle = np.random.uniform(0, np.radians(rot_angle_range))

        # Create a rotation matrix around a random axis
        random_axis = np.random.rand(3)
        random_axis = random_axis / np.linalg.norm(random_axis)
        rand_rot_mat = Rotation.from_rotvec(random_axis * rotation_angle)

        # Apply the random rotation to R
        R = R @ rand_rot_mat
    else:
        # Rotation matrix
        R = np.array([x_axis, y_axis, z_axis]).T
        

    # Translation vector (camera's position in world coordinates)
    t = plane_normal * radius

    # Extrinsic matrix
    extrinsic = np.zeros((3, 4))
    extrinsic[:3, :3] = R.T
    extrinsic[:3, 3] = -R.T @ t  # Convert world position to camera-centric position
    # extrinsic[3, 3] = 1

    ## ADD noise here

    # att_noise = np.random.uniform(0, np.radians(att_noise_range))
    # noise_att = add_noise_to_matrix(R, att_noise)

    return extrinsic

def extract_ellipse_parameters_from_conic(conic):
    A = conic[0, 0]
    B = conic[0, 1] * 2
    C = conic[1, 1]
    D = conic[0, 2] * 2
    F = conic[1, 2] * 2
    G = conic[2, 2]

    # Sanity test.
    denominator = B ** 2 - 4 * A * C
    if (B ** 2 - 4 * A * C >= 0) or (C * np.linalg.det(conic) >= 0):
        # print('Conic equation is not a nondegenerate ellipse')
        return False, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001

    #  Method from:
    #  https://en.wikipedia.org/wiki/Ellipse
    #  Convention in wikipedia:
    #   [ A B/2  D/2]
    #   [ B/2 C  E/2]
    #   [ D/2 E/2 F]]
    #  The following equations reexpresses wikipedia's formulae in Christian et
    #  al.'s convention.

    # Get centres.
    try:
        x_c = (2 * C * D - B * F) / denominator
        y_c = (2 * A * F - B * D) / denominator

        # Get semimajor and semiminor axes.
        KK = 2 * (A * F ** 2 + C * D ** 2 - B * D * F + (B ** 2 - 4 * A * C) * G)
        root = np.sqrt((A - C) ** 2 + B ** 2)
        a = -1 * np.sqrt(KK * ((A + C) + root)) / denominator
        b = -1 * np.sqrt(KK * ((A + C) - root)) / denominator

        if B != 0:
            # phi = math.atan((C - A - root) / B)  # Wikipedia had this as acot; should be atan. Check https://math.stackexchange.com/questions/1839510/how-to-get-the-correct-angle-of-the-ellipse-after-approximation/1840050#1840050
            phi = 0.5 * np.arctan2(-B, (C - A)) - (-np.pi) # to convert to the positive realm
        elif A < C:
            phi = 0
        else:
            phi = np.pi / 2

        return True, x_c, y_c, a, b, phi
    except:
        return False, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001

def visualize_ellipses(image_size, ellipses, center_points, output_filename):
    """
    Visualize ellipses on an image plane using OpenCV and save the result to a file.
    
    :param image_size: Tuple representing the size of the image (width, height).
    :param ellipses: List of ellipses where each ellipse is defined by (center, axes, angle).
                     Example: [(center_x, center_y), (semi_major, semi_minor), angle]
    :param output_filename: The name of the output image file to save (e.g., 'output_image.png').
    """
    # Create a blank image (white background)
    image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
    
    # Iterate through each ellipse and draw it on the image
    for ellipse in ellipses:
        x, y, a, b, theta = ellipse
        
        # Define the center, axes, and angle
        center = (int(x), int(y))
        axes = (int(a), int(b))
        angle = theta
        
        # Draw the ellipse on the image
        cv2.ellipse(image, center, axes, angle, 0, 360, (0, 0, 255), 2)  # Red ellipse with thickness 2
    
    # Iterate through each center point and draw it on the image
    for center_point in center_points:
        x, y = center_point
        # Draw the center point as a small circle (radius 5)
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green center point, filled circle
    
    # Save the image to a file
    cv2.imwrite(output_filename, image)
    print(f"Image saved as {output_filename}")

def save_to_csv(output_filename, i, propagated_positions, propagated_velocities, camera_extrinsic, final_visible_ID, points_on_img_plane):
    """
    Save the position, velocity, camera extrinsic matrix, 2D coordinates, and crater IDs to a CSV file.
    
    :param output_filename: The CSV file to write data to.
    :param i: The current frame index.
    :param propagated_positions: The propagated positions of the spacecraft.
    :param propagated_velocities: The propagated velocities of the spacecraft.
    :param camera_extrinsic: The camera's extrinsic matrix (3x4).
    :param final_visible_ID: List of crater IDs that are visible on the image plane.
    :param points_on_img_plane: The 2D coordinates of the craters on the image plane.
    """
    with open(output_filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write the header if it's the first time
        if i == 0:
            csvwriter.writerow(['Frame', 'Position (x, y, z)', 'Velocity (vx, vy, vz)', 'Camera Extrinsic Matrix', 'Crater ID', '2D Coordinates (x, y)'])

        # Write data for the current frame
        for j in range(len(final_visible_ID)):
            csvwriter.writerow([
                i,  # Frame index
                ','.join(map(str, propagated_positions[i])),  # Spacecraft position
                ','.join(map(str, propagated_velocities[i])),  # Spacecraft velocity
                ','.join(map(str, camera_extrinsic[i].flatten())),  # Flatten camera extrinsic matrix (3x4)
                final_visible_ID[j],  # Crater ID
                ','.join(map(str, points_on_img_plane[:, j]))  # 2D coordinates on image plane
            ])
            
if __name__ == "__main__":
    # generate a random orbit in keplerian state
    coe_oe = np.array([np.random.uniform(1837.7, 2137.7),
                # 1e-8, 1e-8,  # Circular orbit
                np.random.uniform(0, 0.01), #e
                # 0,
                np.random.uniform(np.deg2rad(1), np.deg2rad(179)), #i
                np.random.uniform(0, 2 * np.pi), #ohm
                np.random.uniform(0, 2 * np.pi), #omega
                np.random.uniform(0, 2 * np.pi)]) #v

    # convert to state vector
    rv = KeprvTrans.coe2rv(coe_oe, mu_km_s)
    
    
    # Create a time stack: 0 to 3600 seconds, with a 100-second interval
    time_stack = np.arange(0, 50, 5)  # From 0 to 3600 seconds with 100 seconds interval

    # Initialize lists to store propagated positions and velocities
    propagated_positions = []
    propagated_velocities = []

    # Propagate the orbit for each time in the stack using the two_body function
    # ri, vi = rv  # Initial position and velocity vectors
    for tau in time_stack:
        rf, vf = two_body(mu_km_s, tau, rv[0:3], rv[3:6])
        propagated_positions.append(rf)
        propagated_velocities.append(vf)

    # Convert lists to numpy arrays for easier handling
    propagated_positions = np.array(propagated_positions)
    propagated_velocities = np.array(propagated_velocities)
    
    camera_extrinsic = np.zeros([propagated_positions.shape[0], 3, 4])
    for i in range(propagated_positions.shape[0]):
        camera_extrinsic[i] = create_extrinsic_matrix(propagated_positions[i] / np.linalg.norm(propagated_positions[i]),
                                        np.linalg.norm(propagated_positions[i]), rot_angle_range=0, rotate=False)

    ### read robbin's catalog
    data_dir = 'data'
    all_craters_database_text_dir = data_dir + '/robbins_navigation_dataset_christians_all.txt'

    CW_params, CW_conic, CW_conic_inv, CW_ENU, CW_Hmi_k, ID = \
            read_crater_database(all_craters_database_text_dir)
            
    csv_filename = 'testing_instance.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header row
        csvwriter.writerow(['Camera Position', 'Velocity', 'Camera Extrinsic', 'Centre points 2D coord', 'Crater Indices'])
        
    # multiply by the camera intrinsic
    calibration_file = data_dir + '/calibration.txt'
    K = get_intrinsic(calibration_file)
    img_w = img_h = 1024
    for i in range(propagated_positions.shape[0]):
        curr_cam = camera_extrinsic[i]
        cam_pos = -curr_cam[0:3, 0:3].T @ curr_cam[0:3, 3]
        
        curr_cam = K @ curr_cam

        # 1) project all 3D points onto the image plane
        projected_3D_points = curr_cam @ np.hstack([CW_params[:, 0:3], np.ones((CW_params.shape[0], 1))]).T
        points_on_img_plane = np.array([projected_3D_points[0, :] / projected_3D_points[2, :],
                                        projected_3D_points[1, :] / projected_3D_points[2, :]])

        # 2) Filter points that are within the image dimensions
        within_img_valid_indices = np.where((points_on_img_plane[0, :] >= 0) &
                                    (points_on_img_plane[0, :] <= img_w) &
                                    (points_on_img_plane[1, :] >= 0) &
                                    (points_on_img_plane[1, :] <= img_h) &
                                ~np.isnan(points_on_img_plane[0, :]) &
                                ~np.isnan(points_on_img_plane[1, :]))[0]

        # 3) filter the points that are behind the horizon
        visible_pts, filtered_indices = visible_points_on_sphere(CW_params[:, 0:3], np.array([0, 0, 0]), np.linalg.norm(CW_params[0, 0:3]),
                                    cam_pos, within_img_valid_indices)

        final_visible_ID = []
        imaged_params = []
        final_center_point_2D_coord = []
        # 4) project conic and get ellipse parameters on image plane
        for j in range(len(filtered_indices)):
            A = conic_from_crater(CW_conic[filtered_indices[j]], CW_Hmi_k[filtered_indices[j]], curr_cam)

            # convert A to ellipse parameters
            flag, x_c, y_c, a, b, phi = extract_ellipse_parameters_from_conic(A)

            if np.isnan(a) or np.isnan(b):
                continue

            if (flag): # if it's proper conic
                if b > 5:
                    final_visible_ID.append(filtered_indices[j])
                    imaged_params.append([x_c, y_c, a, b, phi])
                    final_center_point_2D_coord.append(points_on_img_plane[:, filtered_indices[j]])
        
        # 5) visualise ellipse on the image plane, and save it.
        visualize_ellipses([img_h, img_w], imaged_params, final_center_point_2D_coord, 'testing_img_'+str(i)+'.png')
        
        # 6) save position, velocity, camera_extrinsic, 2d coord. and ID in a csv file
        
        with open(csv_filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([propagated_positions[i], propagated_velocities[i], camera_extrinsic[i], final_center_point_2D_coord, final_visible_ID])

        