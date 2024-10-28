import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Ellipse
import cv2 as cv
np.set_printoptions(precision=10, suppress=True)
np.random.seed(0)
# Random seed is set to 0 for reproducibility

class Camera:
    def __init__(self, fov, im_width, im_height):
        self.fov = fov
        self.im_width = im_width
        self.im_height = im_height
        self.intrinsics = self.get_intrinsics()

    def get_intrinsics(self):
        fov = self.fov*math.pi/180
        fx = self.im_width/(2*math.tan(fov/2)) # Conversion from fov to focal length
        fy = self.im_height/(2*math.tan(fov/2)) # Conversion from fov to focal length
        cx = self.im_width/2
        cy = self.im_height/2
        return (np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]]))

class Position:
    def __init__(self, x, y, z, yaw, pitch, roll, extrinsics, projection_matrix):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.extrinsics = extrinsics
        self.projection_matrix = projection_matrix
        self.rvec = np.array([self.yaw, self.pitch, self.roll])

class Spacecraft:
    def __init__(self, camera):
        self.camera = camera
        self.positions = []
        self.projections = []
        self.object_points = []
        self.img_filenames = []

    def add_position(self, x, y, z, yaw, pitch, roll):
        R_w_ci_intrinsic = R.from_euler('ZXZ',np.array([0,-90,0]),degrees=True).as_matrix()
        R_ci_cf_intrinsic = R.from_euler('ZXZ',np.array([yaw, pitch, 0]),degrees=True).as_matrix()
        R_c_intrinsic = np.dot(R_ci_cf_intrinsic, R_w_ci_intrinsic)
        R_w_c_extrinsic = np.linalg.inv(R_c_intrinsic)
        R_c_roll_extrinsic = R.from_euler('xyz',np.array([0,0,roll]),degrees=True).as_matrix()
        R_w_c = np.dot(R_c_roll_extrinsic,R_w_c_extrinsic)

        Tm_c = R_w_c # Rotation matrix from world/moon reference frame to camera reference frame.

        position = ([x, y, z])
        rm = np.array(list(position)) # position of camera in the moon reference frame
        rc = np.dot(Tm_c, -1*rm) # position of camera in the camera reference frame
        so3 = np.empty([3,4]) # Camera extrinsic matrix
        so3[0:3, 0:3] = Tm_c
        so3[0:3,3] = rc 

        projection_matrix = np.dot(self.camera.intrinsics, so3)
        self.positions.append(Position(x, y, z, yaw, pitch, roll, so3, projection_matrix))

    def extract_pos_from_dict(self, positions):
        for key, value in positions.items():
            self.add_position(value['x_pos'], value['y_pos'], value['z_pos'], value['yaw'], value['pitch'], value['roll'])
            if 'name' in value:
                self.img_filenames.append(value['name'])
            else:
                self.img_filenames.append(None)

    def project(self, df):

        object_points = []
        projections = []
        for position in self.positions:

            points_3D = []
            points_2D = []

            for index, row in df.iterrows():
                x = row['x_pos']
                y = row['y_pos']
                points_3D.append([x, y, 0])
                object_point = np.array([x, y, 0, 1])
                projection = np.dot(position.projection_matrix, object_point)
                projection = np.array([projection[0]/projection[2], projection[1]/projection[2]])
                points_2D.append(projection.tolist())

            object_points.append(points_3D)
            projections.append(points_2D)

        self.object_points = [np.array(pts, dtype=np.float32) for pts in object_points]
        self.projections = [np.array(pts, dtype=np.float32) for pts in projections]

    def add_noise(self, stdev):
        for i in range(len(self.projections)):
            noise = np.random.normal(0, stdev, self.projections[i].shape)
            self.projections[i] += noise

class Dataframe:
    def __init__(self):
        self.df = pd.DataFrame()

    def parse_txt(self, filename):
        df = pd.read_csv(filename, delimiter='\t', header=None)
        if df.shape[1] != 10:
            raise ValueError("The input file does not have exactly 10 columns.")
        
        df.columns = ['x_pos', 'y_pos', 'diameter', 'age', 'irregularity_mode', 'irregularity_ratio', 'replace', 'secondary', 'infill_height', 'related_boulders']

        self.df = df

    def filter_by_num(self, num):
        self.df = self.df.sort_values(by='diameter', ascending=False)
        self.df = self.df.head(num)

class GraphingTool:
    def __init__(self):
        pass

    def project_crater_centres(self, points, img_filename, num=10):
        img = Image.open(img_filename)
        fig, ax = plt.subplots()
        ax.imshow(img)

        for i in range(num):
            ax.plot(points[i][0], points[i][1], 'ro', markersize=4)

        ax.set_title('Projected Crater Centres')
        plt.show()

    def graph_error_old(self, data):
        fig, ax = plt.subplots()
        for series, num in data:
            ax.plot([x[1] for x in series], [x[0] for x in series], label=f'{num} Images')
        ax.set_xlabel('Number of imaged craters')
        ax.set_ylabel('Error (%)')
        ax.legend()
        plt.show()

    def graph_error(self, data):
        fig, ax = plt.subplots()
        for series, num in data:
            # Extract x and y values
            x_values = [x[1] for x in series]
            y_values = [x[0] for x in series]
            
            # Plot scatter points
            ax.scatter(x_values, y_values, label=f'{num} Images')
            
            # Plot connecting lines
            ax.plot(x_values, y_values)
            
        ax.set_xlabel('Number of imaged craters')
        ax.set_ylabel('Error (%)')
        ax.legend()
        ax.grid(True)  # Add grid
        plt.show()

    def graph_noise(self, data):
        fig, ax = plt.subplots()
        ax.grid(True)
        for series, num in data:
            # Extract x and y values
            x_values = [x[1] for x in series]
            y_values = [x[0] for x in series]
            
            # Plot scatter points
            ax.scatter(x_values, y_values, label=f'{num} Noise')
            
            # Plot connecting lines
            ax.plot(x_values, y_values)
            
        ax.set_xlabel('Noise (standard deviation)')
        ax.set_ylabel('Error (%)')
        plt.show()

class Calibrator:
    def __init__(self, camera):
        self.camera = camera

    def calibrate(self, object_points, projections, num_craters):
        object_points = [obj[:num_craters] for obj in object_points]
        projections = [proj[:num_craters] for proj in projections]

        flags = (cv.CALIB_FIX_K1 | cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3 |
                 cv.CALIB_FIX_K4 | cv.CALIB_FIX_K5 | cv.CALIB_FIX_K6 |
                 cv.CALIB_FIX_TANGENT_DIST)

        ret, camera_matrix, dist_coeffs, rvecs, tvecs, stdDeviationIntrinsics, stdDeviationExtrinsics, perViewErrors = cv.calibrateCameraExtended(
            object_points, projections, (self.camera.im_width, self.camera.im_height), None, None, flags=flags)
        
        calcError = np.mean(perViewErrors)

        extrinsics = [cv.Rodrigues(rvec)[0] for rvec in rvecs]
        extrinsics = [np.hstack((ext, tvec.reshape(-1, 1))) for ext, tvec in zip(extrinsics, tvecs)]
        return camera_matrix, extrinsics, calcError
    
    def calc_error(self, camera_matrix):
        # return np.linalg.norm(camera_matrix - self.camera.intrinsics, 'fro')

        # print(camera_matrix, "\n", self.camera.intrinsics)

        # Calculate the difference in the focal lengths and principal points
        f_x = np.abs(camera_matrix[0][0] - self.camera.intrinsics[0][0])
        f_y = np.abs(camera_matrix[1][1] - self.camera.intrinsics[1][1])
        c_x = np.abs(camera_matrix[0][2] - self.camera.intrinsics[0][2])
        c_y = np.abs(camera_matrix[1][2] - self.camera.intrinsics[1][2])

        # Ensure the matrices have the same shape
        if camera_matrix.shape != self.camera.intrinsics.shape:
            raise ValueError("Matrices must have the same shape to calculate error.")

        # Calculate the absolute difference
        abs_diff = np.abs(camera_matrix - self.camera.intrinsics)

        # Sum of absolute differences
        sum_abs_diff = np.sum(abs_diff)

        # Sum of absolute values of the original matrix
        sum_abs_matrix1 = np.sum(np.abs(camera_matrix))

        # Calculate the percentage difference
        percent_diff = (sum_abs_diff / sum_abs_matrix1) * 100

        return percent_diff, f_x, f_y, c_x, c_y

if __name__ == "__main__":

    # Camera intrinsic parameters
    fov = 30 # Field of view in degrees
    im_width = 512 # Image width
    im_height = 512 # Image height

    # Camera extrinsic parameters
    positions = {
        'Image_0':{
            'name': 'a0.png',
            'x_pos': 0,
            'y_pos': 0,
            'z_pos': 6000,
            'yaw': 0,
            'pitch': -90,
            'roll': 0
        },
        'Image_1':{
            'name': 'a1.png',
            'x_pos': 0,
            'y_pos': 0,
            'z_pos': 6000,
            'yaw': 0,
            'pitch': -90,
            'roll': 90
        },
        'Image_2':{
            'name': 'a2.png',
            'x_pos': 0,
            'y_pos': 0,
            'z_pos': 6000,
            'yaw': 45,
            'pitch': -90,
            'roll': 90
        },
        'Image_3':{
            'name': 'a3.png',
            'x_pos': 3199.108,
            'y_pos': 4094.672,
            'z_pos': 3000.035,
            'yaw': 322,
            'pitch': -150,
            'roll': 20
        },
        'Image_4':{
            'name': 'a4.png',
            'x_pos': -4484.241,
            'y_pos': -630.219,
            'z_pos': 3936.400,
            'yaw': 278,
            'pitch': -41,
            'roll': 20
        },
        'Image_5':{
            'name': 'a5.png',
            'x_pos': -3677.899,
            'y_pos': -1793.831,
            'z_pos': 4388.173,
            'yaw': 116,
            'pitch': -133,
            'roll': 20
        },
        'Image_6':{
            'name': 'a6.png',
            'x_pos': 3613.968,
            'y_pos': 1625.710,
            'z_pos': 3470.414,
            'yaw': 116,
            'pitch': -41,
            'roll': 20
        },
        'Image_7':{
            'name': 'a7.png',
            'x_pos': 0,
            'y_pos': 0,
            'z_pos': 6000,
            'yaw': 45,
            'pitch': -90,
            'roll': 25
        },
        'Image_8':{
            'name': 'a8.png',
            'x_pos': 0,
            'y_pos': 0,
            'z_pos': 6000,
            'yaw': 125,
            'pitch': -90,
            'roll': 120
        },
    }

    # File name for crater list
    FILENAME = 'crater_list_raw.txt'

    # Cut df to only include the first 100 craters
    NUM_CRATERS_IN_DF = 30

    # Graph sample projection
    GRAPH = False # If graphing is enabled
    GRAPH_PROJECTION = 0 # If graphing is enabled, choose which projection to graph
    GRAPH_NUM = 200 # If graphing is enabled, choose how many points to graph (< NUM_CRATERS_IN_DF)

    # Experiments to run. Add the experiment number to the list to run the experiment
    EXPERIMENTS = ['1b']

    # Create camera object
    camera = Camera(fov, im_width, im_height)
    # print('Camera matrix: \n', camera.intrinsics)

    # Create spacecraft object
    spacecraft = Spacecraft(camera)

    # Add positions of the spacecraft
    spacecraft.extract_pos_from_dict(positions)

    # Create dataframe object
    df = Dataframe()
    df.parse_txt(FILENAME)
    df.filter_by_num(NUM_CRATERS_IN_DF)

    # Project the craters
    spacecraft.project(df.df)

    # If adding a single instance of noise
    truth_values = spacecraft.projections
    
    # stdev = 5
    # spacecraft.add_noise(stdev)

    # Create graphing tool object
    graphing_tool = GraphingTool()

    # Plot the projected crater centres on image
    if GRAPH:
        graphing_tool.project_crater_centres(spacecraft.projections[GRAPH_PROJECTION], spacecraft.img_filenames[GRAPH_PROJECTION], num=min(GRAPH_NUM, NUM_CRATERS_IN_DF))

    # Create calibrator object
    calibrator = Calibrator(camera)

    '''
    Experiments section.

    This section contains all the experiments conducted for part 1 of the project. 
    Each section will explain what the experiment is testing and will produce a graph.
    If multiple experiments are run, they will be conducted in sequence. 
    '''

    # Experiment 1a: Number of imaged craters vs calibration accuracy, 3-5 images, no noise
    if '1a' in EXPERIMENTS:
        print('Experiment 1a: Number of imaged craters vs calibration accuracy, 3-5 images, no noise')
        graphing_data = []

        # For each number of images
        for i in range(len(spacecraft.object_points)-1, 4, -1):
            
            # For each number of craters
            graphing_series = []
            for j in range(8, NUM_CRATERS_IN_DF):
                camera_matrix, extrinsics, calc_error = calibrator.calibrate(spacecraft.object_points[2:i], spacecraft.projections[2:i], j)
                error, f_x, f_y, c_x, c_y = calibrator.calc_error(camera_matrix)
                graphing_series.append([error, j])
            graphing_data.append([graphing_series, i-2])
            
        graphing_tool.graph_error(graphing_data)

    # Experiment 1b: Number of imaged craters vs calibration accuracy f_x, f_y, c_x, c_y, 3 images, no noise
    if '1b' in EXPERIMENTS:
        print('Experiment 1b: Number of imaged craters vs calibration accuracy f_x, f_y, c_x, c_y, 3 images, no noise')
        graphing_series_f_x = []
        graphing_series_f_y = []
        graphing_series_c_x = []
        graphing_series_c_y = []

        for j in range(8, NUM_CRATERS_IN_DF):
            camera_matrix, extrinsics, calc_error = calibrator.calibrate(spacecraft.object_points[2:5], spacecraft.projections[2:5], j)
            error, f_x, f_y, c_x, c_y = calibrator.calc_error(camera_matrix)
            graphing_series_f_x.append([f_x, j])
            graphing_series_f_y.append([f_y, j])
            graphing_series_c_x.append([c_x, j])
            graphing_series_c_y.append([c_y, j])

        graphing_data = []
        graphing_data.append([graphing_series_f_x, 'f_x'])
        graphing_data.append([graphing_series_f_y, 'f_y'])
        graphing_data.append([graphing_series_c_x, 'c_x'])
        graphing_data.append([graphing_series_c_y, 'c_y'])

        graphing_tool.graph_error(graphing_data)
 
    # Experiment 1c: Number of imaged craters vs calibration accuracy, 3-5 images, 2 noise
    if '1c' in EXPERIMENTS:
        np.random.seed(0)
        print('Experiment 1c: Number of imaged craters vs calibration accuracy, 3-5 images, 2 noise')
        graphing_data = []
        spacecraft.add_noise(2)

        # For each number of images
        for i in range(len(spacecraft.object_points)-1, 4, -1):
            
            # For each number of craters
            graphing_series = []
            for j in range(8, NUM_CRATERS_IN_DF):
                camera_matrix, extrinsics, calc_error = calibrator.calibrate(spacecraft.object_points[2:i], spacecraft.projections[2:i], j)
                error, f_x, f_y, c_x, c_y = calibrator.calc_error(camera_matrix)
                graphing_series.append([error, j])
            graphing_data.append([graphing_series, i-2])
            
        graphing_tool.graph_error(graphing_data)
        spacecraft.projections = truth_values

    # Experiment 1d: Number of imaged craters vs calibration accuracy, 3-5 images, 5 noise
    if '1d' in EXPERIMENTS:
        np.random.seed(0)
        print('Experiment 1d: Number of imaged craters vs calibration accuracy, 3-5 images, 5 noise')
        graphing_data = []
        spacecraft.add_noise(5)

        # For each number of images
        for i in range(len(spacecraft.object_points)-1, 4, -1):
            
            # For each number of craters
            graphing_series = []
            for j in range(8, NUM_CRATERS_IN_DF):
                camera_matrix, extrinsics, calc_error = calibrator.calibrate(spacecraft.object_points[2:i], spacecraft.projections[2:i], j)
                error, f_x, f_y, c_x, c_y = calibrator.calc_error(camera_matrix)
                graphing_series.append([error, j])
            graphing_data.append([graphing_series, i-2])
            
        graphing_tool.graph_error(graphing_data)
        spacecraft.projections = truth_values

    # Experiment 1e: Noise vs calibration accuracy, 5 images, 10 craters
    if '1e' in EXPERIMENTS:
        graphing_data = []
        for i in range(len(spacecraft.object_points)-2, 4, -1):
            graphing_series = []
            for j in range(10):
                spacecraft.add_noise(j)
                camera_matrix, extrinsics, calc_error = calibrator.calibrate(spacecraft.object_points[2:i], spacecraft.projections[2:i], 10)
                error, f_x, f_y, c_x, c_y = calibrator.calc_error(camera_matrix)
                graphing_series.append([calc_error, j])
                spacecraft.projections = truth_values
            graphing_data.append([graphing_series, i-2])
            break

        graphing_tool.graph_noise(graphing_data)
        spacecraft.projections = truth_values

    # Experiment 1f: Number of imaged craters vs calibration error, 3-5 images, 2 noise
    if '1f' in EXPERIMENTS:
        np.random.seed(0)
        print('Experiment 1f: Number of imaged craters vs calibration error, 3-5 images, 2 noise')
        graphing_data = []
        spacecraft.add_noise(2)

        # For each number of images
        for i in range(len(spacecraft.object_points)-1, 4, -1):
            
            # For each number of craters
            graphing_series = []
            for j in range(8, NUM_CRATERS_IN_DF):
                camera_matrix, extrinsics, calc_error = calibrator.calibrate(spacecraft.object_points[2:i], spacecraft.projections[2:i], j)
                error, f_x, f_y, c_x, c_y = calibrator.calc_error(camera_matrix)
                graphing_series.append([calc_error, j])
            graphing_data.append([graphing_series, i-2])
            
        graphing_tool.graph_error(graphing_data)
        spacecraft.projections = truth_values

