import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define parameters
# video_file = os.path.join(os.getcwd(), 'media', '1', 'video.mp4')

# Define Root folder and set plot style
root = os.path.join(os.getcwd(), '..', '..')
plt.style.use('ggplot')


def check_folder(folder):
    """
    This function checks if a folder exists, and deletes the folder if it does
    :param folder: path to the folder to be checked
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)


def get_opt_flow_data(video_file, eye):
    """
    Loads the *.csv file generated from the optical flow analysis of one eye.
    :param video_file: path to the video
    :param eye: 'left' or 'right'
    :return: DataFrame with data obtained from the optical flow (pos, vel, accel)
    """

    # Check eye
    if eye == 'left':
        eye = 'left_eye_'
    elif eye == 'right':
        eye = 'right_eye_'

    # Set Optical flow folder directory
    opt_flow_folder = os.path.join(os.path.dirname(video_file), 'optical_flow')

    params_file = os.path.join(root, 'params', 'params.json')
    print('Loading parameters file: %s' % params_file)

    # Load params file
    with open(params_file, 'r') as json_file:
        jf = json.load(json_file)
        fps = jf['camera']['fps']
        opt_flow_csv = eye + jf['opt_flow']['csv_file']

    return pd.read_csv(os.path.join(opt_flow_folder, opt_flow_csv))


def mag_and_phase_from_xy(x, y, normalized=False):
    mag = np.sqrt(x ** 2 + y ** 2)
    phase = np.arctan(y / x)

    # Clean data
    mag = mag.fillna(0)
    phase = phase.fillna(0)

    if normalized:
        mag = mag / np.abs(mag.max())
        phase = phase / np.abs(phase.max())

    return mag, phase


if __name__ == '__main__':

    # Eye
    eyes = ['left', 'right']

    # Set filename per each subject
    video_cp = os.path.join(root, 'test', 'media', '1', 'video.mp4')
    video_nc = os.path.join(root, 'test', 'media', '0', 'video.mp4')
    # video_nc = sys.argv[1]
    # video_cp = sys.argv[2]

    for i, eye in enumerate(eyes):
        # Load velocity and acceleration into a DataFrame
        df_cp = get_opt_flow_data(video_cp, eye)
        df_nc = get_opt_flow_data(video_nc, eye)

        # Define the range of frames to be analysed
        # (for controls nc_* and for cerebral palsy cp_*)
        nc_initial_frame = df_nc['frame'].min().astype(int)
        nc_final_frame = df_nc['frame'].max().astype(int)
        nc_frames = range(nc_initial_frame, nc_final_frame + 1)

        cp_initial_frame = df_cp['frame'].min().astype(int)
        cp_final_frame = df_cp['frame'].max().astype(int)
        cp_frames = range(cp_initial_frame, cp_final_frame + 1)

        print('NC INFO: %s Eye analysis from frame %d to frame %d' % (eye.title(), nc_initial_frame, nc_final_frame))
        print('CP INFO: %s Eye analysis from frame %d to frame %d' % (eye.title(), cp_initial_frame, cp_final_frame))

        # Create folders
        nc_folder = os.path.join(os.path.dirname(video_nc), 'phase_planes_per_frame', eye + '_eye')
        cp_folder = os.path.join(os.path.dirname(video_cp), 'phase_planes_per_frame', eye + '_eye')
        check_folder(nc_folder)
        check_folder(cp_folder)

        # Create velocity and acceleration vectors
        vel_mag_nc = mag_and_phase_from_xy(df_nc['x_vel'], df_nc['y_vel'], normalized=True)[0]
        accel_mag_nc = mag_and_phase_from_xy(df_nc['x_accel'], df_nc['y_accel'], normalized=True)[0]

        vel_mag_cp = mag_and_phase_from_xy(df_cp['x_vel'], df_cp['y_vel'], normalized=True)[0]
        accel_mag_cp = mag_and_phase_from_xy(df_cp['x_accel'], df_cp['y_accel'], normalized=True)[0]

        # Draw 3D
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.set_title('Control Patient (%s eye)' % eye.title())
        ax.scatter(vel_mag_nc, accel_mag_nc, df_nc['frame'], marker='o')
        ax.set_xlabel('Velocity')
        ax.set_ylabel('Acceleration')
        ax.set_zlabel('Time')

        ax = fig.add_subplot(122, projection='3d')
        ax.set_title('CP Patient (%s eye)' % eye.title())
        ax.scatter(vel_mag_cp, accel_mag_cp, df_cp['frame'], c='b', marker='^')
        ax.set_xlabel('Velocity')
        ax.set_ylabel('Acceleration')
        ax.set_zlabel('Time')

        plt.show()

        # Draw phase planes per frame
        # seconds = 5
        # df = df_cp
        # index = np.logical_and(df['frame'].astype(int) >= cp_initial_frame, df['frame'].astype(int) <= cp_initial_frame + seconds * 120)
        # vel_data = vel_mag_cp[index]
        # accel_data = accel_mag_cp[index]
        #
        # plt.subplot(1, 2, 1)
        # plt.title('Info during %d seconds' % seconds )
        # plt.scatter(vel_data, accel_data)
        # plt.show()




