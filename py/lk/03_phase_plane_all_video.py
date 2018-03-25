import os
import sys
import json
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

# Define parameters
# video_file = os.path.join(os.getcwd(), 'media', '1', 'video.mp4')

# Define Root folder and set plot style
root = os.path.join(os.getcwd(), '..', '..')
plt.style.use('ggplot')


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
    print(opt_flow_folder)

    params_file = os.path.join(root, 'params', 'params.json')
    # print('Loading parameters file: %s' % params_file)

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
    video_nc = sys.argv[1]
    video_cp = sys.argv[2]

    plt.figure()

    for i, eye in enumerate(eyes):
        # Load velocity and acceleration into a DataFrame
        df_cp = get_opt_flow_data(video_cp, eye)
        df_nc = get_opt_flow_data(video_nc, eye)

        # Create grid
        hist_nc, x_edges, y_edges = np.histogram2d(df_nc['x_vel'].fillna(0), df_nc['x_accel'].fillna(0))
        var_nc = hist_nc.var()

        hist_cp, x_edges, y_edges = np.histogram2d(df_cp['x_vel'].fillna(0), df_cp['x_accel'].fillna(0))
        var_cp = hist_cp.var()

        # Create velocity and acceleration vectors
        vel_mag_nc = mag_and_phase_from_xy(df_nc['x_vel'], df_nc['y_vel'], normalized=True)[0]
        accel_mag_nc = mag_and_phase_from_xy(df_nc['x_accel'], df_nc['y_accel'], normalized=True)[0]

        vel_mag_cp = mag_and_phase_from_xy(df_cp['x_vel'], df_cp['y_vel'], normalized=True)[0]
        accel_mag_cp = mag_and_phase_from_xy(df_cp['x_accel'], df_cp['y_accel'], normalized=True)[0]

        # Plot the results
        plt.subplot(2, 2, 2 * i + 1)
        plt.scatter(vel_mag_nc, accel_mag_nc)
        if i == 0:
            plt.title('Control Patient')
            plt.ylabel('%s eye acceleration' % eye.title())
        elif i == 1:
            plt.ylabel('%s eye acceleration' % eye.title())
            plt.xlabel('Eye velocity')
        plt.ylim([-0.2, 1.2])
        plt.legend(('Var = %.2E' % var_nc,), loc='lower right')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        plt.subplot(2, 2, 2 * (i + 1))
        plt.scatter(vel_mag_cp, accel_mag_cp, color='blue')
        if i == 0:
            plt.title('CP Patient')
        elif i == 1:
            plt.xlabel('Eye velocity')

        plt.ylim([-0.2, 1])
        plt.legend(('Var = %.2E' % var_cp,), loc='lower right')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.show()
