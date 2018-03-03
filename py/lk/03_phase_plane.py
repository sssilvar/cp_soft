import os
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

    params_file = os.path.join(root, 'params', 'params.json')
    print('Loading parameters file: %s' % params_file)

    # Load params file
    with open(params_file, 'r') as json_file:
        jf = json.load(json_file)
        fps = jf['camera']['fps']
        opt_flow_csv = eye + jf['opt_flow']['csv_file']

    return pd.read_csv(os.path.join(opt_flow_folder, opt_flow_csv))


if __name__ == '__main__':
    # Eye
    eyes = ['left', 'right']

    # Set filename per each subject
    video_cp = os.path.join(root, 'test', 'media', '1', 'video.mp4')
    video_nc = os.path.join(root, 'test', 'media', '0', 'video.mp4')

    plt.figure()

    for i, eye in enumerate(eyes):
        # Load velocity and acceleration into a DataFrame
        df_cp = get_opt_flow_data(video_cp, eye)
        df_nc = get_opt_flow_data(video_nc, eye)

        # Create grid
        hist_nc, x_edges, y_edges = np.histogram2d(df_nc['x_vel'].fillna(0), df_nc['x_accel'].fillna(0), normed=True, bins=20)
        var_nc = hist_nc.var()

        hist_cp, x_edges, y_edges = np.histogram2d(df_cp['x_vel'].fillna(0), df_cp['x_accel'].fillna(0), normed=True, bins=20)
        var_cp = hist_cp.var()

        # Create velocity and acceleration vectors
        vel_mag_cp = np.sqrt(df_nc['x_vel'] ** 2 + df_nc['y_vel'] ** 2).fillna(0)
        accel_mag_cp = np.sqrt(df_nc['x_accel'] ** 2 + df_nc['y_accel'] ** 2).fillna(0)

        vel_mag_nc = np.sqrt(df_cp['x_vel'] ** 2 + df_cp['y_vel'] ** 2).fillna(0)
        accel_mag_nc = np.sqrt(df_cp['x_accel'] ** 2 + df_cp['y_accel'] ** 2).fillna(0)

        # Plot the results
        # plt.subplot(2, 2, 2*i+1)
        # plt.scatter(vel_mag_nc, accel_mag_nc)
        # if i == 0:
        #     plt.title('Control Patient')
        #     plt.ylabel('%s eye acceleration' % eye.title())
        # elif i == 1:
        #     plt.ylabel('%s eye acceleration' % eye.title())
        #     plt.xlabel('Eye velocity')
        # plt.ylim([-2e6, 2e6])
        # plt.legend(('Var = %.2E' % var_nc,), loc='lower right')
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        #
        # plt.subplot(2, 2, 2*(i+1))
        # plt.scatter(vel_mag_cp, accel_mag_cp, color='blue')
        # if i == 0:
        #     plt.title('CP Patient')
        # elif i == 1:
        #     plt.xlabel('Eye velocity')
        #
        # plt.ylim([-2e6, 2e6])
        # plt.legend(('Var = %.2E' % var_cp,), loc='lower right')
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        # Plot histograms
        bins = 25
        normed = False

        if i == 0:
            f, axarr = plt.subplots(2, 2, sharex=True)
            axarr[0, 0].set_title('Control Patient')
            axarr[0, 0].legend(('Var = %.2E' % var_nc,), loc='lower right')
            axarr[0, 0].set_ylabel('%s eye acceleration' % eye.title())
            axarr[0, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            axarr[0, 0].hist2d(vel_mag_nc, accel_mag_nc, bins=bins, norm=LogNorm(), normed=normed)

            axarr[0, 1].set_title('CP Patient')
            axarr[0, 1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            axarr[0, 1].hist2d(vel_mag_cp, accel_mag_cp, bins=bins, norm=LogNorm(), normed=normed)
        elif i == 1:
            axarr[1, 0].set_ylabel('%s eye acceleration' % eye.title())
            axarr[1, 0].set_xlabel('Velocity')
            axarr[1, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            axarr[1, 0].hist2d(vel_mag_nc, accel_mag_nc, bins=bins, norm=LogNorm(), normed=normed)

            axarr[1, 1].set_xlabel('Velocity')
            axarr[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            axarr[1, 1].hist2d(vel_mag_cp, accel_mag_cp, bins=bins, norm=LogNorm(), normed=normed)

        # if i == 0:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     x, y = np.random.rand(2, 100) * 4
        #     hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])
        #
        #     # Construct arrays for the anchor positions of the 16 bars.
        #     # Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
        #     # ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
        #     # with indexing='ij'.
        #     xpos, ypos = np.meshgrid(x_edges[:-1] + 0.25, y_edges[:-1] + 0.25)
        #     xpos = xpos.flatten('F')
        #     ypos = ypos.flatten('F')
        #     zpos = np.zeros_like(xpos)
        #
        #     # Construct arrays with the dimensions for the 16 bars.
        #     dx = 0.5 * np.ones_like(zpos)
        #     dy = dx.copy()
        #     dz = hist_nc.flatten()
        #
        #     ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
        #
        # elif i == 1:
        #     pass

    plt.show()
