import os
import json
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define parameters
# video_file = os.path.join(os.getcwd(), 'media', '1', 'video.mp4')

# Define Root folder and set plot style
root = os.path.join(os.getcwd(), '..')
plt.style.use('ggplot')


def get_opt_flow_data(video_file, eye):
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
    # Set filename per each subject
    video_cp = os.path.join(os.getcwd(), 'media', '1', 'video.mp4')
    video_nc = os.path.join(os.getcwd(), 'media', '0', 'video.mp4')

    # Load velocity and acceleration into a DataFrame
    df_cp = get_opt_flow_data(video_cp, 'left')
    df_nc = get_opt_flow_data(video_nc, 'left')

    # Create grid
    hist_nc, x_edges, y_edges = np.histogram2d(df_nc['x_vel'].fillna(0), df_nc['x_accel'].fillna(0))
    var_nc = hist_nc.var()

    hist_cp, x_edges, y_edges = np.histogram2d(df_cp['x_vel'].fillna(0), df_cp['x_accel'].fillna(0))
    var_cp = hist_cp.var()

    # Plot the results
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Control Patient (var = %d)' % var_nc)
    plt.scatter(df_nc['x_vel'], df_nc['x_accel'])
    plt.xlabel('Velocity')
    plt.ylabel('Acceleration')
    plt.ylim([-2e6, 2e6])

    plt.subplot(1, 2, 2)
    plt.title('CP Patient (var = %d)' % var_cp)
    plt.scatter(df_cp['x_vel'], df_cp['x_accel'], color='blue')
    plt.xlabel('Velocity')
    plt.ylabel('Acceleration')
    plt.ylim([-2e6, 2e6])

    plt.show(block=False)
