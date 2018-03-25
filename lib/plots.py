import os

import numpy as np
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D


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


def hist3d_whole_video(x, y):
    """
    Plots a 3D histogram from an x,y data
    :param x: Vector of data for x-axis
    :param y: Vector of data for y-axis
    :return: 3D histogram plot (needs plt.show() to work)
    """

    data, _, _ = np.histogram2d(x, y, bins=50)

    fig = plt.figure()
    ax = Axes3D(fig)

    lx = len(data[0])  # Work out matrix dimensions
    ly = len(data[:, 0])
    xpos = np.arange(0, lx, 1)  # Set up a mesh of positions
    ypos = np.arange(0, ly, 1)
    xpos, ypos = np.meshgrid(xpos + 0.5, ypos + 0.5)

    xpos = xpos.flatten()  # Convert positions to 1D array
    ypos = ypos.flatten()
    zpos = np.ones(lx * ly) * 1e-10

    dx = 1. * np.ones_like(zpos)
    dy = dx.copy()
    dz = data.flatten()

    # Set colormap
    offset = dz + np.abs(dz.min())
    fracs = offset.astype(float) / offset.max()
    norm = col.Normalize(fracs.min(), fracs.max())
    colors = cm.jet(norm(fracs))

    ax.bar3d(xpos, ypos, zpos, 1, 1, dz, color=colors)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=1., zsort='max')
    # plt.ion()


def hist2d_whole_video(video_files):
    plt.style.use('ggplot')
    fig, axarr = plt.subplots(len(video_files), 2, sharex=True, sharey=True)

    for i, video_file in enumerate(video_files):
        for j in range(0, 2):
            if j == 0:
                df = pd.read_csv(os.path.join(os.path.dirname(video_file), 'optical_flow', 'left_eye_optical_flow.csv'))
            elif j == 1:
                df = pd.read_csv(os.path.join(os.path.dirname(video_file), 'optical_flow', 'right_eye_optical_flow.csv'))

            x = mag_and_phase_from_xy(df['x_vel'], df['y_vel'], normalized=True)[1]
            y = mag_and_phase_from_xy(df['x_accel'], df['y_accel'], normalized=True)[1]
            axarr[i, j].hist2d(x, y, bins=50, cmap=plt.cm.jet)


if __name__ == '__main__':
    # test histogram
    video_files = [
        'C:/Users/sssilvar/Videos/cp_soft/smooth/control/GOPR0229.MP4',  # Control Video
        'C:/Users/sssilvar/Videos/cp_soft/smooth/cp/GOPR0335.MP4'  # CP Video
    ]

    # df = pd.read_csv(os.path.join(os.path.dirname(video_file), 'optical_flow', 'left_eye_optical_flow.csv'))
    #
    # x = df['x_vel'].fillna(0)
    # y = df['x_accel'].fillna(0)

    hist2d_whole_video(video_files)

    # plt.style.use('ggplot')
    # # hist3d_whole_video(x, y)
    #
    # plt.hist2d(x, y, bins=50, cmap=plt.cm.jet)
    # plt.colorbar()
    plt.show()
