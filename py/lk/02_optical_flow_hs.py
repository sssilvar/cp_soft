"""
OPTICAL FLOW
@Description: This program calculates the optical flow for a region of interest (ROI)
:param video_file:
:param eye: 'right_eye' or 'left_eye'

"""

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

# Set root folder
root = os.path.join(os.getcwd(), '..', '..')


def check_folder(folder):
    """
    This function checks if a folder exists, and deletes the folder if it does
    :param folder: path to the folder to be checked
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


def opt_flow(video_file, eye, visualize=True):
    """
    This function performs an optical flow calculation based on the Lucas-Kenade algorithm.
    It saves a file called "left_eye_optical_flow.csv" / "right_eye_optical_flow.csv" inside a folder
    called "optical_flow" in the video path.
    :param video_file: :type str: path to the video
    :param eye: 'left' or 'right'
    """

    # Check eye parameter
    if eye == 'left':
        eye = 'left_eye'
    elif eye == 'right':
        eye = 'right_eye'
    else:
        print('[eye] parameter is not correct. It must be "left" or "right"')
        raise ValueError

    # Define the folder output
    folder_output = os.path.join(os.path.dirname(video_file), 'optical_flow')

    json_filename = os.path.join(os.path.dirname(video_file), 'eyes_detection', 'eyes_detection.json')
    params_file = os.path.join(root, 'params', 'params.json')

    # Check and create folder
    check_folder(folder_output)

    # Load params file
    with open(params_file, 'r') as json_file:
        jf = json.load(json_file)
        fps = jf['camera']['fps']
        opt_flow_csv = jf['opt_flow']['csv_file']

    # Load ROI
    with open(json_filename, 'r') as json_file:
        jf = json.load(json_file)

        # Load boundaries
        roi_x_min = jf[eye]['x_min']
        roi_x_max = jf[eye]['x_max']
        roi_y_min = jf[eye]['y_min']
        roi_y_max = jf[eye]['y_max']

        frame_start = jf['frame_start']

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.05,
                          minDistance=2,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(10, 10),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Define a capture
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start + 100)
    print(cap.get(cv2.CAP_PROP_POS_FRAMES))

    print('[  INFO  ] Checking first frame')
    ret = False
    while not ret:
        print(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(ret)
        ret, first_frame = cap.read()

    # Take first frame and find corners in it: first_frame
    # Extract ROI and replace: first frame
    # Convert ROI to gray scale: roi_ff_gray
    # Define a point of start: p0
    first_frame = first_frame[roi_y_min: roi_y_max, roi_x_min: roi_x_max]
    roi_ff_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(roi_ff_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(first_frame)

    # Define a feature vector
    features = []

    while cap.isOpened():
        # cap = cv2.VideoCapture(video_file)
        ret, frame = cap.read()

        # Get current frame
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print('Loading frame: %d' % current_frame)

        # Start optical flow
        if ret:

            # Extract ROI and replace it in :frame
            frame = frame[roi_y_min: roi_y_max, roi_x_min: roi_x_max]

            # Convert to gray: frame grTypeError: 889 is not JSON serializableTypeError: 889 is not JSON serializableay
            roi_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(roi_ff_gray, roi_frame_gray, p0, None, **lk_params)

            # Select good points
            try:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # draw the tracks
                n_features = 0
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                    n_features += 1
                    features.append([current_frame, i, a, b])
            except TypeError:
                print('Skipping frame (no features found)')

            if visualize:
                img = cv2.add(frame, mask)
                cv2.imshow(eye, img)

        # If there is no frame, finish
        elif current_frame < cap.get(cv2.CAP_PROP_FRAME_COUNT):
            pass
        else:
            break

        # Wait for exit command
        k = cv2.waitKey(30) & 0xff == ord('q')
        if k == 27:
            break

        # Now update the previous frame and previous points
        roi_ff_gray = roi_frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    cap.release()

    position_df = pd.DataFrame(features, columns=['frame', 'feature_id', 'x_pos', 'y_pos'])
    position_df['x_vel'] = position_df['x_pos'].diff() * fps
    position_df['y_vel'] = position_df['y_pos'].diff() * fps
    position_df['x_accel'] = position_df['x_vel'].diff() * fps
    position_df['y_accel'] = position_df['y_vel'].diff() * fps

    print('[  OK  ] Saving file %s' % opt_flow_csv)
    position_df.to_csv(os.path.join(folder_output, eye + '_' + opt_flow_csv))

    print('Done!\n\t Total frames processed: %d\n\t Total features extracted: %d'
          % (position_df['frame'].max() - frame_start, position_df['feature_id'].max()))


# MAIN FUNCTION
if __name__ == '__main__':
    arg = sys.argv[1]
    print('[  OK  ] Calculating optical flow for: %s' % arg)

    opt_flow(arg, 'left', visualize=True)
    opt_flow(arg, 'right', visualize=True)
