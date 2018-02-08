"""
OPTICAL FLOW
@Description: This program calculates the optical flow for a region of interest (ROI)

"""

import os
import cv2
import json
import shutil
import pandas as pd
import numpy as np


# Define parameters
video_file = os.path.join(os.getcwd(), 'media', 'GOPR0216.mp4')
folder_output = os.path.join(os.path.dirname(video_file), 'optical_flow')

json_filename = os.path.join(os.path.dirname(video_file), 'eye_detection', 'eyes_detection.json')
root = os.path.join(os.getcwd(), '..')
params_file = os.path.join(root, 'params', 'params.json')
print(params_file)
# END OF PARAMETERS - DO NOT TOUCH FROM HERE!

# Check and create folder
if not os.path.exists(folder_output):
    os.makedirs(folder_output)
else:
    shutil.rmtree(folder_output)
    os.makedirs(folder_output)

# Load params file
with open(params_file, 'r') as json_file:
    jf = json.load(json_file)
    fps = jf['camera']['fps']
    opt_flow_csv = jf['opt_flow']['csv_file']


# Load ROI
with open(json_filename, 'r') as json_file:
    jf = json.load(json_file)

    # Load boundaries
    eye = 'right_eye'
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

# Initialize a position vector
position = []

while cap.isOpened():
    # cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    # Start optical flow
    if ret:
        # Get current frame
        print('Loading frame: %d' % cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Extract ROI and replace it in :frame
        frame = frame[roi_y_min: roi_y_max, roi_x_min: roi_x_max]

        # Convert to gray: frame grTypeError: 889 is not JSON serializableTypeError: 889 is not JSON serializableay
        roi_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(roi_ff_gray, roi_frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        position.append([cap.get(cv2.CAP_PROP_POS_FRAMES), a, b])

    # If there is no frame, finish
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

position_df = pd.DataFrame(position, columns=['frame', 'x_pos', 'y_pos'])
position_df['x_vel'] = position_df['x_pos'].diff() * fps
position_df['y_vel'] = position_df['y_pos'].diff() * fps
position_df.to_csv(os.path.join(folder_output, opt_flow_csv))
