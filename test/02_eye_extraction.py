"""
EYE EXTRACTION
@Description: This program segments eyes in a frontal video. Saves as results:
- left_eye.jpg  : Left eye segmentation
- right_eye.jpg : Right eye segmentation
- eyes_detection.json : File that contains the coordinates of the eye segmentation in the video
"""
import os
import cv2
import json
import shutil
import matplotlib.pyplot as plt


# Define parameters
video_file = os.path.join('C:\Users\Smith\Videos\pepe\GOPR0305.MP4')
# video_file = os.path.join(os.getcwd(), 'media', 'GOPR0216.mp4')
folder_output = os.path.join(os.path.dirname(video_file), 'eye_detection')

root = os.path.join(os.getcwd(), '..')

# Check and create folder
if not os.path.exists(folder_output):
    os.makedirs(folder_output)
else:
    shutil.rmtree(folder_output)
    os.makedirs(folder_output)

# Create a capture object (from video)
cap = cv2.VideoCapture(video_file)

# Create a Haar cascade
eye_cascade_xml = os.path.join(root, 'lib', 'haarcascades', 'haarcascade_eye.xml')
eye_cascade = cv2.CascadeClassifier(eye_cascade_xml)

# JSON filename load: eyes_file
with open(os.path.join(root, 'params', 'params.json')) as json_file:
    jf = json.load(json_file)
    eyes_file = jf['eye_detection_json']

# Start processing!
while cap.isOpened():
    print('[  OK  ] Checking for frames')
    ret, frame = cap.read()
    if ret:
        print('[  OK  ] Frame found!')
        # Convert frame to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        xt, yt, wt, ht = ([], [], [], [])

        for i, (x, y, w, h) in enumerate(eyes):
            print(i)
            xt.append(x)
            yt.append(y)
            wt.append(w)
            ht.append(h)

            frame = cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 255, 0), 2)

            # Conditions for eye selection
            # 1. Eye vertical separation less than 20% of the video height
            # 2. Ordinate (y) length greater than 20% of the video height
            conditions = i == 1 and \
                         abs(yt[0] - yt[1]) < 0.2 * cap.get(cv2.CAP_PROP_FRAME_HEIGHT) < yt[0] + ht[0] and \
                         yt[1] + ht[1] > 0.2 * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            if conditions:
                print('[  OK  ] Extracting ROI')

                # Detect eye side (left and right)
                if xt[0] < xt[1]:
                    data = {
                        'left_eye': {
                            'x_min': xt[1],
                            'x_max': xt[1] + wt[1],
                            'y_min': yt[1],
                            'y_max': yt[1] + ht[1]
                        },
                        'right_eye': {
                            'x_min': xt[0],
                            'x_max': xt[0] + wt[0],
                            'y_min': yt[0],
                            'y_max': yt[0] + ht[0]
                        }
                    }

                    # Extract eye data (images): left_eye.jpg / right_eye.jpg
                    left_eye = gray[
                        data['left_eye']['y_min']: data['left_eye']['y_max'],  # X ROI boundary
                        data['left_eye']['x_min']: data['left_eye']['x_max']   # Y ROI boundary
                    ]
                    right_eye = gray[
                        data['right_eye']['y_min']: data['left_eye']['y_max'],  # X ROI boundary
                        data['right_eye']['x_min']: data['right_eye']['x_max']  # Y ROI boundary
                    ]
                elif xt[0] > xt[1]:
                    data = {
                        'left_eye': {
                            'x_min': xt[0],
                            'x_max': xt[0] + wt[0],
                            'y_min': yt[0],
                            'y_max': yt[0] + ht[0]
                        },
                        'right_eye': {
                            'x_min': xt[1],
                            'x_max': xt[1] + wt[1],
                            'y_min': yt[1],
                            'y_max': yt[1] + ht[1]
                        }
                    }

                    # Extract eye data (images): left_eye.jpg / right_eye.jpg
                    # Remember: index have to be passed inverted (x and y)
                    left_eye = gray[
                        data['left_eye']['y_min']: data['left_eye']['y_max'],  # X ROI boundary
                        data['left_eye']['x_min']: data['left_eye']['x_max']  # Y ROI boundary
                    ]
                    right_eye = gray[
                        data['right_eye']['y_min']: data['left_eye']['y_max'],  # X ROI boundary
                        data['right_eye']['x_min']: data['right_eye']['x_max']  # Y ROI boundary
                    ]

    try:
            # Visualise image
            frame_start = cap.get(cv2.CAP_PROP_POS_FRAMES)
            data['frame_start'] = frame_start
            print('[  INFO  ] Eyes found at frame: %d' % frame_start)
            print('[  SUCCESS  ] showing')

            cv2.imwrite(os.path.join(folder_output, 'left_eye.jpg'), left_eye)
            cv2.imwrite(os.path.join(folder_output, 'right_eye.jpg'), right_eye)

            # Save coordinates file (see params.json)
            with open(os.path.join(folder_output, eyes_file), 'w') as file_out:
                json.dump(data, file_out)

            # Exit
            break
    except Exception:
        print('[  ERROR  ] Cannot show or eyes not detected')

    # Check for interruption
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close and release everything
cap.release()
cv2.destroyAllWindows()

# Show if you want
plt.style.use('ggplot')

plt.subplot(1, 2, 1)
plt.imshow(right_eye, cmap='gray')
plt.axis('on')
plt.grid('on')
plt.title('Right eye')

plt.subplot(1, 2, 2)
plt.imshow(left_eye, cmap='gray')
plt.axis('on')
plt.grid('on')
plt.title('Left eye')

plt.show()
