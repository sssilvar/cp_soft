import os
import cv2


# Define parameters
# video_file = os.path.join(os.getcwd(), 'media', '1', 'video.mp4')
video_file = 'C:/Users/sssilvar/Videos/cp_soft/smooth/GOPR0229.MP4'

# Create a capture object (from video)
cap = cv2.VideoCapture(video_file)

# Create a Haar cascade
eye_cascade_xml = os.path.join(os.getcwd(), '..', 'lib', 'haarcascades', 'haarcascade_eye.xml')
eye_cascade = cv2.CascadeClassifier(eye_cascade_xml)

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
        for (x, y, w, h) in eyes:
            frame = cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 255, 0), 2)

        # Vizualise image
        cv2.imshow('Eye Segmentation', frame)

    # Check for interruption
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close and release everything
cap.release()
cv2.destroyAllWindows()
