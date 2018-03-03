import os


# Define the root folder
root = os.path.join(os.getcwd(), '..', '..')

# MAIN FUNCTION
if __name__ == '__main__':
    # Define the list of videos:
    video_file = video_cp = os.path.join(root, 'test', 'media', '1', 'video.mp4')

    # 1. Eyes detection
    os.system('python2.7 01_eyes_detection.py %s' % video_file)

    # 2. Optical Flow Calculation
    os.system('python2.7 02_optical_flow_hs.py %s' % video_file)
