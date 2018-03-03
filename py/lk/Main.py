import os


# Define the root folder (DO NOT TOUCH)
root = os.path.join(os.getcwd(), '..', '..')


# MAIN FUNCTION
if __name__ == '__main__':
    # Define the list of videos:
    video_files = [
        '/home/sssilvar/Documents/Dataset/jpgonzalezh/test/NC/SMOOTH/GOPR0215.MP4',  # Control Video
        '/home/sssilvar/Documents/Dataset/jpgonzalezh/test/CP/SMOOTH/GOPR0312.MP4'   # CP Video
    ]

    # =========================================
    # Start pipeline processing (DO NOT TOUCH)
    for video_file in video_files:
        # 0. Normalize path (compatibility Windows and Linux)
        video_file = os.path.normpath(video_file)

        # 1. Eyes detection
        os.system('python2.7 01_eyes_detection.py %s' % video_file)

        # 2. Optical Flow Calculation
        os.system('python2.7 02_optical_flow_hs.py %s' % video_file)

# Plot results
os.system('python2.7 03_phase_plane.py %s %s' % (video_files[0], video_files[1]))