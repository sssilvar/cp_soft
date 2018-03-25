import os


# Define the root folder (DO NOT TOUCH)
root = os.path.join(os.getcwd(), '..', '..')

# Check operating System
if os.name == 'nt':
    # python = os.path.normpath('C:/Users/' + os.environ['USERNAME'] + '/Anaconda2/python.exe')
    python = 'C:/Users/sssilvar/AppData/Local/Programs/Python/Python36/python.exe'
elif os.name == 'Linux':
    python = 'python 2.7'
else:
    python = 'python'


# MAIN FUNCTION
if __name__ == '__main__':
    # Define the list of videos:
    video_files = [
        'C:/Users/sssilvar/Videos/cp_soft/smooth/control/GOPR0229.MP4',  # Control Video
        'C:/Users/sssilvar/Videos/cp_soft/smooth/cp/GOPR0335.MP4'   # CP Video
    ]

    # =========================================
    # Start pipeline processing (DO NOT TOUCH)
    for video_file in video_files:
        # 0. Normalize path (compatibility Windows and Linux)
        video_file = os.path.normpath(video_file)

        # # 1. Eyes detection
        # os.system(python + ' 01_eyes_detection.py %s' % video_file)
        #
        # # 2. Optical Flow Calculation
        # os.system(python + ' 02_optical_flow_hs.py %s' % video_file)

# Plot results
os.system(python + ' 03_phase_plane_all_video.py %s %s' % (video_files[0], video_files[1]))