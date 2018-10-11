__author__ = "Santiago Smith Silva"
__copyright__ = "2018"
__description__ = """
This software compares two videos (healthy and non-healty subjects) and
performs a pipeline for ocular motion processing as follows:\n\n

    1) Eye detection
    2) Calculation of the optical flow (Lucas Kenade algorithm)
    3) Plotting of the phase-space for velocity and acceleration

USAGE: just excecute as follows:
    python main.py -hv [path to the healthy video] -dv [path to the disease/condition video]

Example:
    python main.py -hv healthy.mp4 -dv disease.mp4
"""


import os
import sys
import argparse
from argparse import RawTextHelpFormatter
from os.path import join, dirname, realpath

current_dir = dirname(realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description=__description__, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-hv', #metavar='--healthy-vid',
                        help='Path for healthy video')
    parser.add_argument('-dv', #metavar='--disease-vid',
                        help='Path to non-healthy video')
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()


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
    # Get videos from args
    args = parse_args()

    # Define the list of videos:
    video_files = [
        args.hv,  # Control Video
        args.dv   # CP Video
    ]

    # =========================================
    # Start pipeline processing (DO NOT TOUCH)
    for video_file in video_files:
        # 0. Normalize path (compatibility Windows and Linux)
        video_file = os.path.normpath(video_file)

        # # 1. Eyes detection
        script = join(current_dir, '01_eyes_detection.py')
        os.system(python + ' %s %s' % (script, video_file))
        #
        # 2. Optical Flow Calculation
        script = join(current_dir, '02_optical_flow_hs.py')
        os.system(python + ' %s %s' % (script, video_file))

# Plot results
script = join(current_dir, '03_phase_plane_all_video.py')
os.system(python + ' %s %s %s' % (script, video_files[0], video_files[1]))