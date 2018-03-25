import sys

import json
import os
import shutil
import sys

import cv2

# Define the root folder
root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

sys.path.append(os.path.join(root))
from lib.eyes_extraction import eye_extraction


def check_folder(folder):
    """
    This function checks if a folder exists, and deletes the folder if it does
    :param folder: path to the folder to be checked
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)


# MAIN FUNCTION
if __name__ == '__main__':
    arg = sys.argv[1]
    print('[  OK  ] Eye extraction for: %s' % arg)
    eye_extraction(arg)
