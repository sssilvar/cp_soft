import matplotlib.pyplot as plt
from lib.eyes_extraction import eye_extraction

if __name__ == '__main__':
    video_file = 'C:/Users/sssilvar/Videos/cp_soft/smooth/GOPR0229.MP4'
    eye_extraction(video_file, show_plot=True)
    plt.show()
