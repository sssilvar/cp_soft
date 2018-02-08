import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define parameters
video_file = os.path.join(os.getcwd(), 'media', 'GOPR0216.mp4')
opt_flow_folder = os.path.join(os.path.dirname(video_file), 'optical_flow')

json_filename = os.path.join(os.path.dirname(video_file), 'eye_detection', 'eyes_detection.json')

root = os.path.join(os.getcwd(), '..')
params_file = os.path.join(root, 'params', 'params.json')
print(params_file)
# END OF PARAMETERS - DO NOT TOUCH FROM HERE!

# Set plt style
plt.style.use('ggplot')

# Load params file
with open(params_file, 'r') as json_file:
    jf = json.load(json_file)
    fps = jf['camera']['fps']
    opt_flow_csv = jf['opt_flow']['csv_file']

# Load DataFrame
df = pd.read_csv(os.path.join(opt_flow_folder, opt_flow_csv))
df = df.dropna()

print(df.head(3))
print(df.tail(3))

df['x_vel'].plot(kind='line')
plt.show()
