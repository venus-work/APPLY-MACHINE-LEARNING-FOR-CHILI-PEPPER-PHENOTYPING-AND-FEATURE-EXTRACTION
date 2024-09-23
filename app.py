import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
import subprocess
import glob

st.title('Extracting Phenotying Feature of Chili pepper')
# Tạo thư mục tạm thời nếu chưa tồn tại
temp_dir = 'tempDir'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

source_file = st.file_uploader(
    'Upload your source image:', type=['jpg', 'jpeg', 'png'])
if source_file is not None:
    st.image(source_file, caption='Uploaded Image.', use_column_width=True)
    source_path = os.path.join('tempDir', source_file.name)
    with open(source_path, 'wb') as f:
        f.write(source_file.getbuffer())
    st.success(f'Source file saved at {source_path}')
else:
    st.write("Please upload an image file")

# Nút để thực hiện lệnh
output_txt_path = ""
if st.button('Run Detection and Show csv'):
    command = ['python', 'my_detect.py', '--weights', 'best.pt', '--source',
               source_path, '--save-txt', '--save-conf', '--save-csv', '--save-img-para']
    result = subprocess.run(command, capture_output=True, text=True)
    output_dir = max(glob.glob('runs/detect/exp*'), key=os.path.getmtime)
    name_source_file = source_file.name.split('.')[0]
    output_image_path = os.path.join(
        output_dir, name_source_file, name_source_file + ".jpg")
    output_csv_path = os.path.join(output_dir, "labels", "results.csv")

    output_image_path = output_image_path.replace('\\', '/')
    output_csv_path = output_csv_path.replace('\\', '/')
    if os.path.exists(output_image_path):
        st.image(output_image_path, caption='Detected Image',
                 use_column_width=True)
    else:
        st.error(f'Output image {output_image_path} not found')

    st.write("Data from csv")
    if os.path.exists(output_csv_path):
        df = pd.read_csv(output_csv_path)
        st.dataframe(df)
    else:
        st.error(f'Output csv file {output_csv_path} not found')
