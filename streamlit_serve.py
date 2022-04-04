import os
import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import requests
import cv2
import matplotlib.pyplot as plt

img_ht = 50
img_wd = 50

st.write('Capstone Streamlit App')

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

if __name__ == '__main__':
    if st.checkbox('Select a file in current directory'):
        folder_path = '.\imgs\streamtest'
        if st.checkbox('Change directory'):
            folder_path = st.text_input('Enter folder path', '.\imgs\streamtest')
        filename = file_selector(folder_path=folder_path)
        st.write(f'Selected {filename}')

img_load = (filename)
test_img = keras.preprocessing.image.load_img(img_load, target_size=(img_ht, img_wd))
test_img_arr = keras.preprocessing.image.img_to_array(test_img)
pred_img = test_img_arr / 255
pred_img = np.expand_dims(pred_img, axis=0)
user_input = (np.array(pred_img).tolist())

response = requests.post('http://127.0.0.1:8080/predict', json=user_input)
response.json()

st_pt = (response.json()['response'][1], response.json()['response'][2])
e_pt = (response.json()['response'][3], response.json()['response'][4])
img_final = cv2.rectangle(pred_img[0], st_pt, e_pt, (0, 255, 0), 1)
plt.xlabel(f"{response.json()['response'][0]}", color= 'g')
plt.imshow(img_final)
plt.savefig(f'{folder_path}'+'/'+'processed.png', dpi=400)
processed_img = (folder_path + '/' + 'processed.png')

st.image(processed_img)