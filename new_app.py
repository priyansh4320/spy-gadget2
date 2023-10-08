import os
import cv2
import argparse
import filetype as ft
import numpy as np
from pathlib import Path
from PIL import Image
from facedetector import FaceDetector
import base64
from io import BytesIO
import pandas as pd

import streamlit as st


import requests

## define face extraction--------------------------------------------------------------------------

def face_extraction(uploaded_video):
    video = cv2.VideoCapture(uploaded_video)

    # Create a folder to store the extracted faces.
    if not os.path.exists('faces'):
        os.mkdir('faces')

    # Iterate through the frames of the video.
    images= []
    padding = 1.0
    while True:

        # Read the frame.
        ret, frame = video.read()
        if ret and isinstance(frame, np.ndarray):
            image = {
                "file": frame,
                "sourcePath": video,
                "sourceType": "video",
            }
            images.append(image)
        else:
            break
    video.release()
    cv2.destroyAllWindows()
    for (i, image) in enumerate(images):
        print("[INFO] processing image {}/{}".format(i + 1, len(images)))
        faces = FaceDetector.detect(image["file"])

        array = cv2.cvtColor(image['file'], cv2.COLOR_BGR2RGB)
        img = Image.fromarray(array)

        j = 1
        extracted_face = []
        for face in faces:     
            bbox = face['bounding_box']
            pivotX, pivotY = face['pivot']
            
            if bbox['width'] < 10 or bbox['height'] < 10:
                continue
            
            left = pivotX - bbox['width'] / 1.0 * padding
            top = pivotY - bbox['height'] / 1.0 * padding
            right = pivotX + bbox['width'] / 1.0 * padding
            bottom = pivotY + bbox['height'] / 1.0 * padding
            cropped = img.crop((left, top, right, bottom))
            extracted_face.append(cropped)
        return extracted_face
    pass

def rev_search(img_url):
    url = "https://reverse-image-search-by-copyseeker.p.rapidapi.com/"

    querystring = {"imageUrl":f"{img_url}"}

    headers = {
        "X-RapidAPI-Key": "19f351f953mshe8f8e82d609061bp1ff0a1jsn8affa0ab462b",
        "X-RapidAPI-Host": "reverse-image-search-by-copyseeker.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    print(response.json())

#---------------------------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("SPY KIDS GADGET")
st.write("PERSIST venture Assignment")

face_extract = st.form(key = 'extract faces')
face_extract.title("extract faces from video")
face_extract.text("upload video")
uploaded_video = face_extract.file_uploader("Upload Mp4 file", type=["mp4"])
extract_face_btn = face_extract.form_submit_button("extract_faces")
if extract_face_btn:
    if not uploaded_video:
        face_extract.write("cant recognize")
    else:
        vid = uploaded_video.name
        img = face_extraction(vid)
        dlinks =[]
        for idx, i in enumerate(img):
            img_array = Image.fromarray(np.array(i))
            buffered = BytesIO()
            img_array.save(buffered,format='JPEG')
            img_bytes = buffered.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode()
            href = f'<a href="data:file/jpg;base64,{img_b64}" download="image{idx + 1}.jpg">Download Image {idx + 1}</a>'
            dlink = st.markdown(href, unsafe_allow_html=True)
            dlinks.append(f"data:file/jpg;base64,{img_b64}")
        res = rev_search(img[0])
        st.write(res)

