import streamlit as st
from PIL import Image
import pickle
# import pandas as pd
from numpy import asarray,array
import numpy as np
from deepface import DeepFace
from deepface.detectors import FaceDetector

from slack_sdk import WebClient
# from pathlib import Path
# import os
from datetime import datetime
# from dotenv import load_dotenv
import cv2
import tempfile
import boto3
from ast import literal_eval

SLACK_CHANNEL_ID = "C04EYUXK4UF"
BNAME = 'ds-image-to-quiz'
def get_slack_client():
    # env_path = Path('.') / '.env'
    # load_dotenv(dotenv_path=env_path)
    # return WebClient(token=os.environ['SLACK_TOKEN'])
    return WebClient(token=st.secrets["var1"])

def get_aws_client():
    s3 = boto3.client('s3',aws_access_key_id=st.secrets['var2'],
                    aws_secret_access_key=st.secrets['var3'])
    return s3
    

def log_on_slack(s_client,a_client, data):
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    response = s_client.chat_postMessage(channel=SLACK_CHANNEL_ID, text=f"Dated: {now}")
    msg_time_stamp = response.data['ts']  # Timestamp to the message is needed so that we can reply further messages
    # into that message as a thread.

    # Data contains tuples of <Predicted Player Names, Cropped Faces>
    for dat in data:
        cv2.imwrite("image.jpg", dat[1])  #  todo do we need to write/save the image first? not sure. will have to look into this later.
        s_client.files_upload(channels=SLACK_CHANNEL_ID, thread_ts=msg_time_stamp, file="image.jpg",
                            initial_comment=dat[0])
        names = str('_'.join([i['name'] for i in  literal_eval(dat[0])]))
        s3_name = f"wrong_face_predictions/{names}_{now}.jpg"
        a_client.upload_file("image.jpg",BNAME,s3_name)
@st.cache  #
def loading_encoded_headshots():
    headshots = pickle.loads(open('haedshots_deepface_dlib_encodings_bin.pickle', "rb").read())
    return headshots
    
def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    # import numpy as np
    # if len(face_encodings) == 0:
    #     return np.empty((0))
    
    return np.linalg.norm(face_encodings - face_to_compare, axis=0)

def encode_deefpface(image):
    try:
        image = np.array(image)
        detector = FaceDetector.build_model('dlib')
        faces = FaceDetector.detect_faces(detector, 'dlib', image)
    except:
        return [],False
    encodings = []
    sucess = False
    if len(faces)>=1:
        for face in faces:
            face_array = asarray(face[0])
            encodings.append(DeepFace.represent(img_path=face_array, detector_backend='dlib',
                            enforce_detection=False, model_name='Dlib'))
            sucess = True
    return encodings,sucess

def find_closest_face(unknown_face, faces):
    distances = []   
    for _,face in faces.iterrows():
        distance_image= []
        distance_image.append(face_distance((unknown_face),asarray(face["encoding_tm_dlib"])))
        if type(face["encoding_fbref_dlib"])== list:
            distance_image.append(face_distance(unknown_face,asarray(face["encoding_fbref_dlib"])))
        # else:
        #     distance_image.append(1)

        distances.append(np.median(distance_image))
        
    
    distances_sort = distances.copy()
    distances_sort.sort()
    name = {'name':'unrecognized','confidence':1}
    if len([i for i in distances if i <= 0.47])==1:
        name['name'] = faces["name"][distances.index(distances_sort[0])]
        name['confidence'] = 1
    elif len([i for i in distances if i <= 0.52])==0: 
        if  (distances_sort[1]- distances_sort[0])/ min(distances) >0.07:
          name['name'] = faces["name"][distances.index(distances_sort[0])] 
          name['confidence'] = 1 
    elif min(distances)<0.58:
        pot_names = ""
        for i in range(10):
            pot_names += faces["name"][distances.index(distances_sort[i])] + ' ' + '('+ str(round(distances_sort[i],2)) +') '
        print(pot_names)
        name['name'] =  pot_names
        name['confidence'] = 0
    
    return name

st.set_page_config(layout="wide")
uploaded_file = st.file_uploader("load an image")

col1, col2 = st.columns(2)
if uploaded_file is not None:
    headshots= loading_encoded_headshots()
    image = Image.open(uploaded_file)
    col1.image(image, caption=image.filename)
    encodings_image, success_encoded= encode_deefpface(image)
    if success_encoded:
        predicted_names = []
        for encond in encodings_image:
            predicted_names.append(find_closest_face(encond,headshots))
        predicted_names = [p for p in predicted_names if p['name']!= "unrecognized" ]
        if len(predicted_names)!=0 :
            title1 = f'<p style="font-family:sans-serif; color:White; font-size: 25px;">{len(predicted_names)} face(s) detected :</p>'
            title3 = '(High-Confidence predictions marked in <span style="color:green"> green </span> and Low-Confidence ones in <span style="color:Orange"> orange )'
            col2.markdown(title1,unsafe_allow_html=True)
            st.text(" \n")
            col2.markdown(title3,unsafe_allow_html=True)
            for name in predicted_names:
                st.text(" \n")
                name_shown = str(name['name']).replace('-',' ').upper()
                if name['confidence'] == 1:
                    col2.markdown(f'<p style="font-family:sans-serif; color:Green; font-size: 22px;"> {name_shown}</p>',unsafe_allow_html=True)
                else:
                    col2.markdown(f'<p style="font-family:sans-serif; color:Orange; font-size: 22px;"> {name_shown}</p>',unsafe_allow_html=True)
           
            if col2.button('CLICK ME IF PREDICTION IS WRONG'):
                slack_client = get_slack_client()
                aws_client = get_aws_client()
                wrong_data = (f"{predicted_names}", cv2.cvtColor(np.array(image),cv2.COLOR_BGR2RGB ))
                log_on_slack(slack_client,aws_client, [wrong_data])
                col2.write("Thanks. Your feedback is received")
        else:
            title = f'<p style="font-family:sans-serif; color:Red; font-size: 20px;"> We couldnt detect any player</p>'
            col2.markdown(title,unsafe_allow_html=True)
       
    else:
        col2.write('No face was detected in the image')
    
   