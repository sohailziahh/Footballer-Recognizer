import streamlit as st
from PIL import Image
import pickle
# import pandas as pd
from numpy import asarray,array
import numpy as np
from deepface import DeepFace
from deepface.detectors import FaceDetector


@st.cache  #
def loading_encoded_headshots():
    headshots = pickle.loads(open('haedshots_deepface_dlib_encodings_bin.pickle', "rb").read())
    # headshots["no"] = headshots["no"].fillna(0).astype(int)
    # headshots = headshots.dropna(subset=['encoding_tm']).reset_index(drop=True) 
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
    image = np.array(image)
    detector = FaceDetector.build_model('dlib')
    faces = FaceDetector.detect_faces(detector, 'dlib', image)
    predictions = []
    sucess = False
    if len(faces)>=1:
        for face in faces:
            face_array = asarray(face[0])
            encoding = (DeepFace.represent(img_path=face_array, detector_backend='skip',
                            enforce_detection=False, model_name='Dlib'))
            analysis = DeepFace.analyze(img_path = face_array, actions = ["emotion",],enforce_detection=False,detector_backend='dlib')
            print(analysis)
            predict = {'name':'unrecognized','sure':1,'encoding': encoding,'emotion':analysis['dominant_emotion']}
            predictions.append(predict)
            sucess = True
    return predictions,sucess
# def age_prediction(image):
#     image = np.array(image)
#     analysis = DeepFace.analyze(img_path = image, actions = ["age", "gender", "emotion", "race"])
#     print(analysis)
def find_closest_face(unknown_face, faces):
    distances = []   
    for _,face in faces.iterrows():
        distance_image= []
        # for i in range(face["no"]):
        #     distance_image.append(face_distance(unknown_face,asarray(face[f"encoding_ggl_{i+1}"])))
        # for i in range(face["encoding_tm_g_no"]):
        #     distance_image.append(face_distance((unknown_face),asarray(face[f"encoding_tm_g_{i}"])))
        distance_image.append(face_distance((unknown_face['encoding']),asarray(face["encoding_tm_dlib"])))
        if type(face["encoding_fbref_dlib"])== list:
            distance_image.append(face_distance(unknown_face['encoding'],asarray(face["encoding_fbref_dlib"])))
        # else:
        #     distance_image.append(1)

        distances.append(np.median(distance_image))
        
    
    distances_sort = distances.copy()
    distances_sort.sort()
    # name = {'name':'unrecognized','sure':1}
    if len([i for i in distances if i <= 0.51])==1:
    # if min(distances)<0.51:  
        unknown_face['name'] = faces["name"][distances.index(distances_sort[0])]
        unknown_face['confidence'] = 1
    elif len([i for i in distances if i <= 0.51])==0: 
        if  (distances_sort[1]- distances_sort[0])/ min(distances) >0.21:
          unknown_face['name'] = faces["name"][distances.index(distances_sort[0])] 
          unknown_face['confidence'] = 1 
    elif min(distances)<0.57:
        if  (distances_sort[1]- distances_sort[0])/ min(distances) >0.01:
            unknown_face['name'] = faces["name"][distances.index(distances_sort[0])] 
            unknown_face['confidence'] = 0

    return unknown_face

st.set_page_config(layout="wide")
uploaded_file = st.file_uploader("load an image")
col1, col2 = st.columns(2)
if uploaded_file is not None:
    headshots= loading_encoded_headshots()
    image = Image.open(uploaded_file)
    col1.image(image, caption=image.filename)
    encodings_predictions, success_encoded= encode_deefpface(image)
    if success_encoded:
        for encond in encodings_predictions:
            encond = (find_closest_face(encond,headshots))
        predicted_names = [p for p in encodings_predictions if p['name']!= "unrecognized" ]
        if len(predicted_names)!=0 :
            title1 = f'<p style="font-family:sans-serif; font-size: 25px;">{len(predicted_names)} face(s) detected :</p>'
            # title2 = '<p style="font-family:sans-serif; color:White; font-size: 14px;">(For quite confidently predicted faces, their names indicated in green) <p style="font-family:sans-serif; color:White; font-size: 14px;"> Otherwise they are coloured in yellow)</p>'
            title3 = '(High-Confidence predictions marked in <span style="color:green"> green </span> and Low-Confidence ones in <span style="color:darkorange"> orange )'
            col2.markdown(title1,unsafe_allow_html=True)
            st.text(" \n")
            col2.markdown(title3,unsafe_allow_html=True)
            for name in predicted_names:
                st.text(" \n")
                name_shown = str(name['name']).replace('-',' ').upper()
                if name['confidence'] == 1:
                    col2.markdown(f'<span style="font-family:sans-serif; color:Green; font-size: 22px;"> {name_shown}</span>  (Dominant emotion : {name["emotion"]})',unsafe_allow_html=True)
                    # col2.markdown(f'<p style="font-family:sans-serif; color:white; font-size: 18px;"> Age : {name["age"]}</p>',unsafe_allow_html=True)
                else:
                    col2.markdown(f'<span style="font-family:sans-serif; color:darkorange; font-size: 22px;"> {name_shown}</span>  (Dominant emotion : {name["emotion"]})',unsafe_allow_html=True)
                    # col2.markdown(f'<p style="font-family:sans-serif; color:white; font-size: 12px;"> Dominant emotion : {name["emotion"]}</p>',unsafe_allow_html=True)
                    # col2.write(f'Dominant emotion : {name["emotion"]}')

        else:
            title = f'<p style="font-family:sans-serif; color:Red; font-size: 20px;"> We couldnt detect any player</p>'
            col2.markdown(title,unsafe_allow_html=True)
    else:
        col2.write('No face was detected in the image')
    
   