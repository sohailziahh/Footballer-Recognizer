from deepface import DeepFace
from deepface.detectors import FaceDetector
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
import pandas as pd
import pickle
import cv2
import face_recognition

def encode_deefpface(imagePath):
    if imagePath!='':
        image = cv2.imread(imagePath)
        detector = FaceDetector.build_model('dlib')
        # detector = DlibWrapper.build_model()
        faces = FaceDetector.detect_faces(detector, 'dlib', image)
        encodings = []
        for face in faces:
            face_array = asarray(face[0])
            encodings.append(DeepFace.represent(img_path=face_array, detector_backend='dlib',
                            enforce_detection=False, model_name='Dlib'))
    return encodings

def encode_facerecognition(imagePath):
    if imagePath!='':
        image = cv2.imread(imagePath)
        # ocnverting image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(image,model='cnn')
        encodings = face_recognition.face_encodings(image, boxes)
        return encodings

def check_for_headhsot(encods_to_check):
    if len(encods_to_check) == 1:
        encoed = encods_to_check[0]
        return encoed
    else:
        return ''
    