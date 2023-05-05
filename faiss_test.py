import faiss
# make faiss available
import time
from timeit import timeit
from deepface import DeepFace
import pandas as pd
import ast
from scipy.spatial.distance import cosine
import cv2
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from PIL import Image
from collections import Counter
import dask.dataframe
import numpy as np

d = 128


def rec(image_path):
    try:
        embeddings = DeepFace.represent(img_path=image_path, detector_backend='mtcnn', model_name='Facenet')
        return embeddings
    except:
        return None


def func(index, df):
    test_image_path = "to_test/000_32FR8LC.jpg"
    # generate_face_embeddings()  # Run this if you don't have the csv!
    # recognize_player(test_image_path)

    start_time = time.time()
    face_images = extract_face_from_image(test_image_path)
    end_time = time.time()
    print("Face extraction time taken ", end_time - start_time)

    for face in face_images:
        infer_time = time.time()
        embeddings = rec(face)
        if (embeddings == None):
            continue
        print(f"Inference took {time.time() - infer_time}")
        embeddings = np.array(embeddings).astype("float32").reshape(1, -1)
        faiss_time = time.time()
        res = index.search(embeddings, 2)
        print(res)
        print(f"Answer is {df['PlayerName'].iloc[res[1][0][0]]}")

        print(f"Faiss took {time.time() - faiss_time}")
    end_time = time.time()
    print("Total time taken ", end_time - start_time)


def extract_face_from_image(image_path, required_size=(224, 224)):
    # load image and detect faces
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    face_images = []

    for i, face in enumerate(faces):
        # extract the bounding box from the requested face
        if face['confidence'] < 0.95:
            continue
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face_boundary = image[y1:y2, x1:x2]
        # resize pixels to the model size
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)
        face_image.save(f"face-cropped{i}.jpg")

    return face_images


def main():
    index = faiss.IndexFlatL2(d)  # build the index
    df = pd.read_pickle("data/embeddings_tm+fr+data_2.pkl")
    index.add(pd.Series(df['Embeddings']).explode().values.reshape((df.shape[0], -1)).astype('float32'))

    func(index, df)


if __name__ == '__main__':
    main()
