import os
import time
from deepface import DeepFace
import pandas as pd
from scipy.spatial.distance import cosine
import cv2
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from PIL import Image
from collections import Counter
import face_recognition
import numpy as np

import utils

FACE_DETECTOR = 'skip'
# MODEL = "VGG-Face"  # VGG-Face or Facenet
MODEL = "Facenet"


def generate_face_embeddings(model):
    for data in ["Dataset_fbref/headshots", "headshots_tm"]:

        prefix = "TM_" if data.__eq__("headshots_tm") else "FBREF_"

        image_paths_list = os.listdir(data)
        print(f"Number of Images in the directory: {len(image_paths_list)}")

        if len(image_paths_list) == 0:
            print("No Images found in the directory")
            return

        embeddings_list = list()

        for i, image_path in enumerate(image_paths_list):
            if not cv2.haveImageReader(os.path.join(data, image_path)):
                continue
            try:
                print(os.path.join(data, image_path))

                # faces = utils.extract_face_from_image(os.path.join(data, image_path))
                # embeddings = DeepFace.represent(img_path=faces[0], model=model, detector_backend=FACE_DETECTOR,
                #                                 model_name=MODEL)

                image, face_box = utils.get_face_box(os.path.join(data, image_path))
                embeddings = face_recognition.face_encodings(image, [face_box[0]])[0]
            except:
                print(f"Error for {image_path}")
                continue

            player_name = image_path.split("/")[-1].split("_")[0]
            embeddings_list.append((player_name, embeddings, prefix + image_path))
            print(f"{i} / {len(image_paths_list)}")

        df = pd.DataFrame(embeddings_list, columns=['PlayerName', 'Embeddings', "ImagePath"])
        # Change name of the file when needed!
        df.to_pickle(f"player_facial_embeddings_dlib_{prefix}.pkl")
        print(f"{prefix} : Embeddings generated for all of the images. ")
        print(len(embeddings_list))


def recognize_player(image_path):
    candidates = list()

   # embeddings = DeepFace.represent(img_path=image_path, detector_backend=FACE_DETECTOR, model_name=MODEL)

    image, face_box = utils.get_face_box(image_path)
    embeddings = face_recognition.face_encodings(image, [face_box[0]])[0]

    pkl_file = "data/player_facial_embeddings_facenet_TM_.pkl" if MODEL.__eq__(
        "Facenet") else "player_facial_embeddings_vggface_tm.pkl"  # either VGGFace or Facenet

    pkl_file = "data/embeddings_tm+fr+data_2.pkl"

    pkl_file = "data/TM/player_facial_embeddings_dlib_TM.pkl"

    pkl_file = "data/combined_thanos_embeddings.pkl"

    pkl_file = "../data/thanos+popular_embeddings.pkl"

    df = pd.read_pickle(pkl_file)

    start_time = time.time()
    for i, player_info in enumerate(list(df.itertuples(index=False, name=None))):
        sim_score = cosine(embeddings, player_info[1])
        candidates.append((player_info[0], sim_score))

    top_5_candidates = sorted(candidates, key=lambda x: x[1])[:5]  # get first 5
    [print(candidate) for candidate in top_5_candidates]
    print(f"Execution Time: {time.time() - start_time}")
    if top_5_candidates[0][1] >= 0.047:
        print("We do not know who this player is!")
        return "unrecognized"
    else:
        print(f"Aaaaand the player is {top_5_candidates[0][0]}")
        return top_5_candidates[0][0]


def recognize_player_hyrbid(image_path, vgg_face_model, facenet_model):
    main_cand_list = list()
    candidates = list()
    start_time = time.time()

    # 1st Round
    embeddings = DeepFace.represent(img_path=image_path, model=vgg_face_model, detector_backend=FACE_DETECTOR,
                                    model_name='VGG-Face')
    print(f"first inference took {time.time() - start_time}")

    df = pkl_dfs[0]  # TODO: Re-think this approach!

    stt = time.time()
    for i, row in df[['PlayerName', 'Embeddings']].iterrows():
        sim_score = cosine(embeddings, row['Embeddings'])
        candidates.append((row['PlayerName'], sim_score))
    print(f"first round took {time.time() - stt}")

    top_3_candidates = sorted(candidates, key=lambda x: x[1])[:3]  # get first 5

    print(top_3_candidates)

    main_cand_list.extend([x[0] for x in top_3_candidates])

    # 2nd Round
    candidates.clear()

    stt = time.time()
    embeddings = DeepFace.represent(img_path=image_path, model=facenet_model, detector_backend=FACE_DETECTOR,
                                    model_name='Facenet')

    print(f"second inference took {time.time() - stt}")

    df = pkl_dfs[1]

    stt = time.time()
    for i, row in df[['PlayerName', 'Embeddings']].iterrows():
        sim_score = cosine(embeddings, row['Embeddings'])
        candidates.append((row['PlayerName'], sim_score))
    print(f"second round took {time.time() - stt}")

    top_3_candidates = sorted(candidates, key=lambda x: x[1])[:3]  # get first 5
    print(top_3_candidates)

    main_cand_list.extend([x[0] for x in top_3_candidates])

    # 3rd Round

    candidates.clear()

    # No need to infer embeddings again as we are reusing facenet embeddings.

    df = pkl_dfs[2]

    stt = time.time()
    for i, row in df[['PlayerName', 'Embeddings']].iterrows():
        sim_score = cosine(embeddings, row['Embeddings'])
        candidates.append((row['PlayerName'], sim_score))
    print(f"third round took {time.time() - stt}")

    top_3_candidates = sorted(candidates, key=lambda x: x[1])[:3]  # get first 5
    print(top_3_candidates)

    main_cand_list.extend([x[0] for x in top_3_candidates])

    main_cand_list = [x.replace('-', ' ').lower() for x in main_cand_list]
    counts = Counter(main_cand_list)
    print(counts)
    answer = max(counts, key=counts.get)

    print(f"Aaaaand the player is {answer}")
    print(f"Execution Time: {time.time() - start_time}")


def extract_face_from_image(image_path, required_size=(224, 224)):
    # load image and detect faces
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    face_images = []

    for i, face in enumerate(faces):
        # extract the bounding box from the requested face
        if face['confidence'] < 0.97:
            continue
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face_boundary = image[y1:y2, x1:x2]
        # resize pixels to the model size
        face_image = cv2.resize(face_boundary, required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)

        # Image.fromarray(face_array).save(f"face-cropped{i}.jpg")

    return face_images


pkl_dfs = list()


def main():
    # No need to preprocess image (face detection, cropping, resizing, etc.) as this is done behind the scenes in
    # DeepFace. However, if you still want to do it, then you can pass detector_backend = 'skip' to bypass Deepface's
    # preprocessing step.

    # Load the models first
    vgg_face_model = DeepFace.build_model("VGG-Face")
    facenet_model = DeepFace.build_model("Facenet")

    # Read all the pickle files first.
    pkl_read_time = time.time()

    df_1 = pd.read_pickle("data/TM/player_facial_embeddings_vggface_TM.pkl")
    pkl_dfs.append(df_1)
    print(f"First Pickle read took {time.time() - pkl_read_time}")

    df_2 = pd.read_pickle("data/TM/player_facial_embeddings_facenet_TM.pkl")
    pkl_dfs.append(df_2)

    df_3 = pd.read_pickle("data/FBREF/player_facial_embeddings_facenet_FBREF.pkl")
    pkl_dfs.append(df_3)

    print(f"Pickle read took {time.time() - pkl_read_time}")

    test_image_path = "to_test/rashford.jpg"

    recognize_player(test_image_path)

    start_time = time.time()
    face_images = extract_face_from_image(test_image_path)
    end_time = time.time()
    print("Face extraction time taken ", end_time - start_time)

    start_time = time.time()
    for face in face_images:
        stt = time.time()
        recognize_player_hyrbid(face, vgg_face_model, facenet_model)
        print(f"Recognizer took {time.time() - stt}")
    end_time = time.time()

    print("Total time taken ", end_time - start_time)


if __name__ == '__main__':
    # Load the models first
    # vgg_face_model = DeepFace.build_model("VGG-Face")
    # facenet_model = DeepFace.build_model("Facenet")
    # generate_face_embeddings(facenet_model)  # Run this if you don't have the pickle file!
    main()
