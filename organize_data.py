import glob
import os
import random
import shutil
import time
from timeit import timeit
from deepface import DeepFace
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import ast
from scipy.spatial.distance import cosine
import cv2
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from PIL import Image
from collections import Counter
import utils

import face_recognition

FACE_DETECTOR = 'skip'
MODEL = "Facenet"  # VGG-Face or Facenet
dir_name = "Dataset_"



def extract_face_from_image(image_path, required_size=(224, 224)):
    # load image and detect faces
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    face_images = []

    if len(faces) == 0:
        return face_images

    for i, face in enumerate(faces):
        # extract the bounding box from the requested face
        if face['confidence'] < 0.98:
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
        #face_image.save(f"face-cropped{i}.jpg")

    return face_images





def rec_dlib(image, face_box):
    main_cand_list = list()
    candidates = list()
    start_time = time.time()

    embeddings = face_recognition.face_encodings(image, [face_box])[0]

    df = pkl_dfs[0]

    for i, player_info in enumerate(list(df.itertuples(index=False, name=None))):
        sim_score = cosine(embeddings, player_info[1])
        candidates.append((player_info[0], sim_score))

    top_3_candidates = sorted(candidates, key=lambda x: x[1])[:3]  # get first 3


    if top_3_candidates[0][1] < 0.07:
        main_cand_list.extend([x[0] for x in top_3_candidates])

    # 2nd Round
    candidates.clear()

    df = pkl_dfs[1]

    for i, player_info in enumerate(list(df.itertuples(index=False, name=None))):
        sim_score = cosine(embeddings, player_info[1])
        candidates.append((player_info[0], sim_score))

    top_3_candidates = sorted(candidates, key=lambda x: x[1])[:3]  # get first 5

    if top_3_candidates[0][1] < 0.07:
        main_cand_list.extend([x[0] for x in top_3_candidates])

    if len(main_cand_list) > 0:
        main_cand_list = [x.replace('-', ' ').lower() for x in main_cand_list]
        counts = Counter(main_cand_list)
        answer = max(counts, key=counts.get)

        if counts.get(answer) == 1:
            answer = "unrecognized"
            print("NOT ABLE TO RECOGNIZE! ")
            embeddings = None
        print(f"Aaaaand the player is {answer}")

    else:
        answer = "unrecognized"
        print("NOT ABLE TO RECOGNIZE! ")
        embeddings = None

    print(f"Execution Time: {time.time() - start_time}")

    return answer, embeddings



def recognize_player_hyrbid(image_path, vgg_face_model, facenet_model):

    main_cand_list = list()
    candidates = list()
    start_time = time.time()

    embeddings = DeepFace.represent(img_path=image_path, model=vgg_face_model, detector_backend=FACE_DETECTOR, model_name='VGG-Face')

    df = pkl_dfs[0]  # TODO: Re-think this approach!

    for i, player_info in enumerate(list(df.itertuples(index=False, name=None))):
        sim_score = cosine(embeddings, player_info[1])
        candidates.append((player_info[0], sim_score))

    top_3_candidates = sorted(candidates, key=lambda x: x[1])[:3]  # get first 3
    if top_3_candidates[0][1] < 0.25:
        main_cand_list.extend([x[0] for x in top_3_candidates])

    # 2nd Round
    candidates.clear()
    embeddings = DeepFace.represent(img_path=image_path, model= facenet_model, detector_backend=FACE_DETECTOR, model_name='Facenet')

    df = pkl_dfs[1]

    for i, player_info in enumerate(list(df.itertuples(index=False, name=None))):
        sim_score = cosine(embeddings, player_info[1])
        candidates.append((player_info[0], sim_score))

    top_3_candidates = sorted(candidates, key=lambda x: x[1])[:3]  # get first 5

    if top_3_candidates[0][1] < 0.5:
        main_cand_list.extend([x[0] for x in top_3_candidates])

    # 3rd Round

    candidates.clear()

    # No need to infer embeddings again as we are reusing facenet embeddings.

    df = pkl_dfs[2]

    for i, player_info in enumerate(list(df.itertuples(index=False, name=None))):
        sim_score = cosine(embeddings, player_info[1])
        candidates.append((player_info[0], sim_score))

    top_3_candidates = sorted(candidates, key=lambda x: x[1])[:3]  # get first 5

    if top_3_candidates[0][1] < 0.5:
        main_cand_list.extend([x[0] for x in top_3_candidates])

    if len(main_cand_list) > 0:
        main_cand_list = [x.replace('-', ' ').lower() for x in main_cand_list]
        counts = Counter(main_cand_list)
        answer = max(counts, key=counts.get)

        if counts.get(answer) == 1:
            answer = "unrecognized"
            print("NOT ABLE TO RECOGNIZE! ")
            return answer

        print(f"Aaaaand the player is {answer}")

    else:
        answer = "unrecognized"
        print("NOT ABLE TO RECOGNIZE! ")

    print(f"Execution Time: {time.time() - start_time}")

    return answer


pkl_dfs = list()


def main():
    image_paths_list = os.listdir(dir_name)
    print(f"Number of Images in the directory: {len(image_paths_list)}")

    if len(image_paths_list) == 0:
        print("No Images found in the directory")
        return

    # Load the models first
    vgg_face_model = DeepFace.build_model("VGG-Face")
    facenet_model = DeepFace.build_model("Facenet")

    # Read all the pickle files first.
    df_1 = pd.read_pickle("data/player_facial_embeddings_vggface_tm.pkl")
    pkl_dfs.append(df_1)

    df_2 = pd.read_pickle("data/TM/player_facial_embeddings_facenet_TM.pkl")
    pkl_dfs.append(df_2)

    df_3 = pd.read_pickle("data/FBREF/player_facial_embeddings_facenet_FBREF.pkl")
    pkl_dfs.append(df_3)


    for i, image_path in enumerate(image_paths_list):
        image_path = dir_name + '/' + image_path
        if not cv2.haveImageReader(image_path):
            continue
        try:
            face_images = extract_face_from_image(image_path)
            if len(face_images) == 0:
                continue
            for face in face_images:
                res = recognize_player_hyrbid(face, vgg_face_model, facenet_model)
                if not os.path.isdir('organized/' + res):
                    os.mkdir('organize/' + res)
                    # shutil.copy(image_path, 'organized/' + res)

                save_path = os.path.join("organized", res, f"{i}--{random.randint(0, 10)}--{image_path.split('/')[-1]}")
                Image.fromarray(face).save(save_path)

        except:
            continue


troublesome_images = []


def main_dlib():
    image_paths_list = os.listdir(dir_name)
    print(f"Number of Images in the directory: {len(image_paths_list)}")

    if len(image_paths_list) == 0:
        print("No Images found in the directory")
        return

    embeddings_list = list()

    # Read all the pickle files first.
    df_1 = pd.read_pickle("data/TM/player_facial_embeddings_dlib_TM.pkl")
    pkl_dfs.append(df_1)

    df_2 = pd.read_pickle("data/FBREF/player_facial_embeddings_dlib_FBREF.pkl")
    pkl_dfs.append(df_2)

    for i, image_path in enumerate(image_paths_list):

        print(f"Done with {i}/{len(image_paths_list)}")

        image_path = dir_name + '/' + image_path
        if not cv2.haveImageReader(image_path):
            continue
        try:
            image, face_boxes = utils.get_face_box(image_path)
            if len(face_boxes) == 0:
                continue
            for face_box in face_boxes:
                res, embeddings = rec_dlib(image, face_box)
                if not os.path.isdir('organized/' + res):
                    os.mkdir('organized/' + res)
                    # shutil.copy(image_path, 'organized/' + res)

                save_path = os.path.join("organized", res,
                                         f"{i}--{random.randint(0, 10)}--{image_path.split('/')[-1]}")
                if not res.__eq__("unrecognized"):
                    embeddings_list.append((res, embeddings, save_path))
                y1, x1, y2, x2 = face_box
                cv2.imwrite(save_path, image[y1:y2, x2:x1])
                #Image.fromarray(image[y1:y2, x2:x1]).save(save_path)

        except:
            troublesome_images.append(image_path)
            continue

    df = pd.DataFrame(embeddings_list, columns=['PlayerName', 'Embeddings', "ImagePath"])
    # Change name of the file when needed!
    df.to_csv(f"dlib_data.csv")
    print()
    print(f"Toubllleeee - {troublesome_images}")


def rename():
    image_paths_list = os.listdir('headshots_tm')
    for i, image_path in enumerate(image_paths_list):
        player_name = image_path.split('_')[0]
        modified_plyr_nm = " ".join(w.title() for w in player_name.split('-'))
        os.rename('headshots_tm/'+image_path, 'headshots_tm/'+image_path.replace(player_name, modified_plyr_nm))


if __name__ == '__main__':
    main_dlib()
