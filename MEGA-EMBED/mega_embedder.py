import glob
import math
import os
import random
import time
from deepface import DeepFace
import pandas as pd
import ast
from scipy.spatial.distance import cosine
import cv2
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from collections import Counter
from PIL import Image
import shutil

FACE_DETECTOR = 'skip'
MODEL = "Facenet"  # VGG-Face or Facenet

dir_name = "/cropped_images"
pkl_dfs = list()
crop_images_path = os.path.join(os.getcwd(), "MEGA-EMBED/crops")

tm_path = "/headshots_tm"
fr_path = "/Dataset_fbref/headshots"
crop_path = "/MEGA-EMBED/crops"

init_num = 2061  # TODO magic number!
counter = 0


def extract_face_from_image(image_path, required_size=(224, 224)):
    # load image and detect faces
    global counter
    image = cv2.imread(image_path)
    detector = MTCNN()
    stt = time.time()
    faces = detector.detect_faces(image)
    print(f"Time taken {time.time() - stt}")
    face_images = []

    if len(faces) == 0:
        return face_images

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
        # face_image.save(f"face-cropped{i}.jpg")
    counter = counter + 1
    print(counter)
    return face_images


def recognize_player_hyrbid(image_path):
    main_cand_list = list()
    candidates = list()
    start_time = time.time()

    embeddings = DeepFace.represent(img_path=image_path, detector_backend=FACE_DETECTOR, model_name='VGG-Face')

    df = pkl_dfs[0]  # TODO: Re-think this approach!

    for i, player_info in enumerate(list(df.itertuples(index=False, name=None))):
        sim_score = cosine(embeddings, player_info[1])
        candidates.append((player_info[0], sim_score))

    top_3_candidates = sorted(candidates, key=lambda x: x[1])[:3]  # get first 5
    if top_3_candidates[0][1] < 0.38:
        main_cand_list.extend([x[0] for x in top_3_candidates])

    # 2nd Round
    candidates.clear()
    embeddings = DeepFace.represent(img_path=image_path, detector_backend=FACE_DETECTOR, model_name='Facenet')

    df = pkl_dfs[1]

    for i, player_info in enumerate(list(df.itertuples(index=False, name=None))):
        sim_score = cosine(embeddings, player_info[1])
        candidates.append((player_info[0], sim_score))

    top_3_candidates = sorted(candidates, key=lambda x: x[1])[:3]  # get first 5

    if top_3_candidates[0][1] < 0.38:
        main_cand_list.extend([x[0] for x in top_3_candidates])

    # 3rd Round

    candidates.clear()
    embeddings = DeepFace.represent(img_path=image_path, detector_backend=FACE_DETECTOR, model_name='Facenet')

    df = pkl_dfs[2]

    for i, player_info in enumerate(list(df.itertuples(index=False, name=None))):
        sim_score = cosine(embeddings, player_info[1])
        candidates.append((player_info[0], sim_score))

    top_3_candidates = sorted(candidates, key=lambda x: x[1])[:3]  # get first 5

    if top_3_candidates[0][1] < 0.38:
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


def generate_face_embeddings():
    model = DeepFace.build_model("Facenet")
    embeddings_list = list()

    all_dirs = next(os.walk(dir_name))[1]
    for i, dir in enumerate(all_dirs):

        print(dir)
        if dir.__contains__('unrecognized'):
            continue
        image_path_list = os.listdir(os.path.join(dir_name, dir))

        if len(image_path_list) == 0:
            continue

        for idx, image_path in enumerate(image_path_list):
            print(dir, image_path)
            image_path = os.path.join(dir_name, dir, image_path)
            if not cv2.haveImageReader(image_path):
                continue

            #faces = extract_face_from_image(image_path)
            #for face in faces:
            try:
                    # result = recognize_player_hyrbid(face)
                    # if result.__eq__("unrecognized"):
                    #     continue
                embeddings = DeepFace.represent(img_path=image_path, detector_backend=FACE_DETECTOR,
                                                    model_name=MODEL, model=model)
            except:
                print(f"Error for {image_path}")
                continue

            player_name = dir
            embeddings_list.append((player_name, embeddings, os.path.join(dir_name, dir, image_path)))

            print(f"{idx} / {len(image_path_list)}")

        print(f"{i} / {len(all_dirs)}")

    df = pd.DataFrame(embeddings_list, columns=['PlayerName', 'Embeddings', "ImagePath"])
    df.to_pickle('embeddings_dataset.pkl')

    print("Embeddings generated for all of the images. ")

    print(len(embeddings_list))


def main():
    image_paths_list = os.listdir(dir_name)
    print(f"Number of Images in the directory: {len(image_paths_list)}")

    if len(image_paths_list) == 0:
        print("No Images found in the directory")
        return

import unicodedata


def strip_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')


def combiner():
    global init_num
    image_paths_list_tm = os.listdir(tm_path)
    trimmed_image_paths_list_tm = [data.split('_')[0].lower() for data in image_paths_list_tm]
    image_paths_list_fr = os.listdir(fr_path)
    trimmed_image_paths_list_fr = [data.split('_')[0].lower() for data in image_paths_list_fr]

    df_tm = pd.read_pickle("../data/TM/player_facial_embeddings_dlib_TM.pkl")
    df_fr = pd.read_pickle("../data/FBREF/player_facial_embeddings_dlib_FBREF.pkl")
    df_mega = pd.read_pickle("../data/dlib_data.pkl")

    # TM
    for row in df_tm.PlayerName.str.replace("-", " "):
        index = trimmed_image_paths_list_tm.index(row.lower())
        src_path = os.path.join(tm_path, image_paths_list_tm[index])
        filename = image_paths_list_tm[index].split('_')[0].replace(" ", "-") + f"-{init_num}.jpg"
        init_num = init_num + 1
        dst_path = os.path.join(crop_path, filename)
        print()
        #shutil.copy(src_path, dst_path)


    df_tm.PlayerName = df_tm.PlayerName.str.lower().apply(strip_accents)

    here = list(df_mega.itertuples(index=False, name=None))
    there = list(df_tm.itertuples(index=False, name=None))
    where = here + there

    # FBREF
    for row in df_fr.PlayerName:
        index = trimmed_image_paths_list_fr.index(row.lower())
        src_path = os.path.join(fr_path, image_paths_list_fr[index])
        filename = image_paths_list_fr[index].split('_')[0].replace(" ", "-").lower() + f"-{init_num}.jpg"
        init_num = init_num + 1
        dst_path = os.path.join(crop_path, filename)
        #shutil.copy(src_path, dst_path)

    df_fr.PlayerName = df_fr.PlayerName.str.lower().apply(strip_accents)

    there = list(df_fr.itertuples(index=False, name=None))
    where = where + there

    df_MEGA = pd.DataFrame(where, columns=['PlayerName', 'Embeddings', "ImagePath"])
    df_MEGA.to_pickle("embeddings_tm+fr+data_2.pkl")


if __name__ == '__main__':
    combiner()  # Created Thanos level embeddings list i.e. MEGA_EMBEDDINGS + TM EMBEDDINGS + FBREF EMBEDDINGS
    # Read all the pickle files first.
    pkl_read_time = time.time()

    df_1 = pd.read_pickle("data/player_facial_embeddings_vggface_tm.pkl")
    pkl_dfs.append(df_1)
    print(f"First Pickle read took {time.time() - pkl_read_time}")

    df_2 = pd.read_pickle("data/player_facial_embeddings_facenet_TM.pkl")
    pkl_dfs.append(df_2)

    df_3 = pd.read_pickle("data/player_facial_embeddings_facenet_FBREF.pkl")
    pkl_dfs.append(df_3)

    print(f"Pickle read took {time.time() - pkl_read_time}")

    # generate_face_embeddings()
