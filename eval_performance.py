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
import csv
import utils
import pickle

FACE_DETECTOR = 'skip'
MODELS = ["Facenet", "VGG-Face", "ArcFace", "dlib"]
DATA_SOURCES_ANCHOR = ["TM", "FBREF"]
DISTANCE_METRICS = ["cosine", "euclidean"]

TEST_SET_PATH = "test-set"
UNRECOGNIZED = "unrecognized"


def check_for_multiple_faces():
    return True
    dirs = os.listdir(TEST_SET_PATH)
    troublesome_images = []
    for dir_name in dirs:
        image_paths = os.listdir(os.path.join(TEST_SET_PATH, dir_name))
        for image_path in image_paths:
            if image_path.__contains__(".DS_Store"):
                continue
            image_path = os.path.join(TEST_SET_PATH, dir_name, image_path)  # Absolute path.
            print(image_path)
            faces = utils.extract_face_from_image(image_path)
            if len(faces) > 1:
                troublesome_images.append(image_path)

    if troublesome_images.__sizeof__() > 0:
        print("These Images have more than faces. Get rid of them.")
        print(troublesome_images)
        return False

    elif troublesome_images.__sizeof__() == 0:
        return True


def recognize_player(image_path, df, model, distance_metric="cosine"):
    candidates = list()

    embeddings = DeepFace.represent(img_path=image_path, detector_backend=FACE_DETECTOR, model_name=model,
                                    model=model)

    start_time = time.time()
    for i, player_info in enumerate(list(df.itertuples(index=False, name=None))):
        if distance_metric.__eq__("cosine"):
            sim_score = cosine(embeddings, player_info[1])
        else:
            sim_score = np.linalg.norm(np.asarray(embeddings) - np.array(player_info[1]))
        candidates.append((player_info[0], sim_score))

    top_5_candidates = sorted(candidates, key=lambda x: x[1])[:5]  # get first 5
    [print(candidate) for candidate in top_5_candidates]
    print(f"Execution Time: {time.time() - start_time}")
    if top_5_candidates[0][1] >= 0.45:
        return top_5_candidates[0][0]
    else:
        print(f"Aaaaand the player is {top_5_candidates[0][0]}")
        return top_5_candidates[0][0]





def recognize_player_hybrid(image_path, df, model_name, model):

    main_cand_list = list()
    candidates = list()
    start_time = time.time()

    embeddings = DeepFace.represent(img_path=image_path, model=model, detector_backend=FACE_DETECTOR, model_name=model_name)
    print(f"first inference took {time.time() - start_time}")

    stt = time.time()
    for i, row in df[['PlayerName', 'Embeddings']].iterrows():
        sim_score = cosine(embeddings, row['Embeddings'])
        candidates.append((row['PlayerName'], sim_score))
    print(f"first round took {time.time() - stt}")

    top_3_candidates = sorted(candidates, key=lambda x: x[1])[:3]  # get first 5

    print(top_3_candidates)

    main_cand_list.extend([x[0] for x in top_3_candidates])

    main_cand_list = [x.replace('-', ' ').lower() for x in main_cand_list]
    counts = Counter(main_cand_list)
    print(counts)
    answer = max(counts, key=counts.get)

    print(f"Aaaaand the player is {answer}")
    print(f"Execution Time: {time.time() - start_time}")


def recognize_player_dlib(image, face_box, image_path, df, distance_metric="cosine"):
    candidates = list()

    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # boxes = face_recognition.face_locations(image, model='cnn')

    start_time = time.time()
    embeddings = face_recognition.face_encodings(image, [face_box])[0]
    print(f"Execution Time: {time.time() - start_time}")

    print(f"File name: {image_path}")
    start_time = time.time()
    for i, player_info in enumerate(list(df.itertuples(index=False, name=None))):
        if distance_metric.__eq__("cosine"):
            sim_score = cosine(np.asarray(embeddings), np.asarray(player_info[1]))
        else:
            sim_score = np.linalg.norm(np.asarray(embeddings) - np.array(player_info[1]))
        candidates.append((player_info[0].replace("-", " "), sim_score))

    top_5_candidates = sorted(candidates, key=lambda x: x[1])[:5]  # get first 5
    print(f"Execution Time: {time.time() - start_time}")
    if top_5_candidates[0][1] >= 0.05:
        return top_5_candidates[0][0]
    else:
        print(f"Aaaaand the player is {top_5_candidates[0][0]}")
        return top_5_candidates[0][0]


def setup_stats_counter():
    dirs = os.listdir(TEST_SET_PATH)

    stats_counter = dict()

    for dir_name in dirs:
        if (dir_name.__contains__(".DS_Store")):
            continue
        image_paths = os.listdir(os.path.join(TEST_SET_PATH, dir_name))
        internal_stats_counter = dict()
        total_files = len(image_paths)
        internal_stats_counter['Total'] = total_files
        internal_stats_counter['Num_Correct'] = 0
        internal_stats_counter['Num_Wrong'] = 0

        stats_counter[dir_name] = internal_stats_counter
    return stats_counter


def get_combinations():
    combos = []
    for i in MODELS:
        for j in DATA_SOURCES_ANCHOR:
            for k in DISTANCE_METRICS:
                combos.append([i, j, k])
    return combos


def initiate_perf_evaluation():
    dirs = os.listdir(TEST_SET_PATH)

    configurations = get_combinations()

    stats_counter_list = []

    if "dlib" in DATA_SOURCES_ANCHOR:
        df_headshots_dlib = pickle.loads(open('fbref_tm_fttr_ggle_encodings.pickle', "rb").read())
        df_headshots_dlib = df_headshots_dlib[df_headshots_dlib['encoding_fbref'] != '']
        df_headshots_dlib = df_headshots_dlib[df_headshots_dlib['encoding_tm'] != '']
        df_headshots_dlib = df_headshots_dlib[df_headshots_dlib['encoding_tm'] != np.NaN]
        df_headshots_dlib = df_headshots_dlib.reset_index(drop=True)
        df_headshots_dlib = df_headshots_dlib.rename(columns={'encoding_tm': 'TM', 'encoding_fbref': 'FBREF'})

    for model_name, data_src, distance_metric in configurations:

        print(f"Model Name {model_name}")
        print(f"DataSrc -  {model_name}")

        if not model_name.__eq__("dlib"):

            model = DeepFace.build_model(model_name)

            pkl_file = [ls for ls in os.listdir(os.path.join("data", data_src )) if ls.__contains__(model_name.replace("-", "").lower()) ][0]
            print(f"PklFile -  {pkl_file}")

            df = pd.read_pickle(os.path.join("data", data_src, pkl_file))

        else:

            df = df_headshots_dlib
            df = df[['name', data_src]].dropna().reset_index(drop=True)

        stats_counter = setup_stats_counter()

        num_counts = 0
        total = 0

        for dir_name in dirs:
            if dir_name.__contains__(".DS_Store"):
                continue
            image_paths = os.listdir(os.path.join(TEST_SET_PATH, dir_name))

            for image_path in image_paths:
                if image_path.__contains__(".DS_Store"):
                    continue
                total += 1
                image_path = os.path.join(TEST_SET_PATH, dir_name, image_path)  # Absolute path.

                if model_name.__eq__("dlib"):

                    image, face_box = utils.get_face_box(image_path)
                    res = recognize_player_dlib(image, face_box[0], image_path, df, distance_metric)

                else:

                    face = utils.extract_face_from_image(image_path)[0]
                    res = recognize_player(face, df, model, distance_metric)

                if res.lower().__eq__(dir_name.lower()):
                    num_counts += 1
                    stats_counter[dir_name]["Num_Correct"] = stats_counter[dir_name]["Num_Correct"] + 1
                else :
                    stats_counter[dir_name]["Num_Wrong"] = stats_counter[dir_name]["Num_Wrong"] + 1

        print(stats_counter)
        stats_counter['Model'] = model_name
        stats_counter['DataSource'] = data_src
        stats_counter['DistanceMetric'] = distance_metric
        stats_counter['Accuracy'] = f"{num_counts}/{total} -- {num_counts/total}"
        stats_counter_list.append(stats_counter)

    with open('stats.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for stats_count in stats_counter_list:
            for key, value in stats_count.items():
                writer.writerow([key, value])
            writer.writerow("")


if __name__ == '__main__':
    if check_for_multiple_faces():
        initiate_perf_evaluation()
