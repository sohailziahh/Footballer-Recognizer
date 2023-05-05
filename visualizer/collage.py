import multiprocessing

import cv2
from os.path import isfile, join
import numpy as np
import csv
import glob
import os
import time
import cv2
import pandas as pd
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from collections import Counter
from PIL import Image
import shutil
import sys
import face_recognition

FACE_DETECTOR = 'mtcnn'
MODEL = "Facenet"  # VGG-Face or Facenet

pkl_path = "/data/embeddings_tm+fr+data.pkl"
crop_images_path = "/Downloads/google_images"
save_path = "/Downloads/cropped"
SIZE_FOR_SPRITES = (100, 100)


def extract_face_from_image(image_path, filename, required_size=(224, 224)):
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
        face_image.save(f"{filename}.jpg")

    return face_images


def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


def create_and_save_sprite(images):
    sprite = images_to_sprite(np.asarray(images))
    cv2.imwrite('mega_mega_sprite.png', sprite)


def embeddings_to_tsv(df):
    text = []
    for row in df['Embeddings']:
        with open('embeddings-tf.tsv', 'a') as f:
            for val in row:
                text.append(str(val) + '\t')
                text.append(f'{val}\t')
                f.write(f'{val}\t')
            f.write("\n")


TM_PATH = "/Documents/headshots_tm"
FBREF_PATH = "/Documents/Dataset_fbref/headshots"


def data_to_tsv(df):
    vectors = df['Embeddings'].tolist()  # Load embeddings
    metadata = df['PlayerName'].tolist()  # Load metadata i.e. player names
    image_paths = df['ImagePath'].tolist()
    images = []

    for image_path in image_paths:
        if image_path.split("_")[0].__eq__("TM"):
            image_path = os.path.join(TM_PATH, image_path[3:])
        elif image_path.split("_")[0].__eq__("FBREF"):
            image_path = os.path.join(FBREF_PATH, image_path[6:])
        #image = Image.open(image_path)
        cvimage = cv2.imread(image_path)
        image = Image.fromarray(cvimage)
        image = image.convert('RGB')
        image = image.resize(SIZE_FOR_SPRITES)  # For Collage/Sprites.
        images.append(np.asarray(image, dtype="float32"))

    sprite = images_to_sprite(np.asarray(images))
    cv2.imwrite('mega_mega_sprite_2.png', sprite)

    with open('mega_mega_embeddings_2.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for vector in vectors:
            tsv_writer.writerow(vector)

    with open('mega_mega_metadata_2.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for meta in metadata:
            tsv_writer.writerow([meta])


def main():

    df_mega = pd.read_pickle("/data/embeddings_tm+fr+data_2.pkl")
    data_to_tsv(df_mega)



if __name__ == '__main__':
    main()

