import cv2
from os.path import isfile, join
import numpy as np
import pandas as pd
import os
from PIL import Image
from tensorboard.plugins import projector
import csv

pkl_path = "/data/embeddings_tm+fr+data.pkl"
crop_images_path = "/Documents/MEGA-EMBED/crops"
SIZE_FOR_SPRITES = (100, 100)


def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)

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


def data_to_tsv(df):
    vectors = df['Embeddings'].tolist()  # Load embeddings
    metadata = df['PlayerName'].tolist()  # Load metadata i.e. player names

    with open('mega_mega_embeddings.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for vector in vectors:
            tsv_writer.writerow(vector)

    with open('mega_mega_metadata.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for meta in metadata:
            tsv_writer.writerow([meta])


def main():
    df = pd.read_pickle(pkl_path)

    images = []
    images_paths = os.listdir(crop_images_path)

    for image_path in images_paths:
        if image_path.__contains__("DS_Store"):
            continue

        cvimage = cv2.imread(os.path.join(crop_images_path, image_path))
        img = Image.fromarray(cvimage)
        img = img.convert('RGB')
        image = img.resize(SIZE_FOR_SPRITES)  # For Collage/Sprites.
        images.append(np.asarray(image, dtype="float32"))

    data_to_tsv(df)
    create_and_save_sprite(images)


# def convert():
#     images = []
#     images_paths = os.listdir(crop_images_path)
#
#     for image_path in images_paths:
#         if image_path.__contains__("face-"):
#             dst_path = os.path.join(crop_images_path, image_path)
#             image = cv2.imread(dst_path)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             cv2.imwrite(dst_path, image)


if __name__ == '__main__':
    main()
