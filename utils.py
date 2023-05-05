import cv2
from mtcnn import MTCNN
from numpy import asarray
from PIL import Image


def get_face_box(image_path, required_size=(224, 224)):
    # load image and detect faces
    image = cv2.imread(image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detector = MTCNN()
    faces = detector.detect_faces(image)

    face_images = []

    for i, face in enumerate(faces):
        # extract the bounding box from the requested face
        if face['confidence'] < 0.98:
            continue
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        face_images.append((y1, x2, y2, x1))

    return image, face_images


def extract_face_from_image(image_path, required_size=(224, 224)):
    # load image and detect faces
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    face_images = []

    for i, face in enumerate(faces):
        # extract the bounding box from the requested face
        if face['confidence'] < 0.99:
            continue
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face_boundary = image[y1:y2, x1:x2]
        # resize pixels to the model size
        face_image = cv2.resize(face_boundary, required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)

        #Image.fromarray(face_array).save(f"face-cropped{i}.jpg")

    return face_images