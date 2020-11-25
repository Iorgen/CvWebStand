import cv2
import random
import numpy as np
from keras import load_model
from statistics import mode
from core.utils.singleton import Singleton
from core.utils.emotion_datasets import get_labels
from core.utils.inference import detect_faces
from core.utils.inference import draw_text
from core.utils.inference import draw_bounding_box
from core.utils.inference import apply_offsets
from core.utils.inference import load_detection_model
from core.utils.preprocessor import preprocess_input
from datetime import datetime, time


def date_diff_in_Seconds(dt2, dt1):
    timedelta = dt2 - dt1
    return timedelta.days * 24 * 3600 + timedelta.seconds


class VideoCamera(metaclass=Singleton):
    def __init__(self, patient):
        self.log_date = datetime.now()
        self.current_patient = patient
        # self.current_emotion = Emotion.select().where(Emotion.id == 6).get()
        self.video = cv2.VideoCapture(0)
        self.emotion_model_path = './models/emotion_model.hdf5'
        self.emotion_labels = get_labels('fer2013')
        # hyper-parameters for bounding boxes shape
        self.frame_window = 10
        self.emotion_offsets = (20, 40)
        # loading models
        self.face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
        self.emotion_classifier = load_model(self.emotion_model_path)
        # getting input model shapes for inference
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
        # starting lists for calculating modes
        self.emotion_window = []

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, bgr_image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        # faces = face_cascade.detectMultiScale(image, 1.3, 5)
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #     crop = image[y:y + h, x:x + w]
        #     cv2.imwrite(str(random.random()) +  'some_image.png', crop)

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                                                   minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, self.emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (self.emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            print(x1, x2, y1, y2)
            emotion_prediction = self.emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            print(date_diff_in_Seconds(datetime.now(), self.log_date))

            emotion_text = self.emotion_labels[emotion_label_arg]
            self.emotion_window.append(emotion_text)

            if len(self.emotion_window) > self.frame_window:
                self.emotion_window.pop(0)
            try:
                emotion_mode = mode(self.emotion_window)
            except:
                continue

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()
            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_mode,
                      color, 0, -45, 1, 1)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode('.jpg', bgr_image)
        return jpeg.tobytes()
