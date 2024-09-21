import cv2 as cv
import numpy as np
import os
import mediapipe as mp
import time
import math
import socket
from tensorflow.keras.models import load_model

host = "192.168.4.1"
port = 80

class SocketCommunicator:
    def __init__(self, host, port) -> None:
        self.host = host
        self.port = port
        self.socket = None
        self.connect()

    def connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((self.host, self.port))
            print("Terkoneksi dengan kursi roda")
            self.socket = s
        except socket.error:
            print("Mode Remote anjay")
            
    def send(self, data):
        if self.socket:
            self.socket.send(data)

s = SocketCommunicator(host, port)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

actions = np.array(['Kanan', 'Maju', 'Stop', 'Mundur', 'Kiri'])

sequence = []
predictions = []

def media_pipe_detection(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image, results

def draw_land_marks(image, results):
    custom_pose_connections = list(mp_pose.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, connections=custom_pose_connections)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints_normalize(results):
    midpoint_shoulder_x, midpoint_shoulder_y = 0, 0
    shoulder_length = 1

    if results.pose_landmarks:
        left_shoulder = results.pose_landmarks.landmark[11]
        right_shoulder = results.pose_landmarks.landmark[12]
        midpoint_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        midpoint_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        shoulder_length = math.sqrt((left_shoulder.x - right_shoulder.x) ** 2 + (left_shoulder.y - right_shoulder.y) ** 2)

        selected_pose_landmarks = results.pose_landmarks.landmark[11:23]
        pose = np.array([[(res.x - midpoint_shoulder_x) / shoulder_length,
                          (res.y - midpoint_shoulder_y) / shoulder_length] for res in selected_pose_landmarks]).flatten()
    else:
        pose = np.zeros(12 * 2)

    if results.left_hand_landmarks:
        left_hand = np.array([[(res.x - midpoint_shoulder_x) / shoulder_length,
                               (res.y - midpoint_shoulder_y) / shoulder_length] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand = np.zeros(21 * 2)

    if results.right_hand_landmarks:
        right_hand = np.array([[(res.x - midpoint_shoulder_x) / shoulder_length,
                                (res.y - midpoint_shoulder_y) / shoulder_length] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand = np.zeros(21 * 2)

    return np.concatenate([pose, left_hand, right_hand])

def load_lstm_model():
    model = load_model('elmantap.keras')  
    return model

lstm_model = load_lstm_model()

cap = cv.VideoCapture(0)
sequence = []
threshold = 0.4

prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    ret, frame = cap.read()

    new_frame_time = time.time()

    image, results = media_pipe_detection(frame, holistic_model)
    draw_land_marks(image, results)

    keypoints = extract_keypoints_normalize(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    if len(sequence) == 30:
        res = lstm_model.predict(np.expand_dims(sequence, axis=0))[0]
        
        if res[np.argmax(res)] > threshold:
            predicted_class = actions[np.argmax(res)]
            
            if predicted_class == 'Kanan':
                cv.putText(image, 'Kanan', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                arah = 'A\n'
                s.send(arah.encode('utf-8'))
            elif predicted_class == 'Maju':
                cv.putText(image, 'Maju', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                arah = 'B\n'
                s.send(arah.encode('utf-8'))
            elif predicted_class == 'Stop':
                cv.putText(image, 'Stop', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                arah = 'C\n'
                s.send(arah.encode('utf-8'))
            elif predicted_class == 'Mundur':
                cv.putText(image, 'Mundur', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
                arah = 'D\n'
                s.send(arah.encode('utf-8'))
            elif predicted_class == 'Kiri':
                cv.putText(image, 'Kiri', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)
                arah = 'E\n'
                s.send(arah.encode('utf-8'))

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    cv.putText(image, f'FPS: {int(fps)}', (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    cv.imshow('Smart Wheelchair Control', image)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
