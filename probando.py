import cv2
import mediapipe as mp
import numpy as np
import time


def eye_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))
    return (d_A + d_B) / (2 * d_C)


def boquita_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[7]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[6]))
    d_C = np.linalg.norm(np.array(coordinates[3]) - np.array(coordinates[5]))
    d_D = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[4]))

    return (d_A + d_B+d_C) / (2 * d_D)


# Captura de camara
cap = cv2.VideoCapture(0)

# Variable
mensaje = "ALERTA"
tiempo = 0
inicio = 0
fin = 0
# funcion dibujo


mp_face_mesh = mp.solutions.face_mesh
mp_dibujo = mp.solutions.drawing_utils
ConfDibu = mp_dibujo.DrawingSpec(thickness=1, circle_radius=1)
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]
index_boquita = [61, 39, 0, 269, 291, 405, 17, 181]
EAR_THRESH = 0.25
MAR_THRESH = 1

with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1)as face_mesh:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        coordinates_left_eye = []
        coordinates_right_eye = []
        coordinates_boquita = []
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:

                mp_dibujo.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, ConfDibu, ConfDibu)
                for index in index_left_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_left_eye.append([x, y])
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                    cv2.circle(frame, (x, y), 1, (128, 0, 250), 1)
                for index in index_right_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_right_eye.append([x, y])
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                    cv2.circle(frame, (x, y), 1, (128, 0, 250), 1)
                for index in index_boquita:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_boquita.append([x, y])
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                    cv2.circle(frame, (x, y), 1, (128, 0, 250), 1)

                ear_left_eye = eye_aspect_ratio(coordinates_left_eye)
                ear_right_eye = eye_aspect_ratio(coordinates_right_eye)
                mar_boquita = boquita_aspect_ratio(coordinates_boquita)
                ear = (ear_left_eye+ear_right_eye)/2
                mar = mar_boquita
                print("MAR:", mar)
                # print("EAR izquierdo:", ear_left_eye,
                # "EAR derecho:", ear_right_eye)
                # print("EAR: ", ear)
                # OJOS CERRADOS O MEDIOS CERRADOS

                if ear < EAR_THRESH:
                    cv2.putText(frame, mensaje, (300, 60),
                                cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 3)

                    # print("Inicio:", inicio)
                # BOCA ABIERTA
                if mar > MAR_THRESH:
                    cv2.putText(frame, mensaje, (300, 60),
                                cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
cap.realease()
cv2.destroyAllWindows()
