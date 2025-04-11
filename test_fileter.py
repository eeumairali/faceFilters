import cv2
import mediapipe as mp
import numpy as np

def gigachad_filter():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                points = []
                for i in range(468):
                    pt = face_landmarks.landmark[i]
                    x, y = int(pt.x * w), int(pt.y * h)
                    points.append((x, y))

                # Stretch the jaw region (landmarks 0 to 16)
                for i in range(0, 17):
                    x, y = points[i]
                    dx = int((x - w//2) * 0.3)  # exaggerated stretch from center
                    new_x = x + dx
                    cv2.circle(frame, (new_x, y), 2, (0, 255, 0), -1)
                    frame[y-2:y+2, x-2:x+2] = frame[y-2:y+2, new_x-2:new_x+2]

        cv2.imshow("GigaChad Filter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

gigachad_filter()
