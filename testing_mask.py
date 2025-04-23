import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Load mask
mask_img = Image.open(r'C:\Users\eeuma\Desktop\students_clients_data\andy\faceFilters\thanos2.png').convert("RGBA")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

# Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # Get face bounding box
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h-80 for lm in landmarks]
        wid  = 90
        x_min, x_max = int(min(xs))-wid, int(max(xs))+wid
        y_min, y_max = int(min(ys))-wid, int(max(ys))+wid

        # Ensure bounding box is within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        # Resize mask to match face region
        face_width = x_max - x_min
        face_height = y_max - y_min
        mask_resized = mask_img.resize((face_width, face_height))
        mask_np = np.array(mask_resized)

        # Prepare alpha blending
        alpha_mask = mask_np[..., 3] / 255.0
        alpha_mask = np.stack([alpha_mask] * 3, axis=-1)
        mask_rgb = mask_np[..., :3]

        # Overlay mask on frame
        roi = frame[y_min:y_max, x_min:x_max]
        blended = (1 - alpha_mask) * roi + alpha_mask * mask_rgb
        frame[y_min:y_max, x_min:x_max] = blended.astype(np.uint8)

    cv2.imshow("Thanos Full Face Filter", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
