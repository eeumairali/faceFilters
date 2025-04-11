import cv2
import mediapipe as mp
import numpy as np

def warp_jaw(frame, landmarks, w, h):
    jaw_ids = list(range(0, 17))  # jawline landmark indices
    src_points = np.float32([(landmarks[i].x * w, landmarks[i].y * h) for i in jaw_ids])
    dst_points = src_points.copy()

    # Stretch each point outward from face center
    center_x = w / 2
    for i in range(len(dst_points)):
        dx = (dst_points[i][0] - center_x) * 0.4  # exaggerate
        dst_points[i][0] += dx

    # Generate mesh for warping
    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)
    warped_frame = frame.copy()

    for i in range(len(src) - 1):
        pt1_src = np.array([src[i], src[i + 1], [src[i + 1][0], h], [src[i][0], h]], np.float32)
        pt1_dst = np.array([dst[i], dst[i + 1], [dst[i + 1][0], h], [dst[i][0], h]], np.float32)

        matrix = cv2.getPerspectiveTransform(pt1_src, pt1_dst)
        warped = cv2.warpPerspective(frame, matrix, (w, h))

        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.fillConvexPoly(mask, pt1_dst.astype(int), (255, 255, 255))

        warped_frame = np.where(mask == 255, warped, warped_frame)

    return warped_frame

def gigachad_filter():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible.")
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
                frame = warp_jaw(frame, face_landmarks.landmark, w, h)

        cv2.imshow("GigaChad Filter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

gigachad_filter()
