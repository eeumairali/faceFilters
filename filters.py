import cv2
import numpy as np

import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

def apply_gigachad_jaw_warp(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return frame

    landmarks = result.multi_face_landmarks[0].landmark
    jaw_ids = list(range(0, 17))
    src_points = np.float32([(landmarks[i].x * w, landmarks[i].y * h) for i in jaw_ids])
    dst_points = src_points.copy()

    center_x = w / 2
    for i in range(len(dst_points)):
        dx = (dst_points[i][0] - center_x) * 0.4
        dst_points[i][0] += dx

    warped_frame = frame.copy()
    for i in range(len(src_points) - 1):
        src_quad = np.array([src_points[i], src_points[i + 1], [src_points[i + 1][0], h], [src_points[i][0], h]], np.float32)
        dst_quad = np.array([dst_points[i], dst_points[i + 1], [dst_points[i + 1][0], h], [dst_points[i][0], h]], np.float32)

        matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)
        warped = cv2.warpPerspective(frame, matrix, (w, h))

        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_quad.astype(int), (255, 255, 255))
        warped_frame = np.where(mask == 255, warped, warped_frame)

    return warped_frame


def grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def bitwise_not(frame):
    return cv2.bitwise_not(frame)


def sepia(frame):
    kernel = np.array([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]])
    return cv2.transform(frame, kernel)


def make_cartoon(frame):
    color = cv2.bilateralFilter(frame, 9, 75, 75)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(blurred, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 10)
    return cv2.bitwise_and(color, color, mask=edges)


def sketch(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    return cv2.divide(gray, 255 - blur, scale=256)


def blur(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)


def negative(frame):
    return cv2.bitwise_not(frame)


def emboss(frame):
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])
    return cv2.filter2D(frame, -1, kernel)


def edges(frame):
    return cv2.Canny(frame, 100, 200)


def stylize(frame):
    return cv2.stylization(frame, sigma_s=60, sigma_r=0.6)


def enhance(frame):
    return cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)


def thermal(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def glitch(frame):
    h, w, _ = frame.shape
    glitch = frame.copy()
    for i in range(5):
        y = np.random.randint(0, h)
        glitch[y:y+1, :] = np.roll(glitch[y:y+1, :], np.random.randint(-20, 20), axis=1)
    return glitch
