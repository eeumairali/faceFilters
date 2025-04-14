import cv2
import numpy as np

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