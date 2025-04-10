import cv2
import threading
from filters import *


class VideoProcessor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.current_function = None
        self.is_running = False
        self.frame = None

    def apply_filter(self, frame):
        if self.current_function == "grayscale":
            return grayscale(frame)
        elif self.current_function == "bitwise_not":
            gray = grayscale(frame)
            return bitwise_not(gray)
        elif self.current_function == "sepia":
            return sepia(frame)
        elif self.current_function == "cartoon":
            return make_cartoon(frame)
        elif self.current_function == "sketch":
            return sketch(frame)
        elif self.current_function == "blur":
            return blur(frame)
        elif self.current_function == "negative":
            return negative(frame)
        elif self.current_function == "emboss":
            return emboss(frame)
        elif self.current_function == "edges":
            return edges(frame)
        elif self.current_function == "stylize":
            return stylize(frame)
        elif self.current_function == "enhance":
            return enhance(frame)
        elif self.current_function == "thermal":
            return thermal(frame)
        elif self.current_function == "glitch":
            return glitch(frame)
        return frame

    def process_frame(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame = self.apply_filter(frame)

        self.release_resources()

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.process_frame)
        self.thread.start()

    def stop(self):
        self.is_running = False
        self.thread.join()

    def release_resources(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        return self.frame
