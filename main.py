from tkinter import Tk
from video_processor import VideoProcessor
from app import App


def main():
    processor = VideoProcessor()
    root = Tk()
    app = App(root, processor)
    root.mainloop()


if __name__ == "__main__":
    main()
