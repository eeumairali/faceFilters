from tkinter import Tk
from app_parts.video_processor import VideoProcessor
from app_parts.app import App


def main():
    processor = VideoProcessor()
    root = Tk()
    app = App(root, processor)
    root.mainloop()


if __name__ == "__main__":
    main()
