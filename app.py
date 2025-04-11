import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk
import threading


class App:
    def __init__(self, root, processor):
        self.processor = processor
        self.root = root
        self.root.title("Video Filter App")

        # Create the video display label
        self.video_label = Label(root)
        self.video_label.pack()

        # Create the sidebar
        sidebar = tk.Frame(root)
        sidebar.pack(side="left", fill="y", padx=10, pady=10)

        Label(sidebar, text="Andy Filters").pack(pady=5)

        # Filter buttons
        buttons = [
        ("Grayscale", "grayscale"),
        ("Bitwise NOT", "bitwise_not"),
        ("Sepia", "sepia"),
        ("Cartoon", "cartoon"),
        ("Pencil Sketch", "sketch"),
        ("Blur", "blur"),
        ("Negative", "negative"),
        ("Emboss", "emboss"),
        ("Edge Detect", "edges"),
        ("Stylize", "stylize"),
        ("Detail Enhance", "enhance"),
        ("Thermal Cam", "thermal"),
        ("Glitch", "glitch"),
        ("GigaChad", "gigachad"),  # 👈 Add this line
    ]


        for text, filter_name in buttons:
            Button(sidebar, text=text, command=lambda name=filter_name: self.set_filter(name)).pack(pady=2, side='left', padx=2)

        Button(sidebar, text="Start", command=self.start_video).pack(pady=10)
        Button(sidebar, text="Quit", command=self.quit_app).pack(pady=5)

    def set_filter(self, name):
        self.processor.current_function = name

    def update_video(self):
        frame = self.processor.get_frame()
        if frame is not None:
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)

            self.video_label.configure(image=frame)
            self.video_label.image = frame

        if self.processor.is_running:
            self.root.after(10, self.update_video)  # Update every 10 ms

    def start_video(self):
        self.processor.start()
        self.update_video()

    def quit_app(self):
        self.processor.stop()
        self.root.quit()
        self.root.destroy()
