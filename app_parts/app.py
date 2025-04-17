import tkinter as tk
from tkinter import Button, Label, Frame, LabelFrame
from PIL import Image, ImageTk
import threading
import cv2


class App:
    def __init__(self, root, processor):
        self.processor = processor
        self.root = root
        self.root.title("Video Filter App")
        self.root.configure(bg='#f0f0f0')  # Light gray background

        # Set window size and position
        window_width = 900
        window_height = 600
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

        # Create main frame
        main_frame = Frame(root, bg='#f0f0f0')
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create frame for video display
        video_frame = Frame(main_frame, bg='black')
        video_frame.pack(side="right", fill="both", expand=True)

        # Create the video display label
        self.video_label = Label(video_frame, bg='black')
        self.video_label.pack(fill="both", expand=True, padx=5, pady=5)

        # Create the sidebar
        sidebar = Frame(main_frame, bg='#e0e0e0', width=200)
        sidebar.pack(side="left", fill="y", padx=10, pady=10)

        # App title
        title_frame = Frame(sidebar, bg='#4CAF50', padx=5, pady=5)
        title_frame.pack(fill="x", pady=5)
        Label(title_frame, text="Andy Filters", font=("Arial", 16, "bold"), 
              fg="white", bg='#4CAF50').pack(pady=5)

        # Create scrollable frame for filter buttons
        canvas = tk.Canvas(sidebar, bg='#e0e0e0', highlightthickness=0)
        scrollbar = tk.Scrollbar(sidebar, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas, bg='#e0e0e0')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Filter categories
        self.create_filter_section(scrollable_frame, "Basic Filters", [
            ("Normal", None),
            ("Grayscale", "grayscale"),
            ("Negative", "negative"),
            ("Sepia", "sepia"),
            ("Blur", "blur"),
        ])
        
        self.create_filter_section(scrollable_frame, "Artistic", [
            ("Cartoon", "cartoon"),
            ("Sketch", "sketch"),
            ("Stylize", "stylize"),
            ("Edge Detect", "edges"),
            ("Emboss", "emboss"),
            ("Detail Enhance", "enhance"),
        ])
        
        self.create_filter_section(scrollable_frame, "Fun Effects", [
            ("Thermal Cam", "thermal"),
            ("Glitch", "glitch"),
            ("Pixel Sort", "pixel_sort"),
            ("Mirror", "mirror"),
            ("Kaleidoscope", "kaleidoscope"),
            ("Time Warp", "time_warp"),
            ("Rainbow", "rainbow"),
            ("Disco Lights", "disco"),
        ])
        
        self.create_filter_section(scrollable_frame, "MediaPipe", [
            ("Face Mesh", "face_mesh"),
            ("Face Contours", "face_contours"),
            ("Pose Detection", "pose"),
            ("Holistic", "holistic"),
        ])

        # Control buttons
        control_frame = Frame(sidebar, bg='#e0e0e0')
        control_frame.pack(fill="x", pady=10)
        
        Button(control_frame, text="Start", command=self.start_video, 
               bg="#4CAF50", fg="white", font=("Arial", 12)).pack(side="left", padx=5, fill="x", expand=True)
        Button(control_frame, text="Quit", command=self.quit_app,
               bg="#f44336", fg="white", font=("Arial", 12)).pack(side="right", padx=5, fill="x", expand=True)

    def create_filter_section(self, parent, title, filters):
        """Create a section of filter buttons with a title"""
        section = LabelFrame(parent, text=title, bg='#e0e0e0', fg='#333333', font=("Arial", 10, "bold"))
        section.pack(fill="x", padx=5, pady=5, ipadx=5, ipady=5)
        
        for i, (text, filter_name) in enumerate(filters):
            button = Button(section, text=text, 
                           command=lambda name=filter_name: self.set_filter(name),
                           bg="#dcdcdc", relief=tk.RAISED,
                           borderwidth=2, font=("Arial", 9))
            button.pack(fill="x", pady=2, padx=5)

    def set_filter(self, name):
        self.processor.current_function = name

    def update_video(self):
        frame = self.processor.get_frame()
        if frame is not None:
            # Convert the frame to RGB format (PIL expects RGB)
            if len(frame.shape) == 2:  # If grayscale
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:  # If already color
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            # Convert to PIL Image and then to PhotoImage
            pil_img = Image.fromarray(frame_rgb)
            tk_img = ImageTk.PhotoImage(image=pil_img)
            
            # Update the label
            self.video_label.configure(image=tk_img)
            self.video_label.image = tk_img  # Keep a reference

        if self.processor.is_running:
            self.root.after(10, self.update_video)  # Update every 10 ms

    def start_video(self):
        self.processor.start()
        self.update_video()

    def quit_app(self):
        self.processor.stop()
        self.root.quit()
        self.root.destroy()