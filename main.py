import os
import tkinter as tk
from app_parts.video_processor import VideoProcessor
from app_parts.app import App


def main():
    # Create root window
    root = tk.Tk()
    
    # Set window icon if available
    try:
        root.iconbitmap("icons/app_icon.ico")
    except:
        pass  # Icon not found, use default
        
    # Create processor and app
    processor = VideoProcessor()
    app = App(root, processor)
    
    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    main()