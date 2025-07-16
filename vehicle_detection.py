import numpy as np
import cv2
import time
import threading
import tkinter as tk
from tkinter import Button, filedialog, Label

# Global variables
video_capture = None
start_video = False
vehicle_count = {
    "Car": 0,
    "SUV": 0,
    "Truck/Bus": 0,
    "Motorcycle": 0,
    "Unknown": 0
}

def select_video():
    global video_capture, start_video, vehicle_count
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        video_capture = cv2.VideoCapture(file_path)
        start_button.config(state=tk.NORMAL)
        # Reset vehicle count
        for key in vehicle_count:
            vehicle_count[key] = 0
        update_vehicle_count()

def process_video():
    global start_video, video_capture
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    while start_video:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        fg_mask = bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        edges = cv2.Canny(fg_mask, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 50000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                if 1.0 < aspect_ratio < 2.0 and area < 10000:
                    vehicle_type = "Car"
                elif 2.0 <= aspect_ratio < 3.5 and 10000 <= area < 30000:
                    vehicle_type = "SUV"
                elif aspect_ratio >= 3.5 and area >= 30000:
                    vehicle_type = "Truck/Bus"
                elif aspect_ratio < 1.0 and area < 10000:
                    vehicle_type = "Motorcycle"
                else:
                    vehicle_type = "Unknown"

                vehicle_count[vehicle_type] += 1
                update_vehicle_count()

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, vehicle_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Vehicle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    reset_to_download()

def update_vehicle_count():
    count_text = "\n".join([f"{k}: {v}" for k, v in vehicle_count.items()])
    vehicle_count_label.config(text=count_text)

def start_video_processing():
    global start_video
    start_video = True
    threading.Thread(target=process_video, daemon=True).start()
    start_button.config(state=tk.DISABLED)

def reset_to_download():
    global start_video, video_capture
    start_video = False
    if video_capture:
        video_capture.release()
    start_button.config(state=tk.DISABLED)
    download_button.config(state=tk.NORMAL)

def exit_app():
    root.quit()

# Create GUI window
root = tk.Tk()
root.title("Vehicle Detection")
root.geometry("400x300")

download_button = Button(root, text="Download Video", command=select_video, font=("Arial", 14))
download_button.pack(pady=10)

start_button = Button(root, text="Start", command=start_video_processing, font=("Arial", 14), state=tk.DISABLED)
start_button.pack(pady=10)

vehicle_count_label = Label(root, text="", font=("Arial", 12))
vehicle_count_label.pack(pady=10)

exit_button = Button(root, text="Exit", command=exit_app, font=("Arial", 14))
exit_button.pack(pady=10)

update_vehicle_count()
root.mainloop()
