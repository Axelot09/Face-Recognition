# Imports
import cv2
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

# Initializing the main window
root = tk.Tk()
root.title("Face Recognition")
root.geometry("800x600")

# Defining a label to show the video feed
video_label = Label(root)
video_label.pack()

# Loading the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to capture video from webcam and detect faces
def start_video():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converting to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Drawing a rectangle around each face and label it as "Person"
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Converting the frame to ImageTk format
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Updating the video_label with the new frame
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        
        # Updating the GUI
        root.update_idletasks()
        root.update()

    cap.release()
    cv2.destroyAllWindows()

# Start button to begin face recognition
start_button = Button(root, text="Start Video", command=start_video)
start_button.pack(pady=20)

root.mainloop()