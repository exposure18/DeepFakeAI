import tkinter as tk
from tkinter import filedialog, PhotoImage
from PIL import Image, ImageTk
import os
import numpy as np  # Needed for some operations
# This is the key change: we import the prediction function directly
from predict import predict_image


def open_image():
    """
    Function to open a file dialog, let the user select an image,
    display it, and make a deepfake prediction.
    """
    filepath = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if filepath:
        # Display the image
        img = Image.open(filepath)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)

        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Display "Analyzing..." message while processing
        result_label.config(text="Analyzing...", fg="yellow")
        root.update_idletasks()  # Update GUI immediately

        # Make prediction
        score = predict_image(filepath)  # Calls the function from predict.py

        # Interpret the prediction
        if score is not None:
            if score > 0.5:
                result_text = f"REAL (Confidence: {score * 100:.2f}%)"
                result_color = "lightgreen"
            else:
                result_text = f"FAKE (Confidence: {(1 - score) * 100:.2f}%)"
                result_color = "red"

            result_label.config(text=f"Prediction: {result_text}", fg=result_color)
        else:
            result_label.config(text="Model not loaded. Cannot predict.", fg="orange")


# --- GUI Setup ---
root = tk.Tk()
root.title("SpectreEye DeepFake Detector")
root.geometry("800x700")
root.config(bg="black")

header_label = tk.Label(root, text="SpectreEye DeepFake Detector", font=("Arial", 24, "bold"), fg="cyan", bg="black")
header_label.pack(pady=20)

control_frame = tk.Frame(root, bg="black")
control_frame.pack(pady=10)

open_button = tk.Button(control_frame, text="Open Image to Analyze", command=open_image, font=("Arial", 14),
                        bg="darkgrey", fg="white", relief="raised")
open_button.pack(side=tk.LEFT, padx=10)

image_label = tk.Label(root, bg="black")
image_label.pack(pady=20)

result_label = tk.Label(root, text="No image selected.", font=("Arial", 16, "bold"), fg="white", bg="black")
result_label.pack(pady=10)

root.mainloop()