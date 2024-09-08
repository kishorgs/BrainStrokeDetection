import tkinter as tk
from tkinter import filedialog, Label, messagebox, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

class StrokeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Stroke Detection")
        self.root.geometry("800x600")
        self.root.configure(bg="#E9F3FB")  

        custom_font = ("Helvetica", 18, "bold")
        
        label = tk.Label(root, text="Brain Stroke Detection Using Machine Learning", bg='#E9F3FB', font=custom_font, fg='#0B2F9F')
        label.pack(pady=20)

        self.panel_frame = tk.Frame(self.root, bg="#FFFFFF", bd=2, relief="solid", highlightbackground="#0B2F9F", highlightthickness=2)
        self.panel_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=600, height=400)

        self.inner_frame = tk.Frame(self.panel_frame, bg="#FFFFFF", relief="flat", bd=0)
        self.inner_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=596, height=396)

        self.inner_frame.config(highlightbackground="#0B2F9F", highlightthickness=2)

        self.load_placeholder_image()

        self.label = tk.Label(self.inner_frame, text="Drag & Drop\nor Browse", font=("Arial", 16), fg="#295F98", bg="#FFFFFF", justify=tk.CENTER)
        self.label.place(relx=0.5, rely=0.65, anchor=tk.CENTER)

        self.large_image_label = tk.Label(self.root, bg="#E9F3FB")
        self.large_image_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER, width=500, height=300)
        self.large_image_label.place_forget()

        self.support_text = tk.Label(self.inner_frame, text="Supports: JPEG, JPG, PNG", font=("Arial", 10), fg="#A0A0A0", bg="#FFFFFF")
        self.support_text.place(relx=0.5, rely=0.85, anchor=tk.CENTER)

        self.image_reference = None

        self.result_label = tk.Label(self.root, text="", font=("Arial", 16), fg="green", bg='#FFFFFF')
        self.result_label.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

        self.retry_button = tk.Button(self.root, text="Retry", command=self.reset, font=("Arial", 14), bg="#295F98", fg="#FFFFFF", padx=20)
        self.retry_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)
        self.retry_button.place_forget()

        self.inner_frame.bind("<Button-1>", lambda e: self.load_image())

    def load_placeholder_image(self):
        try:
            self.icon_img = Image.open("img/placeholder.jpg").resize((150, 150))
            self.icon_img_tk = ImageTk.PhotoImage(self.icon_img)
            self.icon_label = tk.Label(self.inner_frame, image=self.icon_img_tk, bg="#FFFFFF")
            self.icon_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
            self.image_reference = self.icon_img_tk
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            try:
                selected_image = Image.open(file_path).resize((500, 300))
                selected_image_tk = ImageTk.PhotoImage(selected_image)

                self.large_image_label.place(x=150, y=120)
                self.large_image_label.config(image=selected_image_tk)
                self.large_image_label.image = selected_image_tk
                
                self.icon_label.place_forget()
                self.label.place_forget()

                prediction_result = self.predict_stroke(file_path)
                self.result_label.config(text=f"Prediction: {prediction_result}")

                self.retry_button.config(state=tk.NORMAL)
                self.retry_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def predict_stroke(self, image_path):
        try:
            image = Image.open(image_path).resize((224, 224)).convert('RGB')
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            model = tf.keras.models.load_model('stroke_detection_model.h5')
            prediction = model.predict(image_array)
            
            if prediction[0] > 0.5:
                return "Stroke"
            else:
                return "No Stroke"
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
            return "Error"

    def reset(self):
        self.label.place(relx=0.5, rely=0.65, anchor=tk.CENTER)
        self.icon_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
        self.large_image_label.config(image='')
        self.result_label.config(text="")
        self.retry_button.place_forget()

# Initialize Tkinter window
root = tk.Tk()
app = StrokeDetectionApp(root)
root.mainloop()
