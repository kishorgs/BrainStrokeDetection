from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("stroke_detection_model.h5")

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224)).convert('RGB')
    image_array = np.array(image) / 255.0 
    image_array = np.expand_dims(image_array, axis=0)  
    return image_array

# Stroke detection endpoint
@app.post("/predict/")
async def predict_stroke(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        image_array = preprocess_image(image)
        
        prediction = model.predict(image_array)
        result = "Stroke" if prediction[0] > 0.5 else "No Stroke"
        
        return {"prediction": result}
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"error": "Failed to process the image."}
