from fastapi import FastAPI,UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app=FastAPI()
MODEL=tf.keras.models.load_model("../saved_model/2")
CLASS_NAMES=['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

@app.get("/ping")
async def ping():
    return " Hello, I am Alive"


def read_file_as_image(data)-> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile= File(...)):
    image = read_file_as_image(await file.read())
    img_array=tf.keras.preprocessing.image.img_to_array(image)
    image_batch=np.expand_dims(img_array,0)
    prediction=MODEL.predict(image_batch)
    predicted_class=CLASS_NAMES[np.argmax(prediction[0])]
    confidence=np.max(prediction[0])
    return {"class":predicted_class,"confidence":float(confidence)}


if __name__== "__main__":

    uvicorn.run(app,host="localhost",port=8000)
