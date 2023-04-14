from fastapi import FastAPI,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app=FastAPI()
origins=[
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#endpoint= 'http://localhost:8605/v1/models/emotion_model:predict'
endpoint= 'http://localhost:8605/v1/models/emotion_model/versions/2:predict'
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
    json_data={
        "instances":image_batch.tolist()
    }

    response= requests.post(endpoint,json=json_data)
    prediction=np.array(response.json()["predictions"][0])
    predicted_class=CLASS_NAMES[np.argmax(prediction)]
    confidence=np.max(prediction)
    return {"class":predicted_class,"confidence":float(confidence)}


if __name__== "__main__":

    uvicorn.run(app,host="localhost",port=8000)
