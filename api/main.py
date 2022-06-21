from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import cv2
import tensorflow as tf
import numpy as np
import uvicorn
import requests


app = FastAPI()
origins = [
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

production_endpoint = "http://localhost:8080/v1/models/clothes_model/labels/production:predict"
beta_endpoint = "http://localhost:8080/v1/models/clothes_model/labels/beta:predict"

OUTFITS = ['black_dress', 'black_pants', 'black_shirt', 'black_shoes',
            'black_shorts', 'blue_dress', 'blue_pants', 'blue_shirt',
            'blue_shoes', 'blue_shorts', 'brown_pants', 'brown_shoes',
            'brown_shorts', 'green_pants', 'green_shirt', 'green_shoes',
            'green_shorts', 'red_dress', 'red_pants', 'red_shoes', 'white_dress',
            'white_pants', 'white_shoes', 'white_shorts']
OUTFITS = list(map(lambda outfit: outfit.replace('_', ' ').capitalize(), OUTFITS))


def convert_to_array(file):
    """
    Converts input bytes of an image to array and resizes it
    """
    return cv2.resize(np.array(Image.open(BytesIO(file))), (128, 128))


@app.get("/home")
async def home():
    return "CNN model"

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = convert_to_array(await file.read())
    image_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": image_batch.tolist()
    }

    res = requests.post(production_endpoint, json=json_data)

    softmax_out = np.array(res.json()["predictions"][0])
    predicted_outfit = OUTFITS[np.argmax(softmax_out)]
    prob = float(np.max(softmax_out))

    return {
        "outfit": predicted_outfit,
        "probability": prob
        }


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8085)
