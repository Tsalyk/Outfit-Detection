from google.cloud import storage
from PIL import Image
import tensorflow as tf
import numpy as np

BUCKET_NAME = "clothes-detection-model"
OUTFITS = ['black_dress', 'black_pants', 'black_shirt', 'black_shoes',
            'black_shorts', 'blue_dress', 'blue_pants', 'blue_shirt',
            'blue_shoes', 'blue_shorts', 'brown_pants', 'brown_shoes',
            'brown_shorts', 'green_pants', 'green_shirt', 'green_shoes',
            'green_shorts', 'red_dress', 'red_pants', 'red_shoes', 'white_dress',
            'white_pants', 'white_shoes', 'white_shorts']
OUTFITS = list(map(lambda outfit: outfit.replace('_', ' ').capitalize(), OUTFITS))

model = None


def download_blob(bucket: str, src_blob: str, dest_file: str):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket)
    blob = bucket.blob(src_blob)
    blob.download_to_filename(dest_file)

def predict(req):
    global model

    if not model:
        download_blob(BUCKET_NAME, "models/outfit.h5", "/tmp/outfit.h5")
        model = tf.keras.models.load_model("/tmp/outfit.h5")

    print('model loaded')

    img = req.files["file"]
    img = np.array(Image.open(img).convert("RGB").resize((128, 128)))
    img = tf.expand_dims(img, 0)

    softmax_out = model.predict(img)[0]
    predicted_outfit = OUTFITS[np.argmax(softmax_out)]
    prob = round(100 * np.max(softmax_out), 2)

    return {
        "outfit": predicted_outfit,
        "probability": prob
        }
