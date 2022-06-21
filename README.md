# Outfit-Detection
***

### Aim
Main goal of the project is to build a model, that can classify outfit in images. Next step is to share this model in public usage so that everybody can upload an image of the apparel and get the prediction.
***

## Approach
***

### Architecture
Deep Learning and Convoulutional Neural Network is a primary part of the model. Neural Network architecture is following:

**Resizing -> Rescailing -> Flip -> Rotation
        -> Conv2D -> MaxPooling2D -> Conv2D -> MaxPooling2D
        -> Conv2D -> MaxPooling2D -> Conv2D -> MaxPooling2D
        -> Conv2D -> MaxPooling2D -> Conv2D -> MaxPooling2D
        -> Flatten -> FC RELU -> FC SOFTMAX**
        
First 4 layers are responsible for data engineering and augmentation. Then there are 6 blocks of Convolution+Relu -> MaxPooling, Flatten layer, followed by Relu activation Fully Connected layer and, finally, Softmax activation Fully Connected layer. Last layer is $\mathbb{R}^{24}$, since there are [24 classes of outfit](https://github.com/Tsalyk/Outfit-Detection/blob/main/outfits.py).
***

### Deployment
To run API was used **[RestAPI](https://python-rest-framework.readthedocs.io/en/latest/#) & [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)**.

To deploy a model was used [Google Cloud Platform](https://console.cloud.google.com/marketplace/product/thorn-technologies-public/sftp-gateway?project=thorn-technologies-public&gclid=Cj0KCQjw2MWVBhCQARIsAIjbwoPDr0JVE1PZBEv2QbLG2zrdv3eOROAWKYr4VmZ4W9OarkK6aGvlvUQaApFHEALw_wcB).
***

### Data
Load dataset from [Kaggle](https://www.kaggle.com/datasets/trolukovich/apparel-images-dataset) to launch the application.

It contains *11385* images of different types of outfit.
***

## Usage
***

### Python dependencies

1. Install Python packages

```
pip3 install -r training/requirements.txt
pip3 install -r api/requirements.txt
```

2. Install Tensorflow Serving ([Setup instructions](https://www.tensorflow.org/tfx/serving/setup))
***

### ReactJS dependencies

```bash
cd frontend
npm install --from-lock-json
npm audit fix
```
***

### React-Native dependencies
```bash
cd mobile-app
yarn install
```

  - Only for mac users
```bash
cd ios && pod install && cd ../
```
***

### Postman
1. Open [Postman](https://www.postman.com/downloads/)
2. Use link for deployed function on GCP https://us-central1-clothes-detection-353914.cloudfunctions.net/predict
3. Choose `file` in body section and upload your image
4. Push `execute`

## Running Applications
***

### FastAPI & TF Serve

1. Get inside `api` folder

```bash
cd api
```

2. Run [Docker](https://www.docker.com/products/docker-desktop/) daemon

3. Run the TF Serve (Update config file path below)

```bash
docker run -it -v /path/to/Outfit-Detection:/Outfit-Detection -p 8080:8080 --entrypoint /bin/bash tensorflow/serving
ls -ltr Outfit-Detection
tensorflow_model_server --rest_api_port=8080  --allow_version_labels_for_unavailable_models --model_config_file=/Outfit-Detection/models.config
```

4. Run the FastAPI Server using uvicorn

   For this you can directly run it from main.py
   
   OR you can run it from command prompt as shown below,

```bash
uvicorn main:app --reload --host 0.0.0.0
```

5. API is now running at `0.0.0.0:8080`
***

### Frontend

1. Get inside `frontend` folder

```bash
cd frontend
```

2. Run the frontend

```bash
npm run start
```
***

### Mobile App

1. Get inside `mobile-app` folder

```bash
cd mobile-app
```

2. Run the app (android/IOS)

```bash
npm run android
```

or

```bash
npm run ios
```
***

## Results
***

### Evaluation
The model was trained with *60* epochs. It performs pretty well with ***90%*** accuracy on the test set.
***

### Web-app
<img width="1440" alt="Screenshot 2022-06-21 at 16 22 37" src="https://user-images.githubusercontent.com/73395389/174810118-8b0dccae-05fc-43d6-8aac-57cfb6cd2601.png">
***

### Mobile-app
<img width="250" height="500" alt="Screenshot 2022-06-21 at 13 56 38" src="https://user-images.githubusercontent.com/73395389/174806243-4a990d3e-5302-4203-adf9-657cf594d1c5.png">
***

## Credits
© [Markiian Tsalyk](https://www.linkedin.com/in/markiian-tsalyk-193758224/)

[License](https://github.com/Tsalyk/Outfit-Detection/blob/main/LICENSE)
