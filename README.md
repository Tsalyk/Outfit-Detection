# Outfit-Detection
***

## Aim
Main goal of the project is to build a model, that can classify outfit in images. Next step is to share this model in public usage so that everybody can upload an image of the apparel and get the prediction.
***

## Approach
Deep Learning and Convoulutional Neural Network is a primary part of the model. Neural Network architecture is following:

**Resizing -> Rescailing -> Flip -> Rotation
        -> Conv2D -> MaxPooling2D -> Conv2D -> MaxPooling2D
        -> Conv2D -> MaxPooling2D -> Conv2D -> MaxPooling2D
        -> Conv2D -> MaxPooling2D -> Conv2D -> MaxPooling2D
        -> Flatten -> FC RELU -> FC SOFTMAX**
        
First 4 layers are responsible for data engineering and augmentation. Then there are 6 blocks of Convolution+Relu -> MaxPooling, Flatten layer, followed by Relu activation Fully Connected layer and, finally, Softmax activation Fully Connected layer. Last layer is $\mathbb{R}^{24}$, since there are 24 classes of outfit.
***

## Python dependencies

1. Install Python packages

```
pip3 install -r training/requirements.txt
pip3 install -r api/requirements.txt
```

2. Install Tensorflow Serving ([Setup instructions](https://www.tensorflow.org/tfx/serving/setup))

***
