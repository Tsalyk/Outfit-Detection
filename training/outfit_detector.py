import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers.experimental.preprocessing import Resizing, Rescaling, RandomFlip, RandomRotation
from keras import backend as K
import matplotlib.pyplot as plt
import os
from data_preprocessing.preprocessor import DataLoader

K.set_image_data_format('channels_last')


class OutfitDetector:
    def __init__(self, train_ds, dev_ds, test_ds, classes, image_shape, channels):
        # useful constants
        self.SEED = 1
        self.IMAGE_SHAPE = image_shape
        self.CHANNELS = channels
        self.BATCH_SIZE = 128
        self.TRAIN_SIZE = 0.8
        self.DEV_SIZE = 0.1
        self.TEST_SIZE = 0.1
        # change it, if you want to make more training of the model
        self.EPOCHS = 0
        self.train_ds, self.dev_ds, self.test_ds = train_ds, dev_ds, test_ds
        self.classes = classes

    def preprocessing_layers(self):
        """
        Performs data augmentation

        Resizing -> Rescailing -> Flip -> Rotation
        """
        return tf.keras.Sequential([
            Resizing(*self.IMAGE_SHAPE),
            Rescaling(1.0/255),
            RandomFlip('horizontal_and_vertical'),
            RandomRotation(0.2)
        ])

    def CNN(self):
        """
        Architecture of the Convoulutional Neural Network

        Resizing -> Rescailing -> Flip -> Rotation -> 
        -> Conv2D -> MaxPooling2D -> Conv2D -> MaxPooling2D ->
        -> Conv2D -> MaxPooling2D -> Conv2D -> MaxPooling2D ->
        -> Conv2D -> MaxPooling2D -> Conv2D -> MaxPooling2D ->
        -> Flatten -> FC RELU -> FC SOFTMAX
        """
        input_shape = (self.BATCH_SIZE,) + self.IMAGE_SHAPE + (self.CHANNELS,)

        model = models.Sequential([
            self.preprocessing_layers(),
            Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(24, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        model.build(input_shape=input_shape)

        return model

    def train(self, model):
        """
        Performs training with Adam optimizer
        """
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        history = model.fit(
            self.train_ds,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            verbose=1,
            validation_data=self.dev_ds
        )

        return history

    def retrain(self, model):
        """
        Performs retraining using computed weights
        """
        # extract latest version
        version = max([int(i) for i in os.listdir('models') if len(i) == 1])
        model.load_weights(f'weights/{version}')

        return self.train(model)

    def evaluate(self, model):
        """
        Returns accuracy of the model peforming on the test set
        """
        return model.evaluate(self.test_ds)

    def predict_outfit(self, model, img):
        """
        Returns predicted label and probability of this label for a single image
        """
        # convert an image to tensor array;
        # create a batch with single image
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        predicted_class = self.classes[np.argmax(predictions[0])]
        prob = round(100 * (np.max(predictions[0])), 2)

        return predicted_class, prob

    def plot_predicted_sample(self, model):
        """
        Plots sample of predicted images
        """
        plt.figure(figsize=(10, 10))

        for images, labels in self.test_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i+1)
                plt.imshow(images[i].numpy().astype('uint8'))
                plt.axis('off')

                predicted_class, prob = self.predict_outfit(model, images[i].numpy())
                actual_class = self.classes[labels[i]]

                plt.title(f'Actual: {actual_class}\n Predicted: {predicted_class}')

        plt.show()

    def plot_results(self, history, metric):
        """
        Plots comparing of Training/Validation performance
        : metric: ['loss', 'accuracy']
        """
        plt.plot(range(self.EPOCHS), history[metric], label='Training '+metric)
        plt.plot(range(self.EPOCHS), history['val_'+metric], label='Validation '+metric)
        plt.legend()
        plt.title('Traininig/Validation '+metric)
        plt.show()

    def save_model(self, model):
        """
        Saves the latest trained model as a new version
        """
        # save the model as the latest version
        model_v = max([int(i) for i in os.listdir('../models') if i.isnumeric()])+1
        model.save(f'../models/{model_v}')
        model.save_weights(f'../weights/{model_v}')


if __name__ == "__main__":
    dl = DataLoader('ClothesData')
    train_ds, dev_ds, test_ds = dl.get_data()

    detector = OutfitDetector(train_ds, dev_ds, test_ds, dl.classes, (128, 128), 3)

    model = detector.CNN()
    detector.retrain(model) # now model weights are loaded and it is ready to be used
