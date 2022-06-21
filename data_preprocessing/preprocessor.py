import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class DataLoader:
    def __init__(self, path: str):
        self.SEED = 1
        self.IMAGE_SHAPE = 128, 128
        self.BATCH_SIZE = 128
        self.TRAIN_SIZE = 0.8
        self.DEV_SIZE = 0.1
        self.TEST_SIZE = 0.1
        self.ds = self.load_ds(path)
        self.classes = self.ds.class_names

    def load_ds(self, path: str):
        """
        Data loading
        """
        ds = tf.keras.preprocessing.image_dataset_from_directory(
                                                    path,
                                                    shuffle=True,
                                                    image_size=self.IMAGE_SHAPE,
                                                    batch_size=self.BATCH_SIZE,
                                                    seed=self.SEED
                                                )

        return ds

    def partition_data(self, shuffle=True, shuffle_size=1000):
        """
        Splits data on training, developing, testing sets
        """
        if shuffle:
            self.ds = self.ds.shuffle(shuffle_size, self.SEED)

        n = len(self.ds)

        train_ds = self.ds.take(int(n * self.TRAIN_SIZE))
        dev_test_ds = self.ds.skip(int(n * self.TRAIN_SIZE))
        dev_ds = dev_test_ds.take(int(n * self.DEV_SIZE))
        test_ds = dev_test_ds.skip(int(n * self.DEV_SIZE))

        return train_ds, dev_ds, test_ds

    def cache_ds(self, train_ds, dev_ds, test_ds, shuffle_size=1000):
        """
        Saves images for faster recovering during CNN training
        """
        return train_ds.cache().shuffle(shuffle_size).prefetch(buffer_size=tf.data.AUTOTUNE),\
                dev_ds.cache().shuffle(shuffle_size).prefetch(buffer_size=tf.data.AUTOTUNE),\
                test_ds.cache().shuffle(shuffle_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    def get_data(self):
        """
        Returns training, developing, testing datasets
        """
        return self.cache_ds(*self.partition_data())

    def plot_actual_sample(self):
        """
        Plots actual sample of images
        """
        plt.figure(figsize=(10, 10))
        for image_batch, label_batch in self.ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i+1)
                plt.imshow(image_batch[i].numpy().astype('uint8'))
                plt.title(self.classes[label_batch[i]])
                plt.axis('off')
        plt.show()
