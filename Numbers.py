import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from keras.datasets import mnist
from PIL import Image #delete later
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.np_utils import to_categorical


class numbers:
    def __init__(self, image_path=None):
         filepath = "C:/Users/vbacho/Downloads/Untitled.png"
         img = Image.open(filepath)
         img = self.convert_image(img)
         self.model = self.train_model()
         self.predict_image(img)

    def convert_image(self, image):
        #image = image.rotate(270)
        image = image.convert('L')
        image = image.resize((28,28))
        image = np.array(image) / 255.0
        image = image.reshape(1,28,28,1)
        return image

    def train_model(self):
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        train_X, test_X = train_X / 255.0, test_X / 255.0
        train_y = to_categorical(train_y, 10)
        test_y = to_categorical(test_y, 10)
        return self.build_model(train_X, test_X, train_y, test_y)

    def build_model(self, train_X, test_X, train_y, test_y):
        model = Sequential()


        model.add(Conv2D(32, (3,3), activation='relu',input_shape=(28,28,1)))

        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64,(3,3), activation='relu'))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))

        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_X, train_y, epochs=5, batch_size=64, validation_data=(test_X, test_y))

        model.save('my_model.keras')


        return model

    def predict_image(self, image):
        self.model = tf.keras.models.load_model('my_model.keras')
        prediction = self.model.predict(image)
        predicted_label = np.argmax(prediction, axis=1)
        print(f"Predicted label: {predicted_label[0]}")

if __name__ == "__main__":
    numbers()