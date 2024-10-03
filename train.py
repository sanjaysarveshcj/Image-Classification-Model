import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import pickle
import numpy as np
import os

def load_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        X = dict[b'data']
        Y = dict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        return X, np.array(Y)

def load_training_data(data_dir):
    X_train = []
    Y_train = []
    for i in range(1, 6):
        file = os.path.join(data_dir, 'data_batch_' + str(i))
        X, Y = load_batch(file)
        X_train.append(X)
        Y_train.append(Y)
    
    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)
    return X_train, Y_train

def load_test_data(data_dir):
    X_test, Y_test = load_batch(os.path.join(data_dir, 'test_batch'))
    return X_test, Y_test

data_dir = 'cifar-10-batches-py' 

X_train, Y_train = load_training_data(data_dir)
X_test, Y_test = load_test_data(data_dir)

X_train = X_train / 255.0
X_test = X_test / 255.0

Y_train = tf.keras.utils.to_categorical(Y_train, 10)
Y_test = tf.keras.utils.to_categorical(Y_test, 10)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_data=(X_test, Y_test))

test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"Test accuracy: {test_acc}")

model.save('cifar10_model.h5')
print('Saved model!')

