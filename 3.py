
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.constraints import max_norm
from keras.datasets import mnist
from keras.utils import to_categorical, np_utils
from keras_applications.densenet import layers
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation


"""Loading in the MNIST dataset"""
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train_WithoutNoise = X_train;
X_test_WithoutNoise = X_test;  #
Y_train_duplicate = Y_train;
Y_test_duplicate = Y_test;  #this is for final evaluations


# add noise to train and test images
noise_factor = 0.25
X_train = X_train + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test = X_test + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
X_train = np.clip(X_train, 0., 1.)
X_test = np.clip(X_test, 0., 1.)


# plot images
num_row = 2
num_col = 5
num = 10
images = X_train[:num]
labels = Y_train[:num]

fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('Label: {}'.format(labels[i]))
plt.tight_layout()
plt.show()


print("X_train shape", X_train.shape)
print("y_train shape", Y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", Y_test.shape)


Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)


print("X_train shape", X_train.shape)
print("y_train shape", Y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", Y_test.shape)


# building the input vector from the 28x28 pixels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255


# building the input vector from the 28x28 pixels
X_train_WithoutNoise = X_train_WithoutNoise.reshape(60000, 784)
X_test_WithoutNoise = X_test_WithoutNoise.reshape(10000, 784)
X_train_WithoutNoise = X_train_WithoutNoise.astype('float32')
X_test_WithoutNoise = X_test_WithoutNoise.astype('float32')

# normalizing the data to help with the training
X_train_WithoutNoise /= 255
X_test_WithoutNoise /= 255


print("X_train shape", X_train.shape)
print("y_train shape", Y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", Y_test.shape)


# Autoencoder to reduce noise
auto_encoder = Sequential()
auto_encoder.add(Dense(784, input_shape=(784,)))
auto_encoder.add(Activation('relu'))
auto_encoder.add(Dense(784, input_shape=(784,)))
auto_encoder.add(Activation('relu'))
auto_encoder.add(Dense(784, input_shape=(784,)))
auto_encoder.add(Activation('softmax'))

auto_encoder.summary()

# compile autoencoder
auto_encoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

auto_encoder.fit(X_train, X_train_WithoutNoise, epochs=3)

denoised_train_images = auto_encoder.predict(X_train)
denoised_test_images = auto_encoder.predict(X_test)


X_train = denoised_train_images
X_test = denoised_test_images


# building a linear stack of layers with the sequential model
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# training the model and saving metrics in history
history = model.fit(X_train, Y_train,
          batch_size=128, epochs=20,
          verbose=2)


# plotting the accuracy
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='lower right')

# plotting the loss
plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')

plt.tight_layout()


loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])


predicted_classes = model.predict_classes(X_test)

# see which we predicted correctly and which not
correct_indices = np.nonzero(predicted_classes == Y_test_duplicate)[0]
incorrect_indices = np.nonzero(predicted_classes != Y_test_duplicate)[0]
print()
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")

# adapt figure size to accomodate 18 subplots
plt.rcParams['figure.figsize'] = (7,14)

figure_evaluation = plt.figure()

# plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted: {}, Truth: {}".format(predicted_classes[correct],
                                        Y_test_duplicate[correct]))
    plt.xticks([])
    plt.yticks([])


# plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6,3,i+10)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted {}, Truth: {}".format(predicted_classes[incorrect],
                                       Y_test_duplicate[incorrect]))
    plt.xticks([])
    plt.yticks([])

figure_evaluation

