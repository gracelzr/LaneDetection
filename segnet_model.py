
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


def create_model(input_shape, pool_size):

    model = Sequential()

    # Normalizes incoming inputs. First layer needs the input shape to work
    model.add(BatchNormalization(input_shape=input_shape))

    # encoder
    model.add(Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'))

    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5'))
    model.add(Dropout(0.2))

    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7'))
    model.add(Dropout(0.2))

    model.add(MaxPooling2D(pool_size=pool_size))

    # decoder
    model.add(UpSampling2D(size=pool_size))

    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1'))
    model.add(Dropout(0.2))
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2'))
    model.add(Dropout(0.2))

    model.add(UpSampling2D(size=pool_size))

    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3'))
    model.add(Dropout(0.2))
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4'))
    model.add(Dropout(0.2))
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5'))
    model.add(Dropout(0.2))

    model.add(UpSampling2D(size=pool_size))

    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6'))
    # Final layer - only including one channel so 1 filter
    model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Final'))

    return model



if __name__ == '__main__':

    pickledImgs = './data/full_train.p'
    pickledGt = './data/full_labels.p'

    # Load pickled dataset
    images = pickle.load(open(pickledImgs, "rb"))
    labels = pickle.load(open(pickledGt, "rb"))

    # Randomly split into 90% train/ 10% val sets
    X_train, X_val, y_train, y_val = train_test_split(np.array(images), np.array(labels), test_size=0.1, random_state=None)

    # Hyper-parameters

    input_shape = X_train.shape[1:]
    pool_size = (2, 2)

    optimizer_alg = "Adam"
    loss_func = "mean_squared_error"
    batch_size = 128
    epochs = 10

    # Create the neural network
    model = create_model(input_shape, pool_size)

    # Using a generator to help the model use less data
    # Channel shifts help with shadows slightly
    datagen = ImageDataGenerator(channel_shift_range=0.3)
    datagen.fit(X_train)

    # Compiling and training the model
    model.compile(optimizer=optimizer_alg, loss=loss_func)
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / batch_size,
                        epochs=epochs, verbose=1, validation_data=(X_val, y_val))

    # Freeze layers since training is done
    model.trainable = False
    model.compile(optimizer=optimizer_alg, loss=loss_func)

    # Save model architecture and weights
    model.save('segnet_model_weights.h5')

    # Show summary of model
    model.summary()
