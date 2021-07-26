
import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
import tensorflow as tf

def create_model(input_shape, pool_size):
    # Create the actual neural network here
    model = Sequential()
    # Normalizes incoming inputs. First layer needs the input shape to work
    model.add(BatchNormalization(input_shape=input_shape))

    # Below layers were re-named for easier reading of model summary; this not necessary
    # Conv Layer 1
    model.add(Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))

    # Conv Layer 2
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'))

    # Pooling 1
    model.add(MaxPooling2D(pool_size=pool_size))

    # Conv Layer 3
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'))
    model.add(Dropout(0.2))

    # Conv Layer 4
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'))
    model.add(Dropout(0.2))

    # Conv Layer 5
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5'))
    model.add(Dropout(0.2))

    # Pooling 2
    model.add(MaxPooling2D(pool_size=pool_size))

    # Conv Layer 6
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6'))
    model.add(Dropout(0.2))

    # Conv Layer 7
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7'))
    model.add(Dropout(0.2))

    # Pooling 3
    model.add(MaxPooling2D(pool_size=pool_size))

    # Upsample 1
    model.add(UpSampling2D(size=pool_size))

    # Deconv 1
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1'))
    model.add(Dropout(0.2))

    # Deconv 2
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2'))
    model.add(Dropout(0.2))

    # Upsample 2
    model.add(UpSampling2D(size=pool_size))

    # Deconv 3
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3'))
    model.add(Dropout(0.2))

    # Deconv 4
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4'))
    model.add(Dropout(0.2))

    # Deconv 5
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5'))
    model.add(Dropout(0.2))

    # Upsample 3
    model.add(UpSampling2D(size=pool_size))

    # Deconv 6
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6'))

    # Final layer - only including one channel so 1 filter
    model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Final'))

    return model



def decode(serialized_example):
    """
    Parses an image and label from the given `serialized_example`
    :param serialized_example:
    :return:
    """
    RESIZE_IMAGE_HEIGHT = 80
    RESIZE_IMAGE_WIDTH = 160

    features = tf.compat.v1.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'gt_image_raw': tf.compat.v1.FixedLenFeature([], tf.string),
            'gt_binary_image_raw': tf.compat.v1.FixedLenFeature([], tf.string)
        })

    # decode gt image
    gt_image_shape = tf.stack([RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, 3])
    gt_image = tf.compat.v1.decode_raw(features['gt_image_raw'], tf.uint8)
    gt_image = tf.reshape(gt_image, gt_image_shape)

    # decode gt binary image
    gt_binary_image_shape = tf.stack([RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, 1])
    gt_binary_image = tf.compat.v1.decode_raw(features['gt_binary_image_raw'], tf.uint8)
    gt_binary_image = tf.reshape(gt_binary_image, gt_binary_image_shape)

    return gt_image, gt_binary_image




def create_dataset(filepath):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(decode, num_parallel_calls=8)

    # This dataset will go on forever
    dataset = dataset.repeat()

    BATCH_SIZE=179
    # Set the batchsize
    dataset = dataset.batch(BATCH_SIZE)

    # Create an iterator
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

    # Create your tf representation of the iterator
    image, label = iterator.get_next()

    return image, label

def main():


    # Load training images
    train_images = pickle.load(open("./data/full_CNN_train_2.p", "rb" ))

    # Load image labels
    labels = pickle.load(open("./data/full_CNN_labels_2.p", "rb" ))

    # Make into arrays as the neural network wants these
    train_images = np.array(train_images)
    labels = np.array(labels)

    # Normalize labels - training images get normalized to start in the network
    labels = labels / 255

    # Shuffle images along with their labels, then split into training/validation sets
    train_images, labels = shuffle(train_images, labels)
    # Test size may be 10% or 20%
    X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)

    # Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
    batch_size = 128
    epochs = 10
    pool_size = (2, 2)
    input_shape = X_train.shape[1:]

    # Create the neural network
    model = create_model(input_shape, pool_size)

    # Using a generator to help the model use less data
    # Channel shifts help with shadows slightly
    datagen = ImageDataGenerator(channel_shift_range=0.2)
    datagen.fit(X_train)

    # Compiling and training the model
    model.compile(optimizer='Adam', loss='mean_squared_error')
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size,
    epochs=epochs, verbose=1, validation_data=(X_val, y_val))

    # Freeze layers since training is done
    model.trainable = False
    model.compile(optimizer='Adam', loss='mean_squared_error')

    # Save model architecture and weights
    model.save('full_CNN_model_2.h5')

    # Show summary of model
    model.summary()

if __name__ == '__main__':
    main()

#
# if __name__ == '__main__':
#     # Get your datatensors
#     image, label = create_dataset('./data/tfrecords/tusimple_train.tfrecords')
#
#     # Combine it with keras
#     model_input = tf.keras.layers.Input(tensor=image)
#
#     train_model = create_model((80,160), (2,2))
#
#     # Compile your model
#     train_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
#                         loss='mean_squared_error',
#                         metrics=['accuracy'],
#                         target_tensors=[label])
#
#     # Train the model
#     train_model.fit(image, epochs=10,
#                     steps_per_epoch=2)


