# https://www.kaggle.com/c/dogs-vs-cats
import tensorflow
import scipy
import numpy
import os

# TODO you can use Keras to convert img to array
from PIL import Image

dir_path = r'G:/Scripts/Cats_VS_Dogs/kagglecatsanddogs_5340/PetImages'

train_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,  # rescale pixel values between 0 and 1
    shear_range=0.2,  # apply random shear transformations
    zoom_range=0.2,  # apply random zoom transformations
    horizontal_flip=True,  # flip images horizontally
    validation_split=0.2  # add validation split
)
train_generator = train_datagen.flow_from_directory(
    directory=dir_path,
    target_size=(256, 256),  # resize images to (256, 256)
    batch_size=32,  # generate batches of 32 images
    class_mode='binary',  # binary classification problem (e.g. cats vs dogs)
    subset='training'  # specify the subset as training
)
validation_generator = train_datagen.flow_from_directory(
    directory=dir_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # specify the subset as validation
)

# Create a model and compile it
model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Flatten())  # 28x28[2D] needs to be converted to [1D]
model.add(
    tensorflow.keras.layers.Dense(252,
                                  activation=tensorflow.nn.relu))  # Use nn.relu as default and then change it to
# other layers
model.add(tensorflow.keras.layers.Dense(64, activation=tensorflow.nn.relu))
model.add(tensorflow.keras.layers.Dense(10, activation=tensorflow.nn.softmax))  # Use softmax for probability

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
batch_size = 64
for e in range(1):
    for i, (x, y) in enumerate(train_generator):
        try:
            model.fit(train_generator, epochs=1,steps_per_epoch=len(train_generator))
        except tensorflow.keras.preprocessing.image.UnidentifiedImageError as e:
            print(f"UnidentifiedImageError in epoch {e + 1}, batch {i + 1}: {e}")
            print(f"Filename: {train_generator.filenames[i * batch_size]}")
            continue

val_loss, val_acc = model.evaluate(validation_generator)
model.save('num_reader')
print(val_loss, val_acc)
# predictions = model.predict([x_test])
