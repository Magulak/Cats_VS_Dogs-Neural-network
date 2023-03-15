# https://www.kaggle.com/c/dogs-vs-cats
import tensorflow

# def get_available_devices():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos]
#
#
# print(get_available_devices())
# print(tensorflow.test.is_built_with_cuda())
# my output was => ['/device:CPU:0']
# good output must be => ['/device:CPU:0', '/device:GPU:0']



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

model = tensorflow.keras.Sequential([
    tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tensorflow.keras.layers.MaxPooling2D((2, 2)),
    tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tensorflow.keras.layers.MaxPooling2D((2, 2)),
    tensorflow.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tensorflow.keras.layers.MaxPooling2D((2, 2)),  # Myślę, że powinna się zmniejszać ilość neuronów a nie zwiększać
    tensorflow.keras.layers.Flatten(),
    tensorflow.keras.layers.Dense(128, activation='relu'),
    tensorflow.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
batch_size = 128
with tensorflow.device('/GPU:0'):
    model.fit(train_generator, epochs=3, steps_per_epoch=len(train_generator))

val_loss, val_acc = model.evaluate(validation_generator)
model.save('CvD ver 1') # TODO add more model info to name
# TODO ADD SLEEP FOR 30 min after training
print(val_loss, val_acc)
# predictions = model.predict([x_test])
