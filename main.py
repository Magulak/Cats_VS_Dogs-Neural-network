# https://www.kaggle.com/c/dogs-vs-cats

# def get_available_devices():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos]
#
#
# print(get_available_devices())
# print(tensorflow.test.is_built_with_cuda())
# my output was => ['/device:CPU:0']
# good output must be => ['/device:CPU:0', '/device:GPU:0']


import tensorflow
import time

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
global_var = 1


def create_and_train_model(epo, f_layer, s_layer, t_layer, d_layers, opti, act1, act2, act3, act4):
    global global_var
    model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Conv2D(f_layer, (3, 3), activation=act1, input_shape=(256, 256, 3)),
        tensorflow.keras.layers.MaxPooling2D((2, 2)),
        tensorflow.keras.layers.Conv2D(s_layer, (3, 3), activation=act2),
        tensorflow.keras.layers.MaxPooling2D((2, 2)),
        tensorflow.keras.layers.Conv2D(t_layer, (3, 3), activation=act3),
        tensorflow.keras.layers.MaxPooling2D((2, 2)),  # Myślę, że powinna się zmniejszać ilość neuronów a nie zwiększać
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(d_layers, activation=act4),
        tensorflow.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=opti, loss='binary_crossentropy', metrics=['accuracy'])
    batch_size = 128
    with tensorflow.device('/GPU:0'):
        model.fit(train_generator, epochs=epo, steps_per_epoch=len(train_generator))

    val_loss, val_acc = model.evaluate(validation_generator)
    val_loss = round(val_loss, 3)
    val_acc = round(val_acc, 3)

    model.save(
        f'CvD ver {global_var} e{epo} f{f_layer} s{s_layer} t{t_layer} d{d_layers} o{opti} l{val_loss} acc{val_acc} a1{act1} a2{act2} a3{act3} a4{act4}')  # TODO add more model info to name
    time.sleep(600)
    global_var = global_var + 1


def train(e, f, s, t, d, act1, act2, act3, act4):
    create_and_train_model(e, f, s, t, d, 'SGD', act1, act2, act3, act4)
    create_and_train_model(e, f, s, t, d, 'RMSprop', act1, act2, act3, act4)
    create_and_train_model(e, f, s, t, d, 'Adam', act1, act2, act3, act4)


train(7, 32, 64, 128, 128, 'ReLU', 'ReLU', 'ReLU', 'ReLU')
train(8, 32, 64, 128, 128, 'ELU', 'LeakyReLU', 'ReLU', 'ReLU')
train(8, 32, 64, 128, 128, 'ReLU', 'ReLU', 'ELU', 'LeakyReLU')

train(8, 32, 32, 32, 32, 'ReLU', 'ReLU', 'ReLU', 'ReLU')
train(8, 32, 32, 32, 32, 'ELU', 'LeakyReLU', 'ReLU', 'ReLU')
train(8, 32, 32, 32, 32, 'ReLU', 'ReLU', 'ELU', 'LeakyReLU')

train(7, 32, 64, 128, 512, 'ReLU', 'ReLU', 'ReLU', 'ReLU')
train(8, 32, 64, 128, 512, 'ELU', 'LeakyReLU', 'ReLU', 'ReLU')
train(8, 32, 64, 128, 512, 'ReLU', 'ReLU', 'ELU', 'LeakyReLU')

train(7, 64, 128, 512, 1024, 'ReLU', 'ReLU', 'ReLU', 'ReLU')
train(8, 64, 128, 512, 1024, 'ELU', 'LeakyReLU', 'ReLU', 'ReLU')
train(8, 64, 128, 512, 1024, 'ReLU', 'ReLU', 'ELU', 'LeakyReLU')

train(7, 512, 128, 64, 32, 'ReLU', 'ReLU', 'ReLU', 'ReLU')
train(8, 512, 128, 64, 32, 'ELU', 'LeakyReLU', 'ReLU', 'ReLU')
train(8, 512, 128, 64, 32, 'ReLU', 'ReLU', 'ELU', 'LeakyReLU')
# jak model sie overfituje to można pozmieniać batch size




# predictions = model.predict([x_test])
