# coding: utf-8
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Data Preprocessing
data = np.loadtxt(open('train.csv'), delimiter=',', skiprows=1)
train, valid = train_test_split(data, test_size=0.2)
train_x, train_y = train[:,1:], train[:, 0]
valid_x, valid_y = train[:,1:], train[:, 0]

train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)/255.0
valid_x = valid_x.reshape(valid_x.shape[0], 28, 28, 1)/255.0

train_y = np_utils.to_categorical(train_y)
valid_y = np_utils.to_categorical(valid_y)
print(train_x.shape, train_y.shape)


num_train = train_x.shape[0]
num_valid = valid_x.shape[0]
num_classes = len(train_y[0])
EPOCHS = 10
BATCH_SIZE = 64

# The model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='same',activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.5))
 
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Image generator for augement image
image_gen = ImageDataGenerator(zoom_range = 0.15, height_shift_range = 0.15, width_shift_range = 0.15,rotation_range = 30)
                     
model.fit_generator(image_gen.flow(train_x, train_y, batch_size=BATCH_SIZE), steps_per_epoch=num_train/BATCH_SIZE,
                   epochs=EPOCHS, validation_data=(valid_x, valid_y))

model.save('mnist.h5')

test = np.loadtxt(open('test.csv'), delimiter=',', skiprows=1)
test = test.reshape(test.shape[0], 28, 28, 1)/255.0

res = model.predict(test, batch_size=BATCH_SIZE)
res = np.argmax(res, axis=1)

output = open('submission.csv', 'w')
output.write('ImageId,Label\n')
for i in range(0, len(res)):
    output.write("".join([str(i+1),',',str(res[i]),'\n']))

output.close()
