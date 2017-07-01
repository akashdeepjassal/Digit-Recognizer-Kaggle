'''
  Keras CNN based losely on a slightly shallower version of the vgg16 CNN.
  Code uses various snippets from other projects on github and kaggle kernels
  for which I've lost notes of the original authors. Apologies in advance!
  This version with 30 epochs scored 99.4% on the Kaggle competition and took about 30 minutes
  to run on an AWS EC2 p2.xlarge instance
'''

import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.utils import np_utils
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.image as mpimg

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.datasets import mnist

epochs = 25

train_size = 42000
test_size = 2000

batch_size = 128
num_classes = 10

img_rows = 28
img_cols = 28

# Chose to use the Kaggle competition set or the large MNIST Keras set
data_source = "kaggle" # "keras"

if (data_source == "kaggle"):
    '''
    Split our test / train data for local validation and testing. For competition
    (once parameters are tuned) it makes sense to use as much data for training
    as possible so at this stage change the second line to:
      (train , test) = (data[:42000], data[38000:42000])
    '''
    data = pd.read_csv("input/train.csv", nrows=42000)
    (train , test) = (data[:train_size], data[42000 - test_size:42000])

    # The Kaggle data has the Y values (the actual number represented) as a row in the
    # train data set. We need to split that into a separate input here.
    x_train = (train.ix[:,1:].values).astype('float32')
    y_train = train.ix[:,0].values.astype('int32')

    x_test = (test.ix[:,1:].values).astype('float32')
    y_test = (test.ix[:,0].values).astype('int32')

    # We might use the unshaped, non-normalize x_test data later for visualisation so keep it here
    x_test_raw = x_test

    # Reshape the data. NB. using the Theano backend or possibly an older version of Keras
    # reorders the shape here. We could test for that but I've not checked that the alternative
    # ordering works so I've removed it here. If you do get an issue with the x_train shape when
    # training then change the ordering below to 'img_rows, img_cols, 1'
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)

else:
    '''
    Use the larger MNIST data set included with the keras distro
    THIS SHOULDN'T BE USED TO TRAIN FOR COMPETITION
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    input_shape = (1, img_rows, img_cols)

    x_train = x_train.reshape(60000,1,28,28)
    x_test  = x_test.reshape(10000,1,28,28)

# Save the y_test for identifying failures later (before we one-hot encode)
y_test_nv = y_test

# Normalize the data
x_train /= 255
x_test /=255

# We're going to run a classification model and for that the Y side of our training data needs
# to be a one-hot version of our number. Eg:
#
#   y = 7
#
# One-hot encoding:
#
#   y = 0000001000
#
# Numpy has a function to do this for us.
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

'''
  Model is losely based on the vgg16 model. There's a Keras version included in the distro here:
  https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py
  I removed the last Convolutional layer (laziness because I couldn't get it to work) and added
  some dropout between the dense layers (although I've not tested that that brings benefit)
'''
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

'''
  Data Augmentation
  -----------------
  Basic affine transformations of the input data to increase the training set.
  This actually lowered the performance so I've commented it out. It'd defintiely
  be worth playing with the parameters and trying again.
'''
#gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
#                         height_shift_range=0.08, zoom_range=0.08)
#
#test_gen = ImageDataGenerator()
#
#train_generator = gen.flow(x_train, y_train, batch_size=64)
#test_generator = test_gen.flow(x_test, y_test, batch_size=64)
#
#history = model.fit_generator(train_generator, steps_per_epoch=100000//64, epochs=epochs,
#                    validation_data=test_generator, validation_steps=10000//64)

# Train the model here.
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test,y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("Baseline Error: %.2f%%" % (100-score[1]*100))

# Run the model to predict for our test set
pred = model.predict_classes(x_test,verbose=0)


def plot_failures(pred, x_test, y_test_nv, file):
  '''
  Plot Failures creates a 10 x 10 plot of the first 100 failed idenfications attempts.

  I've used Pandas because I have some familiarity with it but I'd imagine there was a more
  elegant solution!
  Credit to mnielsen for some of this:
    https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/fig/mnist.py

  '''

  # Stack the predictions alongside the actual values
  cmp = np.column_stack((pred, y_test_nv))

  # Create a panda from the arrays of actual vs predicted
  df = pd.DataFrame(cmp)

  # Add a column that explicitly identifies matches versus failures
  df['match'] = np.where(df[0]==df[1], 1, 0)

  # Retrive indexes of failed matches
  fail = df[df['match'] == 0].index.tolist()

  # Create array of just the failed images
  x_fail = x_test[np.array(fail)]

  if (len(x_fail) > 100):
    x_fail = x_fail[:100]

  # Reshape the failure images for plotting and pad to 100 examples
  num_images, img = x_fail.shape
  x_fail = x_fail.reshape(num_images,28,28)
  x_fail = np.concatenate((x_fail, np.zeros((100-num_images,28,28))), axis=0)

  """
  Plot and save the images
  """
  fig = plt.figure()
  for x in range(1,10):
      for y in range(1,10):
          ax = fig.add_subplot(10, 10, 10*y+x)
          ax.matshow(x_fail[10*y+x], cmap = matplotlib.cm.binary)
          plt.xticks(np.array([]))
          plt.yticks(np.array([]))

  plt.savefig(file + '.png')

# Plot the first 100 failures.
#plot_failures(pred, x_test_raw, y_test_nv, "failures")

'''
Read the Kaggle test data, predict and build a submission CSV for the competition
'''
comp = pd.read_csv("input/test.csv")
x_comp = comp.iloc[:,:].values
x_comp = x_comp.reshape(x_comp.shape[0], 1, img_rows, img_cols)

pred = model.predict_classes(x_comp,verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(pred)+1)), "Label" : pred})
submissions.to_csv("DR.csv", index=False, header=True)
