from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#Argument values for the following functions were chosen by experiments
CNN_classifier = Sequential()
CNN_classifier.add(Convolution2D(filters = 32, kernel_size=[3,3], input_shape = (64, 64, 3), activation ='relu'))
CNN_classifier.add(MaxPooling2D(pool_size=(2,2)))
#Adding a second convolutional layer
CNN_classifier.add(Convolution2D(filters = 32, kernel_size=[3,3], activation = 'relu'))
CNN_classifier.add(MaxPooling2D(pool_size = (2, 2)))

CNN_classifier.add(Flatten())
#Hidden layer
CNN_classifier.add(Dense(128, activation = 'relu'))
#Output layer
CNN_classifier.add(Dense(1, activation = 'sigmoid'))

CNN_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Image augmentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

CNN_classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 20,
                         validation_data = test_set,
                         validation_steps = 2000)




