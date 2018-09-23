import csv
import cv2
from scipy import ndimage
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, BatchNormalization, Cropping2D, Dropout

def img_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = 0.25 + np.random.uniform()
    hsv[:,:,2] = hsv[:,:,2] * brightness
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def readImages(dataFolder):
    lines = []
    with open(dataFolder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
        images = []
        measurements = []
        lines.pop(0)
        for line in lines:
            for i in range(3):
                source_path = line[i]
                file_name = source_path.split('/')[-1]
                current_path = dataFolder + '/IMG/' + file_name
                image = ndimage.imread(current_path)
                images.append(image)
                measurement = float(line[3])
                if i == 1:
                    measurement = measurement + 0.20
                if i == 2:
                    measurement = measurement - 0.20
                
                measurements.append(measurement)
        
        return images, measurements

def data_augmentation(images, measurements):
    augmented_images, augmented_measurements = [],[]
    for image, measurement in zip(images, measurements):
        if np.random.randint(5) == 1:
                img_new = img_brightness(image)
                images.append(img_new)
                measurements.append(measurement)
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        if np.random.randint(3) == 1:
            augmented_images.append(cv2.flip(image, 1))
            augmented_measurements.append(measurement*-1.0)
        elif np.random.randint(3) == 1:
                img_new = img_brightness(cv2.flip(image, 1))
                images.append(img_new)
                measurements.append(measurement*-1.0)
    
    return augmented_images, augmented_measurements
    
def salih_Lenet(X_train, y_train):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))    
    model.add(Cropping2D(cropping=((70,25),(0,0))))

    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Activation('relu'))

    model.add(Dense(512))
    model.add(Dropout(.2))
    model.add(Activation('relu'))

    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Dropout(.5))
    model.add(Activation('relu'))

    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)
    
    model.save('model.h5')

print('Read images...')
images, measurements = readImages('./model_data')

print('Data augmentation..')
augmented_images, augmented_measurements = data_augmentation(images, measurements)

print('Reverse drive images..')
reverse_images, reverse_measurements = readImages('./model_data_reverse')

print('Add both image data..')
augmented_images = augmented_images + reverse_images
augmented_measurements = augmented_measurements + reverse_measurements

print('Create numpy arrays.. ' + str(len(augmented_images)))
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print('Shuffle..')
X_train, y_train = shuffle(X_train, y_train)

print('Training started..')
salih_Lenet(X_train, y_train)