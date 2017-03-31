import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# generator declaration
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
                #Shuffle input samples
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # Crop image to keep only relevant features (road section)

            # Convert images to numpy arrays
            X_train = np.array(images)
            y_train = np.array(angles)

            # yeild
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format






for col in lines:
    #3 cameras
    for i in range(3):
        source_path = col[i]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(col[3])
        if i == 1:
            measurement += 0.2
        elif i == 2:
            measurement -= 0.2
        measurements.append(measurement)


# Data augmentation
# Flip image
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print("Data augmentation completed!")





#Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers import Conv2D
import keras.models import Model

model = Sequential()

# Normalize
model.add(Lambda(lambda x: x/127.5 - 1, input_shape=(row, col, ch)))
# # Crop image to keep only relevant features
# model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# Model based on NVIDIA architecture
# 5x5 convolution layers (2x2 stride)
model.add(Conv2D(24, (5, 5), subsample=(2, 2), activation="elu"))
model.add(Conv2D(36, (5, 5), subsample=(2, 2), activation="elu"))
model.add(Conv2D(48, (5, 5), subsample=(2, 2), activation="elu"))
# 3x3 convolution layers (2x2 stride)
model.add(Conv2D(64, (3, 3), activation="elu"))
model.add(Conv2D(64, (3, 3), activation="elu"))
# Flatten
model.add(Flatten())
# # Dropout
# model.add(Dropout(.2))
# Fully connected output = 100
model.add(Dense(100, activation='elu'))
# # Dropout
# model.add(Dropout(.5))
# Fully connected output = 50
model.add(Dense(50, activation='elu'))
# # Dropout
# model.add(Dropout(.5))
# voir W-regularizer=12(0.001)
# Fully connected output = 10
model.add(Dense(10, activation='elu'))
# Output layer
model.add(Dense(1))

# Compile and train the model
learning_rate =1e-4
model.compile(loss='mse', optimizer='adam',)
history_object = model.fit_generator(train_generator,
    samples_per_epoch=len(train_samples), validation_data=validation_generator,
    nb_val_samples=len(validation_samples), epochs=3, verbose=1,
    validation_split=0.2, shuffle=True)

# Save the model
model.save('model.h5')
print("Model saved!")

# Print the keys contained in the history object
print(history_object.history.keys())

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()