import csv
import cv2
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers import Conv2D
from keras.callbacks import ModelCheckpoint

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 32, 'batch size.')
flags.DEFINE_integer('num_epochs', 6, 'The number of epochs to train for.')
flags.DEFINE_float('lrate', 1e-3, 'The learning rate for training.')
flags.DEFINE_float('lratedecay', 1e-7, 'Learning rate decay over each epoch')


def image_preprocessing(image):
    # Switch from BGR to RGB to fit with drive.py function
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # # Reduce image size by 2  (320x160) -> (160x80)
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    # Apply a slight Gaussian blur
    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


def data_augmentation(image, angle):
    # Hyperparameters
    translation_hor_max = 10  # pixels
    translation_vert_max = 1  # pixels
    brightness_max = 0.1

    # Generate random transformation values
    shift_x = np.random.uniform(-translation_hor_max, translation_hor_max)
    shift_y = np.random.uniform(-translation_vert_max, translation_vert_max)
    bright = np.random.uniform(-brightness_max, brightness_max)

    images = image_augmentation(image, shift_x, shift_y, bright)
    angles = angle_augmentation(angle, shift_x)

    return images, angles


def image_augmentation(image, shift_x, shift_y, bright):

    image_size = image.shape
    images = []

    # TODO: Random shadow?

    # Brightness radomization..
    white_img = 255 * np.ones(image_size, np.uint8)
    black_img = np.zeros(image_size, np.uint8)

    if bright >= 0:
        aug_image = cv2.addWeighted(image, 1 - bright, white_img, bright, 0)
    else:
        aug_image = cv2.addWeighted(image, bright + 1, black_img, bright * -1, 0)

    h, w, ch = aug_image.shape

    # Randomly shift horizon to take the hilly conditions into account
    # horizon position (from the top)
    horizon = h * 0.375

    # road edges at the horizon..
    edge1 = w * 0.3
    edge2 = w * 0.7

    # origin points
    pts1 = np.float32([[edge1, horizon], [edge2, horizon], [0, h], [w, h]])
    # destination points
    pts2 = np.float32([[edge1, horizon + shift_y], [edge2, horizon + shift_y], [shift_x, h], [w+shift_x, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    aug_image = cv2.warpPerspective(aug_image, M, (w, h))  # borderMode=cv2.BORDER_REPLICATE)

    # Flip images
    flip_image = cv2.flip(image, 1)
    flip_aug_image = cv2.flip(aug_image, 1)

    # Append original, flipped and augmented image.
    images.append(image)
    images.append(flip_image)
    images.append(aug_image)
    images.append(flip_aug_image)

    return images


def angle_augmentation(angle, shift_x):
    angles = []

    # Augment steering angle after perspective transformation
    aug_angle = angle + shift_x * 0.00415

    # Flip angles
    flip_angle = angle * -1.0
    flip_aug_angle = aug_angle * -1.0

    # Append original, flipped and augmented image.
    angles.append(angle)
    angles.append(flip_angle)
    angles.append(aug_angle)
    angles.append(flip_aug_angle)

    return angles


def filter_data(lines, speeds, angles, min_speed):
    filtered_lines = []
    filtered_speeds = []
    filtered_angles = []

    for line, speed, angle in zip(lines, speeds, angles):
        if speed > min_speed:
            filtered_lines.append(line)
            filtered_speeds.append(speed)
            filtered_angles.append(angle)

    return filtered_lines, filtered_speeds, filtered_angles


def fetch_data_csv(line, index=0):
    source_path = line[index]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    speed = float(line[6])

    # Steering data augmentation depending on which camera the image is from.
    angle = float(line[3])
    if index == 1:
        angle += 0.25
    elif index == 2:
        angle -= 0.25

    return image, angle, speed


# Show data set sample
def show_dataset_sample(lines, sample_num=1, sample_index=-1, augmented=0, display=1, name="figure0.png"):

    col_num = 3  # Center, Left, Right

    if augmented == 1:
        row_num = int(sample_num * 12 / col_num)
    else:
        row_num = int(sample_num * 3 / col_num)

    # Create array subplot
    f, axarr = plt.subplots(row_num, col_num, figsize=(col_num * 4, row_num * 2.5))

    index = sample_index - 1

    for i in range(0, sample_num, 1):
        # Either pick a random sample or sequentialy incrementaly
        if sample_index < -1:
            index = random.randint(0, len(lines))
        else:
            index = index + 1

        imgs, angs = [], []

        # Fetch images from CSV lines..
        for j in range(3):
            image, angle, speed = fetch_data_csv(lines[index], index=j)

            # Preprocess image
            image = image_preprocessing(image)

            if augmented == 1:
                # Generate a list of augmented images
                temp_imgs, temp_angs = data_augmentation(image, angle)

                for temp_img, temp_ang in zip(temp_imgs, temp_angs):
                    imgs.append(temp_img)
                    angs.append(temp_ang)
            else:
                imgs.append(image)
                angs.append(angle)

        # Reorder image list..
        if augmented == 1:
            # myorder = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            #            ----------  ----------  -----------
            #            o  f  a af  o  f  a af  o  f   a  af
            #              center       left        right
            myorder = [4, 0, 8, 5, 1, 9, 6, 2, 10, 7, 3, 11]
            ordered_imgs = [imgs[k] for k in myorder]
            ordered_angs = [angs[k] for k in myorder]
        else:
            myorder = [1, 0, 2]
            ordered_imgs = [imgs[k] for k in myorder]
            ordered_angs = [angs[k] for k in myorder]

        # Convert lists to numpy arrays
        images = np.array(ordered_imgs, dtype=np.uint8)
        angles = np.array(ordered_angs)*25  # scale angle in degree for visualisation

        # Generate image figure
        if sample_num > 1:
            for k in range(3):
                # Customize title..
                if k % 3 == 0:
                    str1 = "Left"
                elif k % 3 == 1:
                    str1 = "Center"
                elif k % 3 == 2:
                    str1 = "Right"
                # show left image (column 0 in figure, column 1 in CSV file)
                axarr[i, k].set_title('%s camera\nAngle: %.3f\u00b0' % (str1, angles[k]))
                axarr[i, k].imshow(images[k])
                axarr[i, k].axis('off')
                axarr[i, k].axis('off')
                axarr[i, k].axis('off')
        elif len(images) > 3:
            for k in range(len(images)):
                # Customize title..
                if k % 3 == 0:
                    str1 = "Left"
                elif k % 3 == 1:
                    str1 = "Center"
                elif k % 3 == 2:
                    str1 = "Right"

                if int(k/3) == 0:
                    str2 = "Original"
                elif int(k / 3) == 1:
                    str2 = "Flipped"
                elif int(k / 3) == 2:
                    str2 = "Augmented"
                elif int(k / 3) == 3:
                    str2 = "Flipped + Augmented"
                axarr[int(k/3), k % 3].set_title('%s camera - %s\nAngle: %.3f\u00b0' % (str1, str2, angles[k]))
                axarr[int(k/3), k % 3].imshow(images[k])
                axarr[int(k/3), k % 3].axis('off')
                axarr[int(k/3), k % 3].axis('off')
                axarr[int(k/3), k % 3].axis('off')
        elif sample_num == 1:
            for k in range(3):
                # Customize title..
                if k % 3 == 0:
                    str1 = "Left"
                elif k % 3 == 1:
                    str1 = "Center"
                elif k % 3 == 2:
                    str1 = "Right"

                # show image
                axarr[k].set_title('%s camera\nAngle: %.3f\u00b0' % (str1, angles[k]))
                axarr[k].imshow(images[k])
                axarr[k].axis('off')
                axarr[k].axis('off')
                axarr[k].axis('off')

    f.tight_layout()

    if display == 1:
        plt.show()
    else:
        plt.savefig("./examples/generated_figures/" + name)


def plot(data, x_label, y_label, title, display=1, name="figure0.png"):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(data)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    if display == 1:
        plt.show()
    else:
        plt.savefig("./examples/generated_figures/" + name)


def plot_histogram(data, x_label, y_label, title, display=1, name="figure0.png"):
    fig, ax = plt.subplots(figsize=(15, 5))

    n_bins = 25
    # Generate histogram and bins
    hist, bins = np.histogram(data, n_bins)
    # Bar width
    width = 0.8 * (bins[1] - bins[0])
    # left side of the bars : center - width/2
    left = (bins[:-1] + bins[1:])/2 - width/2
    ax.bar(left, hist, width=width)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    if display == 1:
        plt.show()
    else:
        plt.savefig("./examples/generated_figures/" + name)


def load_data():
    display_figures = 0
    min_speed = 10
    validation_split = 0.3

    lines = []
    ang = []
    sp = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

            # Steering angles are scaled for visualisation
            ang.append(25. * float(line[3]))
            sp.append(float(line[6]))
            # Numpy array for dataset visualisation
            speeds = np.array(sp)
            angles = np.array(ang)

    # Plot Steering and Speed values
    plot(angles, 'sample', 'Steering angle (\u00b0)', 'Non-augmented/filtered steering values vs sample',
         display=display_figures, name="figure1.png")
    plot(speeds, 'sample', 'Speed value (mph)', 'Non-augmented/filtered Speed values vs sample',
         display=display_figures, name="figure2.png")

    # Plot histogram steering and speed values before filtering
    plot_histogram(angles, 'Steering angle (\u00b0)', 'Number of occurence',
                   'Non-augmented/filtered histogram of the steering angles', display=display_figures,
                   name="figure3.png")
    plot_histogram(speeds, 'Speed (mph)', 'Number of occurence',
                   'Non-augmented/filtered histogram of the speeds', display=display_figures, name="figure4.png")

    # Don't consider lines with speeds below : min speed)
    filtered_lines, filtered_speeds, filtered_angles = filter_data(lines, speeds, angles, min_speed)

    # Plot Steering and Speed values after filtering
    plot(filtered_angles, 'sample', 'Steering value (\u00b0)', 'Filtered steering values vs sample',
         display=display_figures, name="figure5.png")
    plot(filtered_speeds, 'sample', 'Speed value (mph)', 'Filtered Speed values vs sample',
         display=display_figures, name="figure6.png")

    # Plot histogram steering and speed values before filtering
    plot_histogram(filtered_angles, 'Steering angle (\u00b0)', 'Number of occurence',
                   'Filtered histogram of the steering angles', display=display_figures, name="figure7.png")
    plot_histogram(filtered_speeds, 'Speed (mph)', 'Number of occurence',
                   'Filtered histogram of the speeds', display=display_figures, name="figure8.png")

    # show a random sample of the remaining dataset
    show_dataset_sample(filtered_lines, sample_num=1, sample_index=-1, augmented=0, display=display_figures, name="figure9.png")

    # show a random sample of the preprocessed\augmented dataset
    show_dataset_sample(filtered_lines, sample_num=1, sample_index=100, augmented=1, display=display_figures, name="figure10.png")

    # Show an example of data distribution after dataset augmentation
    # Hyperparameters
    angs = []

    translation_hor_max = 10  # pixels
    for angle in filtered_angles:
        shift_x = np.random.uniform(-translation_hor_max, translation_hor_max)
        temp_angs = angle_augmentation(angle, shift_x)

        for ang in temp_angs:
            angs.append(ang)

    # Plot histogram steering and speed values before filtering
    plot_histogram(angs, 'Steering angle (\u00b0)', 'Number of occurence',
                   'Histogram of filtered/augmented/flipped steering angles', display=display_figures,
                   name="figure11.png")

    # Randomly split the csv file lines in a validation set and a training set
    train_samples, validation_samples = train_test_split(filtered_lines, test_size=validation_split)

    n_train = len(train_samples)
    n_valid = len(validation_samples)

    # Print summary of the dataset
    print("Number of training examples after filtering/augmentation/flipping =", n_train * 4)
    print("Number of validation examples after filtering/augmentation/flipping =", n_valid * 4)

    return train_samples, validation_samples


def get_model():
    model = Sequential()

    # Normalize
    model.add(Lambda(lambda x: x/127.5 - 1, input_shape=(80, 160, 3)))

    # Crop image to keep only relevant features
    model.add(Cropping2D(cropping=((27, 10), (0, 0))))

    # Model based on NVIDIA architecture (Section 4 - https://arxiv.org/pdf/1604.07316.pdf)
    # 5x5 convolution layers (2x2 stride)
    model.add(Conv2D(24, (3, 3), strides=(2, 2), activation="elu"))
    model.add(Conv2D(36, (3, 3), strides=(2, 2), activation="elu"))
    model.add(Conv2D(48, (3, 3), strides=(2, 2), activation="elu"))
    # 3x3 convolution layers
    model.add(Conv2D(64, (2, 2), activation="elu"))
    model.add(Conv2D(64, (2, 2), activation="elu"))
    # Flatten
    model.add(Flatten())
    # # Dropout
    model.add(Dropout(.2))
    # Fully connected output = 100
    model.add(Dense(100, activation='elu'))
    # # Dropout
    model.add(Dropout(.2))
    # Fully connected output = 50
    model.add(Dense(50, activation='elu'))
    # # Dropout
    model.add(Dropout(.2))
    # Fully connected output = 10
    model.add(Dense(10, activation='elu'))
    # Output layer
    model.add(Dense(1))

    # Print out model summary
    model.summary()

    return model


def generator(samples, batch_size=32):
    num_samples = len(samples)
    # Loop forever so the generator never terminates
    while 1:
        # shuffles the input sample
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            # CSV file line subset
            batch_samples = samples[offset:offset+batch_size]

            imgs = []
            angs = []

            # for each CSV file line..
            for batch_sample in batch_samples:
                # For each image (center, left, right)
                for i in range(3):
                    image, angle, speed = fetch_data_csv(batch_sample, index=i)

                    # Data preprocessing
                    image = image_preprocessing(image)

                    # Data augmentation
                    temp_imgs, temp_angs = data_augmentation(image, angle)

                    for temp_img, temp_ang in zip(temp_imgs, temp_angs):
                        imgs.append(temp_img)
                        angs.append(temp_ang)

            # Convert images to numpy arrays
            X_train = np.array(imgs, dtype=np.uint8)
            y_train = np.array(angs)

            # yield
            yield shuffle(X_train, y_train)


def main(_):
    # Load images path + steering angle data
    train_samples, validation_samples = load_data()

    # Compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=FLAGS.batch_size)
    validation_generator = generator(validation_samples, batch_size=FLAGS.batch_size)

    from keras.optimizers import Adam

    # Get model from functon
    model = get_model()

    # # load weights
    # model.load_weights('./tmp/best-weights.hdf5')
    # model.save('model_best.h5')
    # print("Model saved!")

    # Compile and train the model
    model.compile(optimizer=Adam(lr=FLAGS.lrate, decay=FLAGS.lratedecay), loss='mse')

    # saves the model weights after each epoch if the validation loss decreased
    filepath = './tmp/best-weights.hdf5'
    checkpointer = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True,  mode='min')

    history_object = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_samples),
        epochs=FLAGS.num_epochs,
        verbose=1,
        callbacks=[checkpointer],
        validation_data=validation_generator,
        validation_steps=len(validation_samples))

    # Save the model
    model.save('model.h5')
    print("Model saved!")

    # Plot the training and validation loss for each epoch
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history_object.history['loss'])
    ax.plot(history_object.history['val_loss'])
    ax.set_ylabel('Mean squared error loss')
    ax.set_xlabel('Epoch')
    ax.set_title('Model mean squared error loss vs time')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.tight_layout()
    plt.savefig('./examples/generated_figures/model_loss_vs_epoch.png')
    plt.show()

if __name__ == '__main__':
    tf.app.run()
