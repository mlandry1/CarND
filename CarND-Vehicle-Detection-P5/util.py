import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import numpy as np
import cv2
from sklearn.utils import shuffle


def image_preprocessing(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_chan = hls[:, :, 1]

    # create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(l_chan)

    hls[:, :, 1] = enhanced

    image = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    return image


def show_sample(features, labels, preprocess=0, sample_num=1, sample_index=-1, display=1):
    col_num = 2
    # Create training sample + histogram plot
    f, axarr = plt.subplots(sample_num, col_num, figsize=(col_num * 4, sample_num * 3))

    index = sample_index - 1
    for i in range(0, sample_num, 1):

        if sample_index == -1:
            index = random.randint(0, len(features))
        else:
            index = index + 1

        if labels[index] == 1:
            label_str = "Car"
        else:
            label_str = "Non-Car"

        image = (mpimg.imread(features[index]) * 255).astype(np.uint8)

        if preprocess == 1:
            image = image_preprocessing(image)

        axarr[i, 0].set_title('%s' % label_str)
        axarr[i, 0].imshow(image)

        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max()/ cdf.max()

        axarr[i, 1].plot(cdf_normalized, color='b')
        axarr[i, 1].plot(hist, color='r')
        axarr[i, 1].legend(('cdf', 'histogram'), loc='upper left')

        axarr[i, 0].axis('off')

    # Tweak spacing to prevent clipping of title labels
    f.tight_layout()
    if display == 1:
        plt.show()
    else:
        if preprocess == 1:
            plt.savefig("./output_images/dataset_sample_preprocessed.png")
        else:
            plt.savefig("./output_images/dataset_sample.png")


def show_training_dataset_histogram(labels_train, labels_valid, display=1):
    fig, ax = plt.subplots(figsize=(5, 5))

    n_classes = np.unique(labels_train).size

    # Generate histogram and bins
    hist_train, bins = np.histogram(labels_train, 2)
    hist_valid, bins = np.histogram(labels_valid, 2)

    # Bar width
    width = 1.0 * (bins[1] - bins[0])

    ax.bar([-1, 1], hist_train, width=width, label="Train")
    ax.bar([-1, 1], hist_valid, width=width, label="Valid")

    ax.set_xlabel('Classes')
    ax.set_ylabel('Number of occurence')
    ax.set_title('Histogram of the data set')

    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()

    if display == 1:
        plt.show()
    else:
        plt.savefig("./output_images/histogram_dataset.png")


def keras_generator(features, labels, batch_size=32):
    num_features = len(features)
    # Loop forever so the generator never terminates
    while 1:
        # shuffles the input sample
        shuffle(features, labels)
        for offset in range(0, num_features, batch_size):
            # File path subset
            batch_features = features[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]
            imgs = []

            for feature in batch_features:
                image = (mpimg.imread(feature) * 255).astype(np.uint8)

                # Image preprocessing
                image = image_preprocessing(image)
                imgs.append(image)

            # Convert images to numpy arrays
            X = np.array(imgs, dtype=np.uint8)
            y = np.array(batch_labels)

            yield shuffle(X, y)


def generator(features, labels):
    for iterable in keras_generator(features, labels, batch_size=len(features)):
        return iterable


def plot_train_results(history_object, display):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history_object.history['acc'])
    ax.plot(history_object.history['val_acc'])
    ax.set_ylabel('Model accuracy')
    ax.set_xlabel('Epoch')
    ax.set_title('Model accuracy vs epochs')
    plt.legend(['training accuracy', 'validation accuracy'], bbox_to_anchor=(1.01, 1.0))

    fig.tight_layout()
    if display == 1:
        plt.show()
    else:
        plt.savefig("./output_images/accuracy_over_epochs.png")