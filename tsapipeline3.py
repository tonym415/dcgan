# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 02:12:21 2017

@author: JonStewart
"""

# import libraries
from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
import os
import re

from tflearn.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
# from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import matplotlib.pyplot as plt
import random
from timeit import default_timer as timer

# ---------------------------------------------------------------------------------------
# Constants
#
# INPUT_FOLDER:                 The folder that contains the source data
#
# PREPROCESSED_DATA_FOLDER:     The folder that contains preprocessed .npy files 
# 
# STAGE1_LABELS:                The CSV file containing the labels by subject
#
# THREAT_ZONE:                  Threat Zone to train on (actual number not 0 based)
#
# BATCH_SIZE:                   Number of Subjects per batch
#
# EXAMPLES_PER_SUBJECT          Number of examples generated per subject
#
# FILE_LIST:                    A list of the preprocessed .npy files to batch
# 
# TRAIN_TEST_SPLIT_RATIO:       Ratio to split the FILE_LIST between train and test
#
# TRAIN_SET_FILE_LIST:          The list of .npy files to be used for training
#
# TEST_SET_FILE_LIST:           The list of .npy files to be used for testing
#
# IMAGE_DIM:                    The height and width of the images in pixels
#
# LEARNING_RATE                 Learning rate for the neural network
#
# N_TRAIN_STEPS                 The number of train steps (epochs) to run
#
# TRAIN_PATH                    Place to stor                           e the tensorboard logs
#
# MODEL_PATH                    Path where model files are stored
#
# MODEL_NAME                    Name of the model files
#
# ----------------------------------------------------------------------------------------
INPUT_FOLDER = 'D:\\stage1_aps'
PREPROCESSED_DATA_FOLDER = 'D:\\kaggle_tsa\\stage_1dproc'
STAGE1_LABELS = 'D:\\kaggle_tsa\\stage1_labels.csv'
THREAT_ZONE = 1
BATCH_SIZE = 16
THREAT_PRESENT = 1
EXAMPLES_PER_SUBJECT = 182

FILE_LIST = []
TRAIN_TEST_SPLIT_RATIO = 0.2
TRAIN_SET_FILE_LIST = []
TEST_SET_FILE_LIST = []

IMAGE_DIM = 224
LEARNING_RATE = 1e-3
N_TRAIN_STEPS = 1
TRAIN_PATH = 'D:\\kaggle_tsa\\tsa_logs\\train'
MODEL_PATH = 'D:\\kaggle_tsa\\tsa_logs\\model'
MODEL_NAME = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('alexnet-v0.1', LEARNING_RATE, IMAGE_DIM,
                                                IMAGE_DIM, THREAT_ZONE))


# ---------------------------------------------------------------------------------------
# preprocess_tsa_data(): preprocesses the tsa datasets
#
# parameters:      none
#
# returns:         none
# ---------------------------------------------------------------------------------------

def preprocess_tsa_data():
    # OPTION 1: get a list of all subjects for which there are labels
    df = pd.read_csv(STAGE1_LABELS)
    df['Subject'], df['Zone'] = df['Id'].str.split('_', 1).str
    SUBJECT_LIST = df['Subject'].unique()

    # OPTION 2: get a list of all subjects for whom there is data
    # SUBJECT_LIST = [os.path.splitext(subject)[0] for subject in os.listdir(INPUT_FOLDER)]

    # OPTION 3: get a list of subjects for small bore test purposes
    # SUBJECT_LIST = ['00360f79fd6e02781457eda48f85da90','0043db5e8c819bffc15261b1f1ac5e42',
    #                '0050492f92e22eed3474ae3a6fc907fa','006ec59fa59dd80a64c85347eef810c7',
    #                '0097503ee9fa0606559c56458b281a08','011516ab0eca7cad7f5257672ddde70e']

    # intialize tracking and saving items
    batch_num = 1
    threat_zone_examples = []
    start_time = timer()

    for subject in SUBJECT_LIST:

        # read in the images
        print('--------------------------------------------------------------')
        print('t+> {:5.3f} |Reading images for subject #: {}'.format(timer() - start_time,
                                                                     subject))
        print('--------------------------------------------------------------')
        images = tsa.read_data(INPUT_FOLDER + '/' + subject + '.aps')

        # transpose so that the slice is the first dimension shape(16, 620, 512)
        images = images.transpose()

        # for each threat zone, loop through each image, mask off the zone and then crop it
        for tz_num, threat_zone_x_crop_dims in enumerate(zip(tsa.zone_slice_list,
                                                             tsa.zone_crop_list)):

            threat_zone = threat_zone_x_crop_dims[0]
            crop_dims = threat_zone_x_crop_dims[1]

            # get label
            label = np.array(tsa.get_subject_zone_label(tz_num,
                                                        tsa.get_subject_labels(STAGE1_LABELS, subject)))
            if label == '1':
                for img_num, img in enumerate(images):

                    print('Threat Zone:Image -> {}:{}'.format(tz_num, img_num))
                    print('Threat Zone Label -> {}'.format(label))

                    if threat_zone[img_num] is not None:

                        # correct the orientation of the image
                        print('-> reorienting base image')
                        base_img = np.flipud(img)
                        print('-> shape {}|mean={}'.format(base_img.shape,
                                                           base_img.mean()))

                        # convert to grayscale
                        print('-> converting to grayscale')
                        rescaled_img = tsa.convert_to_grayscale(base_img)
                        print('-> shape {}|mean={}'.format(rescaled_img.shape,
                                                           rescaled_img.mean()))

                        # spread the spectrum to improve contrast
                        print('-> spreading spectrum')
                        high_contrast_img = tsa.spread_spectrum(rescaled_img)
                        print('-> shape {}|mean={}'.format(high_contrast_img.shape,
                                                           high_contrast_img.mean()))

                        # get the masked image
                        print('-> masking image')
                        masked_img = tsa.roi(high_contrast_img, threat_zone[img_num])
                        print('-> shape {}|mean={}'.format(masked_img.shape,
                                                           masked_img.mean()))

                        # crop the image
                        print('-> cropping image')
                        cropped_img = tsa.crop(masked_img, crop_dims[img_num])
                        print('-> shape {}|mean={}'.format(cropped_img.shape,
                                                           cropped_img.mean()))

                        # normalize the image
                        print('-> normalizing image')
                        normalized_img = tsa.normalize(cropped_img)
                        print('-> shape {}|mean={}'.format(normalized_img.shape,
                                                           normalized_img.mean()))

                        # zero center the image
                        print('-> zero centering')
                        zero_centered_img = tsa.zero_center(normalized_img)
                        print('-> shape {}|mean={}'.format(zero_centered_img.shape,
                                                           zero_centered_img.mean()))

                        # append the features and labels to this threat zone's example array
                        print('-> appending example to threat zone {}'.format(tz_num))
                        threat_zone_examples.append([[tz_num], zero_centered_img, label])
                        print('-> shape {:d}:{:d}:{:d}:{:d}:{:d}:{:d}'.format(
                            len(threat_zone_examples),
                            len(threat_zone_examples[0]),
                            len(threat_zone_examples[0][0]),
                            len(threat_zone_examples[0][1][0]),
                            len(threat_zone_examples[0][1][1]),
                            len(threat_zone_examples[0][2])))
                    else:
                        print('-> No view of tz:{} in img:{}. Skipping to next...'.format(
                            tz_num, img_num))
                    print('------------------------------------------------')

        # each subject gets EXAMPLES_PER_SUBJECT number of examples (182 to be exact, 
        # so this section just writes out the the data once there is a full minibatch 
        # complete.
        if len(threat_zone_examples) % (BATCH_SIZE * EXAMPLES_PER_SUBJECT) == 0:
            for tz_num, tz in enumerate(tsa.zone_slice_list):
                tz_examples_to_save = []

                # write out the batch and reset
                print(' -> writing: ' + PREPROCESSED_DATA_FOLDER +
                      'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(
                          tz_num + 1,
                          len(threat_zone_examples[0][1][0]),
                          len(threat_zone_examples[0][1][1]),
                          batch_num))

                # get this tz's examples
                tz_examples = [example for example in threat_zone_examples if example[0] ==
                               [tz_num]]

                # drop unused columns
                tz_examples_to_save.append([[features_label[1], features_label[2]]
                                            for features_label in tz_examples])

                # save batch.  Note that the trainer looks for tz{} where {} is a 
                # tz_num 1 based in the minibatch file to select which batches to 
                # use for training a given threat zone
                np.save(PREPROCESSED_DATA_FOLDER +
                        'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num + 1,
                                                                           len(threat_zone_examples[0][1][0]),
                                                                           len(threat_zone_examples[0][1][1]),
                                                                           batch_num),
                        tz_examples_to_save)
                del tz_examples_to_save

            # reset for next batch
            del threat_zone_examples
            threat_zone_examples = []
            batch_num += 1

    # we may run out of subjects before we finish a batch, so we write out 
    # the last batch stub
    if len(threat_zone_examples) > 0:
        for tz_num, tz in enumerate(tsa.zone_slice_list):
            tz_examples_to_save = []

            # write out the batch and reset
            print(' -> writing: ' + PREPROCESSED_DATA_FOLDER
                  + 'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num + 1,
                                                                       len(threat_zone_examples[0][1][0]),
                                                                       len(threat_zone_examples[0][1][1]),
                                                                       batch_num))

            # get this tz's examples
            tz_examples = [example for example in threat_zone_examples if example[0] ==
                           [tz_num]]

            # drop unused columns
            tz_examples_to_save.append([[features_label[1], features_label[2]]
                                        for features_label in tz_examples])

            # save batch
            np.save(PREPROCESSED_DATA_FOLDER +
                    'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num + 1,
                                                                       len(threat_zone_examples[0][1][0]),
                                                                       len(threat_zone_examples[0][1][1]),
                                                                       batch_num),
                    tz_examples_to_save)


# unit test ---------------------------------------
# preprocess_tsa_data()

# ---------------------------------------------------------------------------------------
# get_train_test_file_list(): gets the batch file list, splits between train and test
#
# parameters:      none
#
# returns:         none
#
# -------------------------------------------------------------------------------------

def get_train_test_file_list():
    global FILE_LIST
    global TRAIN_SET_FILE_LIST
    global TEST_SET_FILE_LIST

    if os.listdir(PREPROCESSED_DATA_FOLDER) == []:
        print('No preprocessed data available.  Skipping preprocessed data setup..')
    else:
        FILE_LIST = [f for f in os.listdir(PREPROCESSED_DATA_FOLDER)
                     if re.search(re.compile('-tz' + str(THREAT_ZONE) + '-'), f)]
        train_test_split = len(FILE_LIST) - \
                           max(int(len(FILE_LIST) * TRAIN_TEST_SPLIT_RATIO), 1)
        TRAIN_SET_FILE_LIST = FILE_LIST[:train_test_split]
        TEST_SET_FILE_LIST = FILE_LIST[train_test_split:]
        print('Train/Test Split -> {} file(s) of {} used for testing'.format(
            len(FILE_LIST) - train_test_split, len(FILE_LIST)))


# unit test ----------------------------
# get_train_test_file_list()
# print (

# ---------------------------------------------------------------------------------------
# input_pipeline(filename, path): prepares a batch of features and labels for training
#
# parameters:      filename - the file to be batched into the model
#                  path - the folder where filename resides
#
# returns:         feature_batch - a batch of features to train or test on
#                  label_batch - a batch of labels related to the feature_batch
#
# ---------------------------------------------------------------------------------------

def input_pipeline(filename, path):
    preprocessed_tz_scans = []
    feature_batch = []
    label_batch = []
    feature_batch_with_threat = []
    label_batch_with_threat = []
    feature_batch_wo_threat = []
    label_batch_wo_threat = []
    # Load a batch of preprocessed tz scans
    preprocessed_tz_scans = np.load(os.path.join(path, filename))

    # Shuffle to randomize for input into the model
    np.random.shuffle(preprocessed_tz_scans)
    # the below implies that input pipline of preprocessed tz scans already has label type
    # separate features and labels
    for example_list in preprocessed_tz_scans:
        for example in example_list:
            feature_batch.append(example[0])
            label_batch.append(example[1])
            # below is added
            if example[1] == 1:
                feature_batch_with_threat.append(example[0])
                label_batch_with_threat(example[1])
            else:
                feature_batch_wo_threat.append(example[0])
                label_batch_wo_threat.append(example[1])

    feature_batch = np.asarray(feature_batch, dtype=np.float32)
    label_batch = np.asarray(label_batch, dtype=np.float32)
    feature_batch_with_threat = np.asarray(feature_batch_with_threat, dtype=np.float32)
    label_batch_with_threat = np.asarray(label_batch_with_threat, dtype=np.float32)
    feature_batch_wo_threat = np.asarray(feature_batch_wo_threat, dtype=np.float32)
    label_batch_wo_threat = np.asarray(label_batch_wo_threat, dtype=np.float32)

    return feature_batch, label_batch, feature_batch_with_threat, label_batch_with_threat, label_batch_wo_threat, feature_batch_wo_threat


# unit test ------------------------------------------------------------------------
# print ('Train Set -----------------------------')
# for f_in in TRAIN_SET_FILE_LIST:
#    feature_batch, label_batch = input_pipeline(f_in, PREPROCESSED_DATA_FOLDER)
#    print (' -> features shape {}:{}:{}'.format(len(feature_batch), 
#                                                len(feature_batch[0]), 
#                                                len(feature_batch[0][0])))
#    print (' -> labels shape   {}:{}'.format(len(label_batch), len(label_batch[0])))

# print ('Test Set -----------------------------')
# for f_in in TEST_SET_FILE_LIST:
#    feature_batch, label_batch = input_pipeline(f_in, PREPROCESSED_DATA_FOLDER)
#    print (' -> features shape {}:{}:{}'.format(len(feature_batch), 
#                                                len(feature_batch[0]), 
#                                                len(feature_batch[0][0])))
#    print (' -> labels shape   {}:{}'.format(len(label_batch), len(label_batch[0])))

# ---------------------------------------------------------------------------------------
# shuffle_train_set(): shuffle the list of batch files so that each train step
#                      receives them in a different order since the TRAIN_SET_FILE_LIST
#                      is a global
#
# parameters:      train_set - the file listing to be shuffled
#
# returns:         none
#
# -------------------------------------------------------------------------------------

def shuffle_train_set(train_set):
    sorted_file_list = random.shuffle(train_set)
    TRAIN_SET_FILE_LIST = sorted_file_list


# Unit test ---------------
# print ('Before Shuffling ->', TRAIN_SET_FILE_LIST)
# shuffle_train_set(TRAIN_SET_FILE_LIST)
# print ('After Shuffling ->', TRAIN_SET_FILE_LIST)

# ---------------------------------------------------------------------------------------
# alexnet(width, height, lr): defines the alexnet
#
# parameters:      width - width of the input image
#                  height - height of the input image
#                  lr - learning rate
#
# returns:         none
#
# -------------------------------------------------------------------------------------

def alexnet(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='features')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='momentum', loss='categorical_crossentropy',
                         learning_rate=lr, name='labels')

    model = tflearn.DNN(network, checkpoint_path=MODEL_PATH + MODEL_NAME,
                        tensorboard_dir=TRAIN_PATH, tensorboard_verbose=3, max_checkpoints=1)

    return model


# ---------------------------------------------------------------------------------------
# train_conv_net(): runs the train op
#
# parameters:      none
#
# returns:         none
#
# -------------------------------------------------------------------------------------

def train_conv_net():
    val_features = []
    val_labels = []
    val_threatfeatures = []
    val_threatlabels = []
    val_nothreatlabels = []
    val_nothreatfeatures = []
    # get train and test batches
    get_train_test_file_list()

    # instantiate model
    model = alexnet(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE)

    # read in the validation test set
    for j, test_f_in in enumerate(TEST_SET_FILE_LIST):
        if j == 0:
            val_features, val_labels, val_threatfeatures, val_threatlabels, val_nothreatfeatures, val_nothreatlabels = input_pipeline(
                test_f_in, PREPROCESSED_DATA_FOLDER)
        else:
            tmp_feature_batch, tmp_label_batch, tmp_threatfeaturebatch, tmp_threatlabelbatch, tmp_nothreatlabelbatch, tmp_nothreatfeaturebatch = input_pipeline(
                test_f_in,
                PREPROCESSED_DATA_FOLDER)
            val_features = np.concatenate((tmp_feature_batch, val_features), axis=0)
            val_labels = np.concatenate((tmp_label_batch, val_labels), axis=0)
            val_threatlabels = np.concatenate((tmp_threatlabelbatch, val_threatlabels), axis=0)
            val_threatfeatures = np.concatenate((tmp_threatfeaturebatch, val_featurebatch), axis=0)
            val_nothreatlabels = np.concatenate((tmp_nothreatlabelbatch, val_nothreatlabels), axis=0)
            val_nothreatfeatures = np.concatenate((tmp_nothreatfeaturebatch, val_nothreatfeatures), axis=0)
    val_features = val_features.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
    val_threatfeatures = val_threatfeatures.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
    val_nothreatlabels = val_nothreatlabels.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)

    # start training process
    for i in range(N_TRAIN_STEPS):

        # shuffle the train set files before each step
        shuffle_train_set(TRAIN_SET_FILE_LIST)

        # run through every batch in the training set
        for f_in in TRAIN_SET_FILE_LIST:

            # read in a batch of features and labels for training
            feature_batch, label_batch, threatfeaturebatch, threatlabelbatch, nothreatlabelbatch, nothreatfeaturebatch = input_pipeline(
                f_in, PREPROCESSED_DATA_FOLDER)
            feature_batch = feature_batch.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
            threatfeaturebatch = threatfeaturebatch.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
            nothreatfeaturebatch = nothreatfeaturebatch.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)

            # print ('Feature Batch Shape ->', feature_batch.shape)

            # run the fit operation
            # model.fit({'features': feature_batch}, {'labels': label_batch}, n_epoch=1,
            #          validation_set=({'features': val_features}, {'labels': val_labels}), 
            #          shuffle=True, snapshot_step=None, show_metric=True, 
            #          run_id=MODEL_NAME)

            # unit test -----------------------------------
            # train_conv_net()

            # Data loading and preprocessing
            # this data, which it gets from the pipeline, is only images containing threats
            X_train = threatfeaturebatch.reshape(X_train.shape[0], 224, 224, 1)
            X_test = val_threatfeatures.reshape(X_test.shape[0], 224, 224, 1)

            X_train = X_train.astype('float32')

            # Scaling the range of the image to [-1, 1]
            # Because we are using tanh as the activation function in the last layer of the generator
            # and tanh restricts the weights in the range [-1, 1]
            X_train = (X_train - 127.5) / 127.5

            X_train.shape
            # 1           #the below generator is build for size 225,224,1 sized inputs. That was our resizing issue last time
            generator = Sequential([
                Dense(128 * 7 * 7, input_dim=200, activation=LeakyReLU(0.2)),
                BatchNormalization(),
                Reshape((7, 7, 128)),
                UpSampling2D(),
                Convolution2D(250, 5, 5, border_mode='same', activation=LeakyReLU(0.2)),
                BatchNormalization(),
                UpSampling2D(),
                Convolution2D(1, 5, 5, border_mode='same', activation=LeakyReLU(0.2)),
                BatchNormalization(),
                UpSampling2D(),
                Convolution2D(1, 5, 5, border_mode='same', activation=LeakyReLU(0.2)),
                BatchNormalization(),
                UpSampling2D(),
                Convolution2D(1, 5, 5, border_mode='same', activation=LeakyReLU(0.2)),
                BatchNormalization(),
                UpSampling2D(),
                Convolution2D(1, 5, 5, border_mode='same', activation='tanh')

            ])
            # this also has to accept 224,224,1 sized input, I also added a layer to this model
            discriminator = Sequential([
                Convolution2D(128, 5, 5, subsample=(2, 2), input_shape=(224, 224, 1), border_mode='same',
                              activation=LeakyReLU(0.2)),
                Dropout(0.3),
                Convolution2D(256, 5, 5, subsample=(2, 2), border_mode='same', activation=LeakyReLU(0.2)),
                Dropout(0.3),
                Convolution2D(512, 5, 5, subsample=(2, 2), border_mode='same', activation=LeakyReLU(0.2)),
                Dropout(0.3),
                Flatten(),
                Dense(1, activation='sigmoid')
            ])
            generator.compile(loss='binary_crossentropy', optimizer=Adam())
            discriminator.compile(loss='binary_crossentropy', optimizer=Adam())
            discriminator.trainable = False
            ganInput = Input(shape=(200,))
            # getting the output of the generator
            # and then feeding it to the discriminator
            # new model = D(G(input))
            x = generator(ganInput)
            ganOutput = discriminator(x)
            gan = Model(input=ganInput, output=ganOutput)
            gan.compile(loss='binary_crossentropy', optimizer=Adam())
            gan.summary()

            def train(epoch=10, batch_size=128):
                batch_count = X_train.shape[0] // batch_size

                for i in range(epoch):
                    for j in tqdm(range(batch_count)):
                        # Input for the generator
                        noise_input = np.random.rand(batch_size, 100)

                        # getting random images from X_train of size=batch_size 
                        # these are the real images that will be fed to the discriminator
                        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]

                        # these are the predicted images from the generator
                        predictions = generator.predict(noise_input, batch_size=batch_size)

                        # the discriminator takes in the real images and the generated images
                        X = np.concatenate([predictions, image_batch])

                        # labels for the discriminator
                        y_discriminator = [0] * batch_size + [1] * batch_size

                        # Let's train the discriminator
                        discriminator.trainable = True
                        discriminator.train_on_batch(X, y_discriminator)

                        # Let's train the generator
                        noise_input = np.random.rand(batch_size, 100)
                        y_generator = [1] * batch_size
                        discriminator.trainable = False
                        gan.train_on_batch(noise_input, y_generator)
                        generator.save_weights('gen_30_scaled_images.h5')
                        discriminator.save_weights('dis_30_scaled_images.h5')
