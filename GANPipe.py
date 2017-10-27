# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 16:13:55 2017

@author: JonStewart
"""

# import libraries
from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
import os
import re

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

import random
from timeit import default_timer as timer

import tsahelper as tsa


class GANPipe(object):

    def __init__(self, *argsDict, **kwargs):
        """
        ---------------------------------------------------------------------------------------
        Constants

        INPUT_FOLDER:                 The folder that contains the source data

        PREPROCESSED_DATA_FOLDER:     The folder that contains preprocessed .npy files

        STAGE1_LABELS:                The CSV file containing the labels by subject

        THREAT_ZONE:                  Threat Zone to train on (actual number not 0 based)
        for the above, each zone needs a separate model. This code has to be ran for each zone separately.
        so, this code, when ran for each zone, will need 17 output folders
        That means the GAN model output should have a separate file for each zone,
        in addition to that, we want to create images for each zone, which means we need a 'contains a threat'
        folder for each zone, and a 'does not contain a threat' folder for each zone.That means we will run a
        version of this script 34 times, changing the zone, and whether the image contains a threat, for each version
        of the script

        BATCH_SIZE:                   Number of Subjects per batch

        EXAMPLES_PER_SUBJECT          Number of examples generated per subject

        FILE_LIST:                    A list of the preprocessed .npy files to batch

        TRAIN_TEST_SPLIT_RATIO:       Ratio to split the FILE_LIST between train and test

        TRAIN_SET_FILE_LIST:          The list of .npy files to be used for training

        TEST_SET_FILE_LIST:           The list of .npy files to be used for testing

        IMAGE_DIM:                    The height and width of the images in pixels

        LEARNING_RATE                 Learning rate for the neural network

        N_TRAIN_STEPS                 The number of train steps (epochs) to run

        TRAIN_PATH                    Place to store the tensorboard logs

        MODEL_PATH                    Path where model files are stored

        MODEL_NAME                    Name of the model files

        LABEL_THREAT                  This is an addition to the regular script, and will alternate between 1 annd 0 for each zone, with 1 indicating that there is a threat, and 0 indicating that there is not

        THREAT_ZONE                   This number will change between 1 and 17

        THREAT_PRESENT                This is a boolean (0,1) value to toggle
        ----------------------------------------------------------------------------------------

        """
        super(GANPipe, self).__init__()
        data = {}
        if argsDict:
            data = argsDict[0]

        self.BATCH_SIZE = 1
        self.EXAMPLES_PER_SUBJECT = 17
        self.INPUT_FOLDER = 'tsa_datasets/stage1/aps'
        self.PREPROCESSED_DATA_FOLDER = 'tsa_datasets/preprocessed/'
        self.STAGE_LABELS = 'tsa_datasets/stage1_labels.csv'
        self.THREAT_ZONE = 1
        self.THREAT_PRESENT = 1
        self.LABEL_THREAT = 1
        self.BATCH_SIZE = 16
        self.EXAMPLES_PER_SUBJECT = 182

        self.FILE_LIST = []
        self.TRAIN_TEST_SPLIT_RATIO = 0.2
        self.TRAIN_SET_FILE_LIST = []
        self.TEST_SET_FILE_LIST = []

        self.IMAGE_DIM = 250
        self.LEARNING_RATE = 1e-3
        self.N_TRAIN_STEPS = 1
        self.TRAIN_PATH = 'tsa_logs/train/'
        self.MODEL_PATH = 'tsa_logs/model/'
        self.MODEL_NAME = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('alexnet-v0.1', self.LEARNING_RATE, self.IMAGE_DIM,
                                                        self.IMAGE_DIM, self.THREAT_ZONE))
        # overwrite defaults if specified in a dictionary and adds ALL keys from provided dict to class
        for attribute in data.keys():
            setattr(self, attribute, data[attribute])
        for key in kwargs:
            setattr(self, key, kwargs[key])

        print(self.info())
    # ---------------------------------------------------------------------------------------
    # preprocess_tsa_data(): preprocesses the tsa datasets
    #
    # parameters:      none
    #
    # returns:         none
    # ---------------------------------------------------------------------------------------

    def info(self):
        from pprint import pprint
        return "\nINST VARS: \n\n"  + " {}".format(pprint(self.__dict__))

    def preprocess_tsa_data(self):
        df = dict()
        # OPTION 1: get a list of all subjects for which there are labels
        # df1 = pd.read_csv(self.STAGE_LABELS)
        # print(df1)
        # df['Probability'] = self.LABEL_THREAT  # this is what is changed with each version of the script. Each zone is ran for zones with and without a threat
        # df['Subject'], df['Zone'] = df1['Id'].str.split('_', 1).str
        # self.SUBJECT_LIST = df['Subject'].unique()

        # OPTION 2: get a list of all subjects for whom there is data
        self.SUBJECT_LIST = [os.path.splitext(subject)[0] for subject in os.listdir(self.INPUT_FOLDER)]

        # OPTION 3: get a list of subjects for small bore test purposes
        # self.SUBJECT_LIST = ['00360f79fd6e02781457eda48f85da90','0043db5e8c819bffc15261b1f1ac5e42',
        #                '0050492f92e22eed3474ae3a6fc907fa','006ec59fa59dd80a64c85347eef810c7',
        #                '0097503ee9fa0606559c56458b281a08','011516ab0eca7cad7f5257672ddde70e']

        # intialize tracking and saving items
        batch_num = 1
        threat_zone_examples = []
        start_time = timer()

        for subject in self.SUBJECT_LIST:

            # read in the images
            print('--------------------------------------------------------------')
            print('t+> {:5.3f} |Reading images for subject #: {}'.format(timer() - start_time,
                                                                         subject))
            print('--------------------------------------------------------------')
            images = tsa.read_data(self.INPUT_FOLDER + '/' + subject + '.aps')

            # transpose so that the slice is the first dimension shape(16, 620, 512)
            images = images.transpose()

            # for each threat zone, loop through each image, mask off the zone and then crop it
            for tz_num, threat_zone_x_crop_dims in enumerate(zip(tsa.zone_slice_list,
                                                                 tsa.zone_crop_list)):

                threat_zone = threat_zone_x_crop_dims[0]
                crop_dims = threat_zone_x_crop_dims[1]

                # get label
                label = np.array(tsa.get_subject_zone_label(tz_num,
                                                            tsa.get_subject_labels(self.STAGE_LABELS, subject)))

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
            if ((len(threat_zone_examples) % (self.BATCH_SIZE * self.EXAMPLES_PER_SUBJECT)) == 0):
                for tz_num, tz in enumerate(tsa.zone_slice_list):
                    tz_examples_to_save = []

                    # write out the batch and reset
                    print(' -> writing: ' + self.PREPROCESSED_DATA_FOLDER +
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
                    np.save(self.PREPROCESSED_DATA_FOLDER +
                            'preprocessed_TSA_scans-tz{}-{}-{}-{}-b{}.npy'.format(tz_num + 1,
                                                                                  self.LABEL_THREAT,
                                                                                  # I added this to further specify that the preprocessing folder subdivide by whether a threat is present or not, not 100% sure this is right?
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
        if (len(threat_zone_examples) > 0):
            for tz_num, tz in enumerate(tsa.zone_slice_list):
                tz_examples_to_save = []

                # write out the batch and reset
                print(' -> writing: ' + self.PREPROCESSED_DATA_FOLDER
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
                np.save(self.PREPROCESSED_DATA_FOLDER +
                        'preprocessed_TSA_scans-tz{}-{}-{}-{}-b{}.npy'.format(tz_num + 1,
                                                                              # changed the number of empty brackets to additionally specify whether threat is present
                                                                              self.LABEL_THREAT,
                                                                              # also changed here, same as above
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

    def get_train_test_file_list(self):
        if not os.listdir(self.PREPROCESSED_DATA_FOLDER):
            print('No preprocessed data available.  Skipping preprocessed data setup..')
        else:
            FILE_LIST = []
            for f in os.listdir(self.PREPROCESSED_DATA_FOLDER):
                if re.search(re.compile('-tz' + str(self.THREAT_ZONE) + str(self.THREAT_PRESENT) + '-'), f):
                    FILE_LIST.append(f)
            train_test_split = len(FILE_LIST) - \
                               max(int(len(FILE_LIST) * self.TRAIN_TEST_SPLIT_RATIO), 1)
            self.TRAIN_SET_FILE_LIST = FILE_LIST[:train_test_split]
            self.TEST_SET_FILE_LIST = FILE_LIST[train_test_split:]
            print('Train/Test Split -> {} file(s) of {} used for testing'.format(
                len(FILE_LIST) - train_test_split, len(FILE_LIST)))


    # unit test ----------------------------
    # get_train_test_file_list()

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
    def input_pipeline(self, filename, path, test=False):
        if not filename:
            if not any(self.TRAIN_SET_FILE_LIST):
                self.get_train_test_file_list()

            filename = self.TRAIN_SET_FILE_LIST[0]

        if not path:
            path = self.PREPROCESSED_DATA_FOLDER

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
                if example[1][0] == 1:
                    feature_batch_with_threat.append(example[0])
                    label_batch_with_threat.append(example[1])
                else:
                    feature_batch_wo_threat.append(example[0])
                    label_batch_wo_threat.append(example[1])

        feature_batch = np.asarray(feature_batch, dtype=np.float32)
        label_batch = np.asarray(label_batch, dtype=np.float32)
        feature_batch_with_threat = np.asarray(feature_batch_with_threat, dtype=np.float32)
        label_batch_with_threat = np.asarray(label_batch_with_threat, dtype=np.float32)
        feature_batch_wo_threat = np.asarray(feature_batch_wo_threat, dtype=np.float32)
        label_batch_wo_threat = np.asarray(label_batch_wo_threat, dtype=np.float32)


        # unit test ------------------------------------------------------------------------
        if test:
            if not self.TRAIN_SET_FILE_LIST:
                self.get_train_test_file_list()

            print('Train Set -----------------------------')
            for f_in in self.TRAIN_SET_FILE_LIST:
                feature_batch, label_batch, feature_batch_with_threat, label_batch_with_threat, \
                    label_batch_wo_threat, feature_batch_wo_threat = self.input_pipeline(f_in, self.PREPROCESSED_DATA_FOLDER)
                print(' -> features shape {}:{}:{}'.format(len(feature_batch),
                                                           len(feature_batch[0]),
                                                           len(feature_batch[0][0])))
                print(' -> features with threat shape {}:{}:{}'.format(len(feature_batch_with_threat),
                                                           len(feature_batch_with_threat[0]),
                                                           len(feature_batch_with_threat[0][0])))
                print(' -> features w/o shape {}:{}:{}'.format(len(feature_batch_wo_threat),
                                                           len(feature_batch_wo_threat[0]),
                                                           len(feature_batch_wo_threat[0][0])))
                print(' -> labels shape   {}:{}'.format(len(label_batch), len(label_batch[0])))
                print(' -> labels with threat shape   {}:{}'.format(len(label_batch_with_threat), len(label_batch_with_threat[0])))
                print(' -> labels w/o threat shape   {}:{}'.format(len(label_batch_wo_threat),
                                                                    len(label_batch_wo_threat[0])))
            print('Test Set -----------------------------')
            for f_in in self.TEST_SET_FILE_LIST:
                feature_batch, label_batch, feature_batch_with_threat, label_batch_with_threat, label_batch_wo_threat, \
                    feature_batch_wo_threat = self.input_pipeline(f_in, self.PREPROCESSED_DATA_FOLDER)
                print(' -> features shape {}:{}:{}'.format(len(feature_batch),
                                                           len(feature_batch[0]),
                                                           len(feature_batch[0][0])))
                print(' -> features with threat shape {}:{}:{}'.format(len(feature_batch_with_threat),
                                                                       len(feature_batch_with_threat[0]),
                                                                       len(feature_batch_with_threat[0][0])))
                print(' -> features w/o shape {}:{}:{}'.format(len(feature_batch_wo_threat),
                                                               len(feature_batch_wo_threat[0]),
                                                               len(feature_batch_wo_threat[0][0])))
                print(' -> labels shape   {}:{}'.format(len(label_batch), len(label_batch[0])))
                print(' -> labels with threat shape   {}:{}'.format(len(label_batch_with_threat),
                                                                    len(label_batch_with_threat[0])))
                print(' -> labels w/o threat shape   {}:{}'.format(len(label_batch_wo_threat),
                                                                   len(label_batch_wo_threat[0])))
        return feature_batch, label_batch, feature_batch_with_threat, label_batch_with_threat, label_batch_wo_threat, feature_batch_wo_threat

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

    def shuffle_train_set(self, train_set, test=False):
        if not self.TRAIN_SET_FILE_LIST:
            self.get_train_test_file_list()
        else:    
            sorted_file_list = random.sample(train_set, len(train_set))
            self.TRAIN_SET_FILE_LIST = sorted_file_list

        # Unit test ---------------
        if test:
            print('Before Shuffling ->', self.TRAIN_SET_FILE_LIST)
            self.shuffle_train_set(self.TRAIN_SET_FILE_LIST)
            print('After Shuffling ->', self.TRAIN_SET_FILE_LIST)

    # now, instead of the Alexnet, which just predicts whether a threat is present, we want to use a GAN model.
    # We need to declare the model, train it, then use the trained model to generate data, which we will save in a spearate
    # folder specified by GAN ZONE THREAT
    # from here, we would import the GAN model, train it, and then generate samples


if __name__ == '__main__':
    # gp = GANPipe({'INPUT_FOLDER': '/test', "tripping": 'trump'})
    gp = GANPipe()
    gp.input_pipeline(None, None, True)
