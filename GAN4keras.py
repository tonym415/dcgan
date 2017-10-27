# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 04:40:11 2017
Code builds off of GAN pipeline. Built to process 224,224,1 sized images. 

Code borrowed from: https://github.com/ilguyi/dcgan.tensorflow/issues/1
and: https://github.com/yashk2810/DCGAN-Keras/blob/master/DCGAN.ipynb
@author: JonStewart
"""

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
