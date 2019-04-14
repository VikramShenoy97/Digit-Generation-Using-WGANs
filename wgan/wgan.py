import numpy as np
import keras
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv2D, LeakyReLU, Dropout, UpSampling2D, ZeroPadding2D, BatchNormalization, Flatten, Dense, Reshape, Activation

from keras import backend as K
import matplotlib.pyplot as plt

import os

class WassersteinGAN():
    def __init__(self, epochs=4000, learning_rate=0.00005, n_critic=5, clip=0.01, batch_size=64, latent_dimension=100, verbose=False):
        """
        Wasserstein Generative Adversarial Networks.

        Parameters
        ----------
        epochs : int, Optional (default = 4000)
            Number of epochs to train both networks.

        learning_rate : int, Optional (default  = 0.00005)
            Rate at which the networks learn.
            The default value is maintained as mentioned in the paper for best
            results.

        n_critic : int, Optional (default = 5)
            Number of iterations for training the critic network.
            The default value is maintained as mentioned in the paper for best
            results.

        clip : float, Optional (default = 0.01)
            The clip value is used to preserve the weights within a specific range,
            in order to enforce the constraints of 1-Lipschitz function.
            The default value is maintained as mentioned in the paper for best
            results.

        batch_size : int, Optional (default = 64)
            Batch size of m examples to sample from both distributions i.e. Real
            image distribution and random noise distributions.
            The default value is maintained as mentioned in the paper for best
            results.

        latent_dimension : int, Optional (default = 100)
            The number of features in the noise distribution.

        verbose : boolean, Optional (default=False)
            Controls verbosity of output:
            - False: No Output
            - True: Displays the Critic Loss and Generator Loss at every iteration.

        Attributes
        ----------
        generator_network_ : Keras Model, Input = latent_variable array,
                                          Output = Image array
            Takes as input a latent_variable (random noise)
            and generates an Image.

        critic_network_ : Keras Model, Input = Image array, Output = float
            Takes as input an image and outputs a score (in the range -1 to 1)
            indicating how "real" the image is.

        combined_network_ : Keras Model, Input = latent_variable, Output = float
            Takes as input a latent_variable (random noise) and outputs a score
            (in the range -1 to 1) indicating how "real" the image is.
        """
        self.epochs = epochs
        # Algorithm's Parameters (as mentioned in the paper).
        self.learning_rate = learning_rate
        self.n_critic = n_critic
        self.clip  = clip
        self.batch_size = batch_size

        self.latent_dimension = latent_dimension
        self.optimizer = optimizers.RMSprop(lr=learning_rate)
        self.verbose = verbose

    def fit(self, X, load_models=False, generator_file=None, critic_file=None):
        """
        Fits the generator and critic network on to the dataset of images.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        load_models : boolean, Optional (default = False)
            Controls training:
            - False: Creates models from scratch.
            - True: Loads models using pretrained generator network and critic
                    network.

        generator_file: .h5 type file, Optional (default = None)
                If load_models = True, specify a generator file to load.

        critic_file: .h5 type file, Optional (default = None)
                If load_models = True, specify a critic file to load.

		Returns
        -------
        self : object
        """
        return self._fit(X, load_models, generator_file, critic_file)

    def train(self, sample_interval=None):
        """
        Trains the Wasserstein GAN.

        Parameters
        ----------
        sample_interval: int, Optional (default = None)
                Defines an interval for storing generated images during training.

		Returns
        -------
        self : object
        """
        return self._train(sample_interval)

    def sample_images(self, grid=[5,5], name=None, random=True):
        """
        Function to sample images from trained generated network.

        Parameters
        ----------
        grid: array-like, Shape=[rows, columns], Optional (default = None)
                Defines a grid for viewing generated images in a subplot.

        name: str, Optional (default = None)
                Name for the final image generated.

        random: boolean, Optional (default = True)
                If set to true, generates an image from a new random distribution.

    	Returns
        -------
        self : object
        """
        return self._sample_images(grid, name, random)

    def _fit(self, X, load_models, generator_file, critic_file):
        """
        Loads the Generator Network, Critic Network, Combined Network, and the
        input data.
        """
        # Initialize image dimensions.
        self.load_models = load_models
        self.input_images = X
        self.input_images = np.expand_dims(self.input_images, axis=3)
        self.noise = None
        image_row = self.input_images.shape[1]
        image_column = self.input_images.shape[2]
        image_channels = self.input_images.shape[3]
        self.image_dimensions = (image_row, image_column, image_channels)


        if(self.load_models == True):
            # Load pre-trained models
            if(os.path.isfile(generator_file)):
                self.generator_network_ = load_model(generator_file, custom_objects={'_wasserstein_loss': self._wasserstein_loss})
            else:
                print "Error: Generator Network File Missing"
            if(os.path.isfile(critic_file)):
                self.critic_network_ = load_model(critic_file, custom_objects={'_wasserstein_loss': self._wasserstein_loss})
            else:
                print "Error: Critic Network File Missing"
        else:
            # Create and Compile Critic Network.
            self.critic_network_ = self._create_critic_network()
            self.critic_network_.compile(loss=self._wasserstein_loss, optimizer=self.optimizer, metrics=['accuracy'])

            # Create Generator Network
            self.generator_network_ = self._create_generator_network()

        # The Generator Network takes as input noise and ouptuts an image.
        latent_variable = Input(shape=(self.latent_dimension, ))
        generated_image =  self.generator_network_(latent_variable)

        # In the Combined Network, the critic is not trained.
        self.critic_network_.trainable=False
        # The Critic network takes as input an Image and outputs a score.
        critic_score = self.critic_network_(generated_image)

        # Create and Compile the Combined Network (Generator Network + Critic Network)
        self.combined_network_ = Model(latent_variable, critic_score)
        self.combined_network_.compile(loss=self._wasserstein_loss, optimizer=self.optimizer, metrics=['accuracy'])

    def _wasserstein_loss(self, y_true, y_prediction):
        """
        Provides an approximation of Wasserstein Metric (Earth Mover Distance).
        """
        return K.mean(y_true * y_prediction)

    def _create_critic_network(self):
        """
        Build the Critic Network which takes in an image and generates a score as
        to how real the image is.
        """
        model = Sequential()
        # Input = (28x28x1)
        model.add(Conv2D(filters=16, kernel_size=3, strides=2, padding="same"))
        # Dim = (14x14x16)
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding="same"))
        # Dim = (7x7x32)
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        # Dim = (8x8x32)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=64, kernel_size=3, strides=2, padding="same"))
        # Dim = (4x4x64)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding="same"))
        # Dim = (4x4x128)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        # Dim = (2048)
        model.add(Dense(1))
        # No activation is used since critic network outputs a single valued
        #score.
        image = Input(shape=self.image_dimensions)
        critic_score = model(image)

        model.summary()
        return Model(image, critic_score)

    def _create_generator_network(self):
        """
        Build the Generator Network which takes in a latent variable
        (noise distribution) and generates an image.
        """
        model = Sequential()
        # Inital Image dimension
        initial_row_dim = self.image_dimensions[0]/4
        initial_column_dim = self.image_dimensions[1]/4
        initial_channel_dim = 128

        model.add(Dense(initial_row_dim*initial_column_dim*initial_channel_dim, activation="relu"))
        # Dim = (2048)
        model.add(Reshape((initial_row_dim, initial_column_dim, initial_channel_dim)))
        # Dim = (7x7x128)
        model.add(UpSampling2D())
        # Dim = (14x14x128)
        model.add(Conv2D(filters=128, kernel_size=4, padding="same"))
        # Dim = (14x14x128)
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        # Dim = (28x28x128)
        model.add(Conv2D(filters=64, kernel_size=4, padding="same"))
        # Dim = (28x28x64)
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(filters=self.image_dimensions[2], kernel_size=4, padding="same"))
        # Dim = (28x28x1)
        # Since the images are normalized to the scale -1 to +1, tanh Activation
        #is used since the activation has the same range.
        model.add(Activation("tanh"))

        noise = Input(shape=(self.latent_dimension, ))
        image = model(noise)

        model.summary()
        return Model(noise, image)

    def _train(self, sample_interval):
        """
        Trains the Generator Network and the Critic Network.
        """
        self.input_images = (self.input_images.astype(np.float32) - 127.5) / 127.5

        label_real = -np.ones((self.batch_size, 1))
        label_fake = np.ones((self.batch_size, 1))

        for epoch in range(self.epochs):

            for _ in range(self.n_critic):

                # Train the Critic Network.

                # Sample minibatch of batch_size examples from data generating
                #distribution.
                index = np.random.randint(0, self.input_images.shape[0], self.batch_size)
                real_images = self.input_images[index]
                # Sample minibatch of batch_size noise samples from random noise
                #distribution.
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dimension))
                generated_images = self.generator_network_.predict(noise)

                real_image_loss = self.critic_network_.train_on_batch(real_images, label_real)
                fake_image_loss = self.critic_network_.train_on_batch(generated_images, label_fake)
                overall_critic_loss = 0.5 * np.add(real_image_loss, fake_image_loss)

                # Clip the weights inorder to enforce the constraints of 1-Lipschitz
                #function i.e. the weights of the critic network must be within
                #a certain range controlled by the hyperparameter 'clip'.

                for layer in self.critic_network_.layers:
                    weights = layer.get_weights()
                    weights = [np.clip(weight, -self.clip, self.clip) for weight in weights]
                    layer.set_weights(weights)

            generator_loss = self.combined_network_.train_on_batch(noise, label_real)

            if(self.verbose == True):
                print "Epoch: %d \tCritic Loss: %f\tGenerator Loss: %f" % (epoch, 1 - overall_critic_loss[0], 1 - generator_loss[0])

            # Sample images while training.
            if(sample_interval != None):
                if(epoch % sample_interval == 0):
                    self.sample_images(name=epoch)

        self.generator_network_.save("Trained_Models/Generator.h5")
        self.critic_network_.save("Trained_Models/Critic.h5")

    def _sample_images(self, grid, name, random):
        rows = grid[0]
        columns = grid[1]
        if(self.load_models == True and random == False):
            self.noise = np.load("Trained_Models/noise.npy")

        if(self.noise is None or random == True):
            # Define noise distribution for Sampling of Images
            assert len(grid) == 2, ("Error: Grid should have two entries: [row, column]")
            self.noise = np.random.normal(0, 1, (rows*columns, self.latent_dimension))
            np.save("Trained_Models/noise.npy", self.noise)

        generated_images = self.generator_network_.predict(self.noise)
        # Rescale images to range 0 - 1.
        generated_images = generated_images*0.5 + 0.5
        fig, axs = plt.subplots(rows, columns)
        sample = 0
        for i in range(0, rows):
            for j in range(0, columns):
                axs[i,j].imshow(generated_images[sample, :, :, 0], cmap='viridis')
                axs[i,j].axis("off")
                sample = sample + 1

        if type(name) is int:
            fig.savefig("Output/generated_%d.png" % name)
        elif type(name) is str:
            fig.savefig("Output/"+name+".png")
        else:
            fig.savefig("Output/Generated_Image_Final.png")

        plt.close()
