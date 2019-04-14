import keras

from keras.datasets import mnist
from wgan import WassersteinGAN

(X, _), (_, _) = mnist.load_data()

"""
# Train from scratch.
wgan = WassersteinGAN(epochs=6000, verbose=True)
wgan.fit(X, load_models=False)
wgan.train(sample_interval=50)
wgan.sample_images(grid=[5,5])
"""

# Use Pre-Trained Models.
wgan = WassersteinGAN(verbose=True)
wgan.fit(X, load_models=True, generator_file="Trained_Models/Generator.h5", critic_file="Trained_Models/Critic.h5")
wgan.sample_images(grid=[5,5], name="Final", random=False)
