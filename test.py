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

# Use Pre-Trained Models.
wgan = WassersteinGAN(verbose=True)
wgan.fit(X, load_models=True, generator_file="Trained_Models/Generator.h5", critic_file="Trained_Models/Critic.h5")
wgan.sample_images(grid=[9,9], name="Final", random=False)
"""

wgan = WassersteinGAN(verbose=True)
for i in range(0, 93000, 3000):
    if(i == 0):
        wgan.fit(X, load_models=False)
        wgan.sample_images(grid=[9,9], name=i, random=False)
    else:
        wgan.fit(X, load_models=True, generator_file="Trained_Models/Backup/Generator_"+str(i)+".h5", critic_file="Trained_Models/Backup/Critic_"+str(i)+".h5")
        wgan.sample_images(grid=[9,9], name=i, random=False)
        
wgan.fit(X, load_models=True, generator_file="Trained_Models/Generator.h5", critic_file="Trained_Models/Critic.h5")
wgan.sample_images(grid=[9,9], name="Final", random=False)
