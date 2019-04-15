# Digit Generation using Wasserstein GANs

Wasserstein Generative Adversarial Networks implemented using Keras.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

For using this implementation of WGANs, you just need to install Keras.

```
pip install keras
```

### Dataset

The ![MNIST Dataset](http://yann.lecun.com/exdb/mnist/) is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.

### Pre-trained Models

I have stored pre-trained Generator and Critic Networks (Trained for 69000 epochs) along with the noise distribution , from which digits are sampled from, in the **Trained_Models** folder. 

Within the same folder is another folder called **Backup**. This folder stores the pretrained Generator Network and Crtic Network across different epochs as *Generator_epoch.h5* and *Critic_epoch.h5*

*Example*

```
Generator_3000.h5 -> Generator Network trained for 3000 epochs.
Critic_3000.h5 -> Critic Network trained for 3000 epochs.
```

### Run

Run the script *test.py* in the terminal as follows.

```
Python test.py
```

## Results

I ran this program on Google Colab to get better results. 

3000 epochs take approximately 15 minutes on Google Colab using their GPU.

### Transition through epochs (Interval = 3000 epochs)

![Transition](https://github.com/VikramShenoy97/Digit-Generation-using-WGANs/blob/master/Transition/wgan.gif)

### FInal Result

![Final_Image](https://github.com/VikramShenoy97/Digit-Generation-using-WGANs/blob/master/Output/Final.png)

## Built With

* [Keras](https://keras.io) - Deep Learning Framework
* [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb) - Cloud Service


## Authors

* **Vikram Shenoy** - *Initial work* - [Vikram Shenoy](https://github.com/VikramShenoy97)

## Acknowledgments

* Project is based on **Martin Arjovsky's** paper, [*Wasserstein GAN*](https://arxiv.org/abs/1701.07875)
* Project is inspired by **Jonathan Hui's** blog, [*GAN -  WGAN and WGAN-GP*](https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490)
