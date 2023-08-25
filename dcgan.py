import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Reshape, UpSampling2D, Conv2D, BatchNormalization, Activation, LeakyReLU, Dropout, ZeroPadding2D, Flatten
from tensorflow.keras.losses import BinaryCrossentropy

def preprocess(data, batch_size):
	data = data / 127.5 - 1.0
	data = np.expand_dims(data, axis=-1)
	data = tf.data.Dataset.from_tensor_slices(data).shuffle(60000).batch(batch_size)
	return data

def dcgan(batch_size, epochs, dataset, output):
	if dataset == "mnist":
		(data, _), (_, _) = mnist.load_data()
	elif dataset == "cifar10":
		(data, _), (_, _) = cifar10.load_data()
	elif dataset == "fashion":
		(data, _), (_, _) = fashion_mnist.load_data()
	else:
		print("[!] Dataset must be mnist/cifar10/fashion")
		sys.exit()

	data = preprocess(data, batch_size)

	generator = the_generator()
	discriminator = the_discriminator()

	discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
	generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

	cross_entropy = BinaryCrossentropy(from_logits=True)

	test_images = tf.random.normal([16, 100])

	train_dcgan(data, 100, test_images, generator, discriminator, cross_entropy, generator_optimizer, discriminator_optimizer, output)

def the_generator():
	generator = Sequential()
	generator.add(Dense(128 * 7 * 7, activation="relu", input_dim=100))
	generator.add(Reshape((7, 7, 128)))
	generator.add(UpSampling2D())
	generator.add(Conv2D(128, kernel_size=3, padding="same"))
	generator.add(BatchNormalization(momentum=0.8))
	generator.add(Activation("relu"))
	generator.add(UpSampling2D())
	generator.add(Conv2D(64, kernel_size=3, padding="same"))
	generator.add(BatchNormalization(momentum=0.8))
	generator.add(Activation("relu"))
	generator.add(Conv2D(1, kernel_size=3, padding="same"))
	generator.add(Activation("tanh"))
	return generator

def the_discriminator():
	discriminator = Sequential()
	discriminator.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding="same"))
	discriminator.add(LeakyReLU(alpha=0.2))
	discriminator.add(Dropout(0.25))
	discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
	discriminator.add(ZeroPadding2D(padding=((0, 1), (0,1))))
	discriminator.add(BatchNormalization(momentum=0.8))
	discriminator.add(LeakyReLU(alpha=0.2))
	discriminator.add(Dropout(0.25))
	discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
	discriminator.add(BatchNormalization(momentum=0.8))
	discriminator.add(LeakyReLU(alpha=0.2))
	discriminator.add(Dropout(0.25))
	discriminator.add(Flatten())
	discriminator.add(Dense(1, activation="sigmoid"))
	return discriminator

def show_generated_images(epoch, generator, test_images, output):
	generated_images = generator(test_images, training=False)

	fig = plt.figure(figsize=(10, 10))
	for i in range(generated_images.shape[0]):
		plt.subplot(4, 4, i+1)
		plt.imshow(generated_images[i,:,:,0] * 127.5 + 127.5, cmap="gray")
		plt.axis("off")

	if not os.path.exists(output):
		os.makedirs(output)

	fig.savefig(f"{output}/dcgan_{epoch}.png")
	plt.show()

def generator_loss(fake_output, cross_entropy):
	return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output, cross_entropy):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def train_step(images, generator, discriminator, cross_entropy, generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([256, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output, cross_entropy)
        disc_loss = discriminator_loss(real_output, fake_output, cross_entropy)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train_dcgan(dataset, epochs, test_images, generator, discriminator, cross_entropy, generator_optimizer, discriminator_optimizer, output):
    for epoch in range(epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator, cross_entropy, generator_optimizer, discriminator_optimizer)

        print(f"[+] Epoch {epoch + 1}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}")
        show_generated_images(epoch, generator, test_images, output)

if __name__ == "__main__":
	argument_parser = argparse.ArgumentParser(description="DCGAN")
	argument_parser.add_argument("--batch_size", required=True, type=int, help="Batch Size")
	argument_parser.add_argument("--epochs", required=True, type=int, help="Epochs")
	argument_parser.add_argument("--dataset", required=True, type=str, help="Dataset = mnist / cifar10 / fashion")
	argument_parser.add_argument("--output", required=True, type=str, help="Output folder")
	arguments = argument_parser.parse_args()

	if (True):
		dcgan(
			batch_size=arguments.batch_size,
			epochs=arguments.epochs,
			dataset=arguments.dataset,
			output=arguments.output
		)

